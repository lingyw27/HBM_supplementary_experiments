# Run via %run /tf/wjy/humaneval_int8_1to0_diagnostic.py --gpu 0
# Existing cells are read only. Only INT8 weight payloads receive faults.
import ast as _fi_ast
import json as _fi_json
import os as _fi_os
import subprocess as _fi_sp
from pathlib import Path as _fi_Path
from collections import deque as _fi_deque

_fi_root = _fi_Path('/tf/wjy')
_fi_python = _fi_root / '.venv-quant/bin/python'
import argparse
_diag_parser = argparse.ArgumentParser()
_diag_parser.add_argument('--gpu', default='0', choices=['0', '1'])
_fi_gpu = _diag_parser.parse_args().gpu
assert _fi_python.is_file(), f'Python not found: {_fi_python}'

# Reuse the saved, original Quanto loader and HumanEval evaluator without executing
# the original cell's subprocess wrapper or changing its source.
_fi_nb = _fi_json.loads((_fi_root / 'humaneval.ipynb').read_text(encoding='utf-8'))
_fi_candidates = []
for _fi_cell in _fi_nb['cells']:
    _fi_source = ''.join(_fi_cell.get('source', []))
    if _fi_cell.get('cell_type') != 'code':
        continue
    try:
        _fi_outer = _fi_ast.parse(_fi_source)
    except SyntaxError:
        continue
    for _fi_node in _fi_outer.body:
        if (isinstance(_fi_node, _fi_ast.Assign)
                and any(isinstance(t, _fi_ast.Name) and t.id == 'quant_code'
                        for t in _fi_node.targets)
                and isinstance(_fi_node.value, _fi_ast.Constant)
                and isinstance(_fi_node.value.value, str)):
            _fi_inner = _fi_ast.parse(_fi_node.value.value)
            _fi_names = {n.name for n in _fi_inner.body
                         if isinstance(n, _fi_ast.FunctionDef)}
            if 'load_quantized_model' in _fi_names and 'HumanEval |' in _fi_node.value.value:
                _fi_candidates.append(_fi_inner)
assert len(_fi_candidates) == 1, 'Expected exactly one original Quanto INT8 cell; save the notebook and check its code.'
_fi_tree = _fi_candidates[0]
assert isinstance(_fi_tree.body[-1], _fi_ast.Try), 'Unexpected original INT8 cell structure.'
assert any(isinstance(n, _fi_ast.Call) and isinstance(n.func, _fi_ast.Name)
           and n.func.id == 'load_quantized_model'
           for n in _fi_ast.walk(_fi_tree.body[-1])), 'Original INT8 entry point not found.'
_fi_entry = _fi_tree.body.pop()
_fi_load_calls = [n for n in _fi_ast.walk(_fi_entry)
                  if isinstance(n, _fi_ast.Call) and isinstance(n.func, _fi_ast.Name)
                  and n.func.id == 'load_quantized_model']
assert len(_fi_load_calls) == 1, 'Expected a separate INT8 entry point.'
assert len(_fi_load_calls[0].args) == 1 and _fi_ast.literal_eval(_fi_load_calls[0].args[0]) == 8, 'Expected INT8 only.'
_fi_entry_source = _fi_ast.unparse(_fi_entry)
_fi_base = _fi_ast.unparse(_fi_tree)

# Instrument only an in-memory copy of the original evaluator.
_diag_found = 0
for _diag_fn in _fi_tree.body:
    if isinstance(_diag_fn, _fi_ast.FunctionDef) and _diag_fn.name == 'run_humaneval':
        for _diag_i, _diag_node in enumerate(_diag_fn.body):
            if (isinstance(_diag_node, _fi_ast.Assign)
                    and any(isinstance(t, _fi_ast.Name) and t.id == 'correct' for t in _diag_node.targets)):
                _diag_insert = _fi_ast.parse(r"""
test = test.select(range(min(3, len(test)))) if hasattr(test, 'select') else test[:3]
print('DIAG1TO0 | task_ids=' + repr([r['task_id'] for r in test]), flush=True)
fixture = {'task_id': 'diagnostic/add', 'prompt': 'def add(a, b):\n',
           'entry_point': 'add', 'test': 'def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0\n'}
with ThreadPoolExecutor(max_workers=1) as probe_pool:
    probe = probe_pool.submit(check_correctness, fixture, '    return a + b\n', 10.0).result()
print('DIAG1TO0 | known_correct_checker=' + repr(probe), flush=True)
assert probe['passed'], 'Checker failed on known correct code; do not interpret model scores.'
""").body
                _diag_fn.body[_diag_i:_diag_i] = _diag_insert
                _diag_found += 1
                break
assert _diag_found == 1, 'Evaluator structure differs; send the original run_humaneval function.'

class _DiagResults(_fi_ast.NodeTransformer):
    def visit_Assign(self, node):
        self.generic_visit(node)
        if any(isinstance(t, _fi_ast.Name) and t.id == 'result' for t in node.targets):
            return [node, _fi_ast.parse("print('DIAG1TO0 | checker_result=' + repr(result), flush=True)").body[0]]
        return node

_fi_tree = _DiagResults().visit(_fi_tree)
_fi_ast.fix_missing_locations(_fi_tree)
_fi_base = _fi_ast.unparse(_fi_tree)
_fi_extra = r'''

import numpy as np
import sys
def uniform_subset(n, k, seed):
    rng = np.random.default_rng(seed)
    selected = np.unique(rng.integers(0, n, size=k, dtype=np.int64))
    while selected.size < k:
        selected = np.union1d(selected, rng.integers(0, n, size=k - selected.size, dtype=np.int64))
    return selected

def clear_chunk(before, selected_ranks, popcount, nth_bit):
    counts = popcount[before]
    cumulative = np.cumsum(counts, dtype=np.int64)
    byte_ids = np.searchsorted(cumulative, selected_ranks, side='right')
    preceding = cumulative[byte_ids] - counts[byte_ids]
    within = selected_ranks - preceding
    bits = nth_bit[before[byte_ids], within]
    mask = np.zeros(before.size, dtype=np.uint8)
    np.bitwise_or.at(mask, byte_ids, np.left_shift(np.uint8(1), bits))
    after = np.bitwise_and(before, np.bitwise_not(mask))
    assert not np.any(np.bitwise_and(np.bitwise_not(before), after))
    changed = int(popcount[np.bitwise_xor(before, after)].sum(dtype=np.int64))
    assert changed == selected_ranks.size
    return after

def inject_one_to_zero(seed):
    config = quantization_map(model)
    assert config and all((v == {'weights': 'qint8', 'activations': 'none'} for v in config.values()))
    popcount = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    nth_bit = np.zeros((256, 8), dtype=np.uint8)
    for value in range(256):
        ones = [bit for bit in range(8) if value & 1 << bit]
        nth_bit[value, :len(ones)] = ones
    payloads, seen = ([], set())
    total_bits, total_ones = (0, 0)
    for name in sorted(config):
        module = model.get_submodule(name)
        w = module.weight
        raw = w._data
        assert module.frozen and w.qtype == qint8 and (raw.dtype == torch.int8), name
        assert raw.ndim == 2 and raw.numel() == w.numel(), name
        key = (str(raw.device), raw.untyped_storage().data_ptr())
        assert key not in seen, 'Shared weight storage needs explicit deduplication.'
        seen.add(key)
        cpu = raw.detach().cpu().contiguous().numpy().view(np.uint8).reshape(-1)
        ones = int(popcount[cpu].sum(dtype=np.int64))
        payloads.append((name, raw, ones))
        total_bits += raw.numel() * 8
        total_ones += ones
        del cpu
    assert len(payloads) == model.config.num_hidden_layers * 7
    k = (total_bits * 3 + 500) // 1000
    assert 0 < k <= total_ones
    chosen = uniform_subset(total_ones, k, seed)
    offset, verified = (0, 0)
    with torch.no_grad():
        for name, raw, expected_ones in payloads:
            cpu = raw.detach().cpu().contiguous().numpy().view(np.uint8).reshape(-1).copy()
            original = cpu.copy()
            matrix_start = offset
            matrix_changed = 0
            for start in range(0, cpu.size, 1048576):
                before = cpu[start:start + 1048576]
                count = int(popcount[before].sum(dtype=np.int64))
                lo, hi = np.searchsorted(chosen, [offset, offset + count])
                if hi > lo:
                    cpu[start:start + 1048576] = clear_chunk(before, chosen[lo:hi] - offset, popcount, nth_bit)
                matrix_changed += int(hi - lo)
                offset += count
            assert offset - matrix_start == expected_ones
            raw.copy_(torch.from_numpy(cpu.view(np.int8).reshape(tuple(raw.shape))).to(raw.device))
            actual = raw.detach().cpu().contiguous().numpy().view(np.uint8).reshape(-1)
            assert np.array_equal(actual, cpu), f'Write-back mismatch: {name}'
            assert not np.any(np.bitwise_and(np.bitwise_not(original), actual)), f'Unexpected 0->1: {name}'
            changed = int(popcount[np.bitwise_xor(original, actual)].sum(dtype=np.int64))
            assert changed == matrix_changed
            verified += changed
            del cpu, original, actual
    assert verified == k and offset == total_ones
    del chosen, payloads
    gc.collect()
    torch.cuda.empty_cache()
    return (k, total_bits)

def check_finite_output(module, args, output):
    # Do not report an argmax of NaNs as a valid evaluation score.
    if not torch.isfinite(output.logits[:, -1, :]).all().item():
        raise RuntimeError('Nonfinite logits after fault injection; accuracy is invalid.')

from optimum.quanto import quantization_map, qint8
import gc

import time
mode = sys.argv[1]
set_seed(42)
print(f'DIAG1TO0 | mode={mode} | fault_seed=2026091201 | loading clean INT8', flush=True)
load_quantized_model(8)
print('DIAG1TO0 | model loaded', flush=True)
try:
    if mode == 'fault':
        print('DIAG1TO0 | sampling and flipping bits', flush=True)
        k, n = inject_one_to_zero(2026091201)
        print(f'DIAG1TO0 | verified 1->0={k} | 0->1=0 (checked on every matrix) | total_INT8_bits={n} | actual_BER={k/n:.10f}', flush=True)
    hook = model.register_forward_hook(check_finite_output)
    original_generate = generate_batch
    original_extract = extract_completion

    def generate_batch(messages, max_new_tokens):
        print(f'DIAG1TO0 | generating {len(messages)} tasks, max_new_tokens={max_new_tokens}', flush=True)
        started = time.monotonic()
        output = original_generate(messages, max_new_tokens)
        print(f'DIAG1TO0 | generation finished in {time.monotonic()-started:.1f}s', flush=True)
        for i, text in enumerate(output):
            print(f'DIAG1TO0 | raw[{i}] | chars={len(text)} | preview={text!r}', flush=True)
        return output

    def extract_completion(text, row):
        value = original_extract(text, row)
        print(f"DIAG1TO0 | extracted | {row['task_id']} | chars={len(value)} | preview={value!r}", flush=True)
        return value

    correct, total = run_humaneval()
    print(f'DIAG1TO0 | mode={mode} | {correct}/{total}', flush=True)
finally:
    release_model()

'''
_fi_program = _fi_base + '\n' + _fi_extra
compile(_fi_program, '<HumanEval_diagnostic>', 'exec')
_fi_env = _fi_os.environ.copy()
_fi_env.update(CUDA_VISIBLE_DEVICES=_fi_gpu, PYTHONWARNINGS='ignore',
               USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')
_fi_env['PATH'] = str(_fi_python.parent) + _fi_os.pathsep + _fi_env.get('PATH', '')
for _fi_key, _fi_folder in [
    ('TMPDIR', '.tmp'), ('HF_HOME', '.cache/huggingface'),
    ('TORCH_HOME', '.cache/torch'), ('TORCH_EXTENSIONS_DIR', '.cache/torch_extensions')
]:
    _fi_dir = _fi_root / _fi_folder
    _fi_dir.mkdir(parents=True, exist_ok=True)
    _fi_env[_fi_key] = str(_fi_dir)


# Run only after interrupting the previous five-round cell or letting it finish.
for _diag_mode in ('clean', 'fault'):
    print(f'DIAG1TO0 | starting {_diag_mode}', flush=True)
    _fi_tail = _fi_deque(maxlen=30)
    with _fi_sp.Popen([str(_fi_python), '-u', '-c', _fi_program, _diag_mode],
                      cwd=str(_fi_root), env=_fi_env, stdout=_fi_sp.PIPE,
                      stderr=_fi_sp.STDOUT, text=True, bufsize=1) as _fi_process:
        try:
            for _fi_line in _fi_process.stdout:
                _fi_tail.append(_fi_line.rstrip())
                if _fi_line.startswith('DIAG1TO0 |'):
                    print(_fi_line, end='', flush=True)
            if _fi_process.wait() != 0:
                raise RuntimeError('Diagnostic failed:\n' + '\n'.join(_fi_tail))
        except BaseException:
            if _fi_process.poll() is None:
                _fi_process.terminate()
                try:
                    _fi_process.wait(timeout=10)
                except _fi_sp.TimeoutExpired:
                    _fi_process.kill()
                    _fi_process.wait()
            raise
