"""INT8 one-to-zero faults. Run from a NEW notebook cell; originals are read only."""
import ast
import json
import os
import subprocess
from pathlib import Path
from collections import deque

FAULT_CODE = r'''
import numpy as np
from optimum.quanto import quantization_map, qint8
import gc

def uniform_subset(n, k, seed):
    rng = np.random.default_rng(seed)
    selected = np.unique(rng.integers(0, n, size=k, dtype=np.int64))
    while selected.size < k:
        selected = np.union1d(selected, rng.integers(0, n, size=k-selected.size, dtype=np.int64))
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
    assert config and all(v == {'weights':'qint8', 'activations':'none'} for v in config.values())
    popcount = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    nth_bit = np.zeros((256, 8), dtype=np.uint8)
    for value in range(256):
        ones = [bit for bit in range(8) if value & (1 << bit)]
        nth_bit[value, :len(ones)] = ones
    payloads, seen = [], set()
    total_bits, total_ones = 0, 0
    for name in sorted(config):
        module = model.get_submodule(name)
        w = module.weight
        raw = w._data
        assert module.frozen and w.qtype == qint8 and raw.dtype == torch.int8, name
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
    # Denominator = ALL INT8 weight bits, not only the bits initially equal to 1.
    k = (total_bits * 3 + 500) // 1000
    assert 0 < k <= total_ones
    # Each existing 1-bit receives one rank in a global concatenation.
    # Uniform sampling of ranks is uniform sampling among eligible 1-bits.
    chosen = uniform_subset(total_ones, k, seed)
    offset, verified = 0, 0
    with torch.no_grad():
        for name, raw, expected_ones in payloads:
            cpu = raw.detach().cpu().contiguous().numpy().view(np.uint8).reshape(-1).copy()
            original = cpu.copy()
            matrix_start = offset
            matrix_changed = 0
            for start in range(0, cpu.size, 1048576):
                before = cpu[start:start+1048576]
                count = int(popcount[before].sum(dtype=np.int64))
                lo, hi = np.searchsorted(chosen, [offset, offset+count])
                if hi > lo:
                    cpu[start:start+1048576] = clear_chunk(before, chosen[lo:hi]-offset, popcount, nth_bit)
                matrix_changed += int(hi-lo)
                offset += count
            assert offset-matrix_start == expected_ones
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
    return k, total_bits

import sys
seed = int(sys.argv[1])
original_loader = load_quantized_model
def load_quantized_model(bits):
    assert bits == 8
    original_loader(bits)
    changed, total = inject_one_to_zero(seed)
    print(f'ONEZERO_BITS|{changed}/{total}', flush=True)
    def finite_check(module, args, output):
        if not torch.isfinite(output.logits[:, -1, :]).all().item():
            raise RuntimeError('Nonfinite logits; score is invalid.')
    model.register_forward_hook(finite_check)
'''

def run(notebook, gpu='0', seeds=(2026091201,2026091202,2026091203,2026091204,2026091205)):
    root = Path('/tf/wjy')
    labels = {'MMUL.ipynb':'MMLU', 'humaneval.ipynb':'HumanEval', 'mathQA.ipynb':'MathQA'}
    assert notebook in labels
    label = labels[notebook]
    nb = json.loads((root/notebook).read_text(encoding='utf-8'))
    candidates = []
    for cell in nb['cells']:
        if cell.get('cell_type') != 'code':
            continue
        try:
            outer = ast.parse(''.join(cell.get('source', [])))
        except SyntaxError:
            continue
        for node in outer.body:
            if not (isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='quant_code' for t in node.targets)):
                continue
            if not (isinstance(node.value,ast.Constant) and isinstance(node.value.value,str)):
                continue
            inner = ast.parse(node.value.value)
            names = {n.name for n in inner.body if isinstance(n,ast.FunctionDef)}
            if 'load_quantized_model' in names and label+' |' in node.value.value:
                candidates.append(inner)
    assert len(candidates)==1, 'Expected one original Quanto INT8 cell. Save notebook first.'
    tree = candidates[0]
    entry = tree.body.pop()
    assert isinstance(entry,ast.Try)
    calls = [n for n in ast.walk(entry) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='load_quantized_model']
    assert len(calls)==1 and len(calls[0].args)==1 and ast.literal_eval(calls[0].args[0])==8
    program = ast.unparse(tree)+'\n'+FAULT_CODE+'\n'+ast.unparse(entry)
    compile(program, '<INT8_1to0>', 'exec')
    python = root/'.venv-quant/bin/python'
    assert python.is_file()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONWARNINGS='ignore', USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')
    env['PATH'] = str(python.parent)+os.pathsep+env.get('PATH','')
    for key, folder in [('TMPDIR','.tmp'),('HF_HOME','.cache/huggingface'),('TORCH_HOME','.cache/torch'),('TORCH_EXTENSIONS_DIR','.cache/torch_extensions')]:
        path = root/folder
        path.mkdir(parents=True,exist_ok=True)
        env[key] = str(path)
    # Avoid accidentally running these new experiments concurrently on one GPU.
    import fcntl
    with (root/'.tmp'/f'int8_1to0_gpu{gpu}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another new 1->0 experiment is using this GPU; wait until it finishes.')
        for trial, seed in enumerate(seeds,1):
            tail, results, bits = deque(maxlen=25), [], None
            with subprocess.Popen([str(python),'-u','-c',program,str(seed)],cwd=str(root),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
                try:
                    for line in process.stdout:
                        tail.append(line.rstrip())
                        if line.startswith('ONEZERO_BITS|'):
                            bits = line.strip().split('|',1)[1]
                        elif line.startswith(label+' |'):
                            results.append(line.strip())
                    rc = process.wait()
                except BaseException:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    raise
            if rc or len(results)!=1 or bits is None:
                raise RuntimeError('1->0 evaluation failed:\n'+'\n'.join(tail))
            print(f'{label} | INT8 1->0 | BER=0.003 (all INT8 bits) | round {trial}/{len(seeds)} | seed={seed} | changed={bits} | '+results[0].split('|',1)[1].strip(),flush=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook', required=True, choices=['MMUL.ipynb','humaneval.ipynb','mathQA.ipynb'])
    parser.add_argument('--gpu',default='0')
    args = parser.parse_args()
    run(args.notebook,args.gpu)
