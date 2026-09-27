from __future__ import annotations

import ast
import datetime as dt
import json
import os
import subprocess
import textwrap
import hashlib
from pathlib import Path

ROOT = Path('/tf/wjy')
SPECC = ROOT / 'specc_INT8'
# Reuse the validated HumanEval worker.  It uses requantize() with the saved
# Quanto quantization state instead of quantizing an already-quantized model.
BASE = ROOT / 'srlr_INT8' / 'results_int8_humaneval_lsb_msb' / 'HumanEval_2026091201_worker.py'
# Keep the failed double-quantized attempt for auditability; write the
# corrected requantize run to a separate, unambiguous result directory.
OUT = SPECC / 'results_specc_int8_humaneval_requantized'
GPU = os.environ.get('SPECC_GPU', '1')
SEEDS = (2026091201, 2026091202, 2026091203, 2026091204, 2026091205)
TARGET_ONE_FLIP_RATE = 0.003

INJECT_SOURCE = r'''
def inject_one_to_zero(seed):
    """Protect INT8[7:4] with Hamming(7,4), inject over all 7+4 stored bits."""
    import os
    import json
    mode = os.environ.get('SPECC_MODE', 'fault')
    assert mode in ('clean', 'roundtrip', 'fault')
    target_one_flip_rate = 0.003 if mode == 'fault' else 0.0
    config = quantization_map(model)
    expected = {f'model.layers.{i}.{part}' for i in range(32) for part in (
        'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
        'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')}
    assert set(config) == expected and len(config) == 224, (len(config), set(config) ^ expected)
    assert all(v == {'weights': 'qint8', 'activations': 'none'} for v in config.values())
    if mode == 'clean':
        print('SPECC_CLEAN|injection_bypassed=true', flush=True)
        return 0, 1

    def encode_nibble(n):
        d0 = n & np.uint8(1)
        d1 = (n >> np.uint8(1)) & np.uint8(1)
        d2 = (n >> np.uint8(2)) & np.uint8(1)
        d3 = (n >> np.uint8(3)) & np.uint8(1)
        return (d0 ^ d1 ^ d3) | ((d0 ^ d2 ^ d3) << np.uint8(1)) | (d0 << np.uint8(2)) | ((d1 ^ d2 ^ d3) << np.uint8(3)) | (d1 << np.uint8(4)) | (d2 << np.uint8(5)) | (d3 << np.uint8(6))

    def decode(cw):
        s1 = ((cw >> np.uint8(0)) ^ (cw >> np.uint8(2)) ^ (cw >> np.uint8(4)) ^ (cw >> np.uint8(6))) & np.uint8(1)
        s2 = (((cw >> np.uint8(1)) ^ (cw >> np.uint8(2)) ^ (cw >> np.uint8(5)) ^ (cw >> np.uint8(6))) & np.uint8(1)) << np.uint8(1)
        s4 = (((cw >> np.uint8(3)) ^ (cw >> np.uint8(4)) ^ (cw >> np.uint8(5)) ^ (cw >> np.uint8(6))) & np.uint8(1)) << np.uint8(2)
        syndrome = s1 | s2 | s4
        for err in range(1, 8):
            m = syndrome == np.uint8(err)
            if np.any(m):
                cw[m] ^= np.uint8(1 << (err - 1))
        d0 = (cw >> np.uint8(2)) & np.uint8(1)
        d1 = (cw >> np.uint8(4)) & np.uint8(1)
        d2 = (cw >> np.uint8(5)) & np.uint8(1)
        d3 = (cw >> np.uint8(6)) & np.uint8(1)
        return d0 | (d1 << np.uint8(1)) | (d2 << np.uint8(2)) | (d3 << np.uint8(3)), int((syndrome != 0).sum())

    pop = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    specs = []
    total_physical_bits = 0
    total_ones = 0
    for name in sorted(config):
        module = model.get_submodule(name)
        weight = module.weight
        assert weight.qtype == qint8, name
        raw_t = getattr(weight, '_data', weight)
        assert raw_t.dtype in (torch.int8, torch.uint8), (name, raw_t.dtype)
        raw = np.ascontiguousarray(raw_t.detach().cpu().numpy().view(np.uint8).reshape(-1))
        hi = encode_nibble(raw >> np.uint8(4))
        lo = raw & np.uint8(15)
        decoded, syndromes = decode(hi.copy())
        assert syndromes == 0 and np.array_equal((decoded << np.uint8(4)) | lo, raw), ('zero_fault_roundtrip', name)
        hi_ones = int(pop[hi].sum(dtype=np.int64))
        lo_ones = int(pop[lo].sum(dtype=np.int64))
        specs.append((name, tuple(raw_t.shape), raw_t.device, raw.size, hi_ones, lo_ones))
        total_physical_bits += int(raw.size * 11)
        total_ones += hi_ones + lo_ones

    k = int(round(total_ones * target_one_flip_rate))
    assert k <= total_ones, (k, total_ones)
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(total_ones, size=k, replace=False).astype(np.int64))
    print(f'SPECC_LAYOUT|protected_bits=INT8[7:4]|unprotected_bits=INT8[3:0]|physical_bits={total_physical_bits}|physical_ones={total_ones}|target_flips={k}|seed={seed}', flush=True)
    cursor = 0
    changed = 0
    corrected = 0
    changed_hi = 0
    changed_lo = 0
    multi_error_codewords = 0
    residual_hi_bits = 0
    residual_lo_bits = 0
    residual_bytes = 0
    for name, shape, device, nbytes, hi_ones, lo_ones in specs:
        module = model.get_submodule(name)
        weight = module.weight
        raw_t = getattr(weight, '_data', weight)
        raw = np.ascontiguousarray(raw_t.detach().cpu().numpy().view(np.uint8).reshape(-1))
        hi = encode_nibble(raw >> np.uint8(4))
        lo = raw & np.uint8(15)
        hi_before = hi.copy()
        lo_before = lo.copy()
        local_lo = np.searchsorted(chosen, cursor, side='left')
        local_hi = np.searchsorted(chosen, cursor + hi_ones + lo_ones, side='left')
        local_rank = chosen[local_lo:local_hi] - cursor
        cursor += hi_ones + lo_ones
        hi_rank = local_rank[local_rank < hi_ones]
        lo_rank = local_rank[local_rank >= hi_ones] - hi_ones
        if hi_rank.size:
            bits = np.unpackbits(hi, bitorder='little').reshape(-1, 8)[:, :7].ravel()
            positions = np.flatnonzero(bits)[hi_rank]
            np.bitwise_and.at(hi, positions // 7, np.uint8(255) ^ (np.uint8(1) << (positions % 7).astype(np.uint8)))
        if lo_rank.size:
            bits = np.unpackbits(lo, bitorder='little').reshape(-1, 8)[:, :4].ravel()
            positions = np.flatnonzero(bits)[lo_rank]
            np.bitwise_and.at(lo, positions // 4, np.uint8(255) ^ (np.uint8(1) << (positions % 4).astype(np.uint8)))
        assert not np.any(hi & ~hi_before) and not np.any(lo & ~lo_before), ('zero_to_one', name)
        actual_hi = int(pop[hi_before ^ hi].sum(dtype=np.int64))
        actual_lo = int(pop[lo_before ^ lo].sum(dtype=np.int64))
        assert actual_hi == hi_rank.size and actual_lo == lo_rank.size, ('actual_flip_count', name)
        changed_hi += actual_hi
        changed_lo += actual_lo
        hi_faults = pop[hi_before ^ hi]
        multi_error_codewords += int(np.count_nonzero(hi_faults >= 2))
        dhi, c2 = decode(hi)
        assert np.array_equal(dhi[hi_faults <= 1], (raw >> np.uint8(4))[hi_faults <= 1]), ('single_error_recovery', name)
        restored = (lo | (dhi << np.uint8(4))).view(np.int8).reshape(shape)
        delta = raw ^ restored.view(np.uint8).reshape(-1)
        residual_hi_bits += int(pop[delta >> np.uint8(4)].sum(dtype=np.int64))
        residual_lo_bits += int(pop[delta & np.uint8(15)].sum(dtype=np.int64))
        residual_bytes += int(np.count_nonzero(delta))
        if mode == 'roundtrip':
            assert not np.any(delta), ('roundtrip_weights_changed', name)
        corrected += c2
        changed += actual_hi + actual_lo
        with torch.no_grad():
            raw_t.copy_(torch.from_numpy(restored).to(device=device, dtype=raw_t.dtype))
        readback = raw_t.detach().cpu().numpy().view(np.uint8).reshape(-1)
        assert np.array_equal(readback, restored.view(np.uint8).reshape(-1)), ('writeback_mismatch', name)

    assert cursor == total_ones and changed == k, (cursor, total_ones, changed, k)
    probe = np.arange(256, dtype=np.uint8)
    probe_hi = encode_nibble(probe >> np.uint8(4))
    for i in range(probe_hi.size):
        for bit in range(7):
            if (probe_hi[i] >> np.uint8(bit)) & np.uint8(1):
                trial = probe_hi.copy()
                trial[i] &= np.uint8(255) ^ (np.uint8(1) << np.uint8(bit))
                decoded, _ = decode(trial)
                assert decoded[i] == (probe[i] >> np.uint8(4)), (i, bit)
    print('SPECC_SELFTEST|hamming74_int8_high4_single_bit=PASS|low4_unprotected=PASS', flush=True)
    print(f'SPECC_VERIFY|changed={changed}|physical_ber={changed / total_physical_bits:.12f}|changed_hi={changed_hi}|changed_lo={changed_lo}|corrected_codewords={corrected}|multi_error_codewords={multi_error_codewords}', flush=True)
    stats = dict(mode=mode, selected_bits=k, actual_flips=changed, initial_one_bits=total_ones,
                 physical_bits=total_physical_bits, target_one_flip_rate=target_one_flip_rate,
                 actual_one_flip_rate=changed / total_ones, actual_physical_ber=changed / total_physical_bits,
                 changed_hi=changed_hi, changed_lo=changed_lo, syndrome_actions=corrected,
                 multi_error_codewords=multi_error_codewords, residual_hi_bits=residual_hi_bits,
                 residual_lo_bits=residual_lo_bits, residual_bytes=residual_bytes)
    assert residual_lo_bits == changed_lo
    print('SPECC_STATS|' + json.dumps(stats), flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    return changed, total_physical_bits
'''


def make_worker() -> str:
    tree = ast.parse(BASE.read_text(encoding='utf-8'))
    replacement = ast.parse(textwrap.dedent(INJECT_SOURCE)).body[0]
    found = False
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.FunctionDef) and node.name == 'inject_one_to_zero':
            tree.body[i] = replacement
            found = True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'MODEL_PATH':
                    node.value = ast.parse("ROOT / 'models/Meta-Llama-3-8B-Instruct-Quanto-INT8-clean'").body[0].value
    assert found, 'inject_one_to_zero not found in base worker'
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + '\n'


def main() -> None:
    # The lock is retained for the lifetime of the controller on the Linux host.
    import fcntl
    lock = (SPECC / 'humaneval_v3.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu_state = subprocess.check_output(['nvidia-smi', '-i', GPU,
        '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
    free_mb, util = [int(x.strip()) for x in gpu_state.split(',')]
    gpu_processes = subprocess.check_output(['nvidia-smi', '-i', GPU,
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if free_mb < 30720 or util > 10 or gpu_processes:
        raise RuntimeError(f'GPU {GPU} is occupied or insufficient: {gpu_state}; pids={gpu_processes}')
    OUT.mkdir(parents=True, exist_ok=True)
    worker = make_worker()
    compile(worker, '<generated-worker>', 'exec')
    manifest = dict(controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    base_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
                    worker_sha256=hashlib.sha256(worker.encode()).hexdigest(),
                    target_one_flip_rate=TARGET_ONE_FLIP_RATE, physical_ber='measured separately',
                    gpu=GPU, baseline_gate='164 questions, clean accuracy 0.55 to 0.66; roundtrip result equals clean')
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    summary = []
    stages = [('clean', SEEDS[0]), ('roundtrip', SEEDS[0])] + [('fault', s) for s in SEEDS]
    baseline = None
    for mode, seed in stages:
        tag = f'HumanEval_{seed}' if mode == 'fault' else f'HumanEval_{mode}'
        result_path = OUT / f'{tag}.json'
        log_path = OUT / f'{tag}.log'
        worker_path = OUT / f'{tag}_worker.py'
        if result_path.exists():
            raise RuntimeError(f'Existing result requires review before rerun: {result_path}')
        worker_path.write_text(worker, encoding='utf-8')
        env = os.environ.copy()
        env['SPECC_MODE'] = mode
        env.update({'CUDA_VISIBLE_DEVICES': GPU, 'PYTHONWARNINGS': 'ignore', 'TOKENIZERS_PARALLELISM': 'false', 'USE_TF': '0', 'USE_FLAX': '0'})
        print(f'START SPECC-HumanEval-V3 mode={mode} seed={seed} gpu={GPU}', flush=True)
        with log_path.open('w', encoding='utf-8', buffering=1) as log:
            proc = subprocess.Popen(['/tf/wjy/.venv-quant/bin/python', '-u', str(worker_path), str(seed)], cwd=str(SPECC), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            lines = []
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line)
                lines.append(line.rstrip())
        rc = proc.wait()
        found = next((json.loads(x.split('RESULT_JSON|', 1)[1]) for x in lines if 'RESULT_JSON|' in x), None)
        stats = next((json.loads(x.split('SPECC_STATS|', 1)[1]) for x in lines if 'SPECC_STATS|' in x), None)
        if mode != 'clean' and stats is None:
            rc = rc or 1
        rec = {'status': 'completed' if rc == 0 and found else 'failed', 'seed': seed, 'gpu': int(GPU), 'returncode': rc, 'result': found, 'finished_at': dt.datetime.now(dt.timezone.utc).isoformat()}
        rec.update(mode=mode, injection=stats)
        if mode == 'clean' and found:
            baseline = found
            if not (found.get('total_questions') == 164 and 0.55 <= found.get('accuracy', 0) <= 0.66):
                rec['status'] = 'baseline_failed'
        if mode == 'roundtrip' and found != baseline:
            rec['status'] = 'roundtrip_failed'
        result_path.write_text(json.dumps(rec, indent=2), encoding='utf-8')
        summary.append(rec)
        if rec['status'] != 'completed':
            raise RuntimeError(f'SpECC HumanEval seed {seed} failed; inspect {log_path}')
    all_results = []
    for p in sorted(OUT.glob('HumanEval_*.json')):
        try:
            x = json.loads(p.read_text(encoding='utf-8'))
            if x.get('status') == 'completed' and x.get('mode') == 'fault': all_results.append(x)
        except Exception:
            pass
    acc = [x['result']['accuracy'] for x in all_results if x.get('result')]
    report = {'task': 'HumanEval', 'protection': 'Hamming(7,4)-INT8-high4-adaptation', 'protected_bits': 'INT8[7:4]', 'unprotected_bits': 'INT8[3:0]', 'physical_storage_bits_per_int8': 11, 'target_one_flip_rate': TARGET_ONE_FLIP_RATE, 'gpu': int(GPU), 'seeds': all_results, 'mean_accuracy': sum(acc)/len(acc) if acc else None, 'n': len(acc)}
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SPECC_HUMANEVAL_SUMMARY|' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
