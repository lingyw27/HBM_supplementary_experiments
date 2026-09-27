import ast
import datetime
import json
import os
import subprocess
import textwrap
from pathlib import Path

ROOT = Path('/tf/wjy')
BASE = ROOT / 'results_hbm_int8_rerun_20260916/MMLU_2026091201_worker.py'
OUT = ROOT / 'srlr/results_int8_mmlu_lsb_msb'
PYTHON = ROOT / '.venv-quant/bin/python'
SEEDS = (2026091201, 2026091202, 2026091203, 2026091204, 2026091205)
TARGET_BER = 0.003
MODEL_DIR = '/tf/wjy/models/Meta-Llama-3-8B-Instruct-Quanto-INT8-clean'

OUT.mkdir(parents=True, exist_ok=True)
assert BASE.is_file(), BASE
assert PYTHON.is_file(), PYTHON


def set_assign(tree, name, value):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            node.value = ast.Constant(value=value)
            return
    raise AssertionError(name)


def make_worker():
    tree = ast.parse(BASE.read_text(encoding='utf-8'))
    set_assign(tree, 'MODEL_PATH', MODEL_DIR)
    set_assign(tree, 'GPU', 0)
    load_body = ast.parse(textwrap.dedent('''
        def load_quantized_model(bits):
            global model
            assert bits == 8
            release_model()
            set_seed(42)
            from optimum.quanto import requantize as _requantize
            from safetensors.torch import load_file as _load_file
            base_model = ROOT / 'models/Meta-Llama-3-8B-Instruct'
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                local_files_only=True,
                torch_dtype=torch.float16,
                device_map={'': 'cpu'},
                attn_implementation='sdpa'
            ).eval()
            qcfg = json.loads((Path(MODEL_PATH) / 'quantization_map.json').read_text())
            qstate = _load_file(str(Path(MODEL_PATH) / 'model.safetensors'), device='cpu')
            _requantize(model, qstate, qcfg, device=torch.device(f'cuda:{GPU}'))
            model = model.eval()
            config = quantization_map(model)
            expected = {
                f'model.layers.{i}.{part}'
                for i in range(32)
                for part in (
                    'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                    'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'
                )
            }
            assert set(config) == expected, (len(config), len(expected))
            assert all(v == {'weights': 'qint8', 'activations': 'none'} for v in config.values())
            encoded = 0
            total_values = 0
            for name in sorted(config):
                module = model.get_submodule(name)
                raw = module.weight._data
                assert module.frozen and raw.dtype == torch.int8, name
                cpu = raw.detach().cpu().contiguous().numpy().view(np.uint8).reshape(-1)
                msb = cpu >> np.uint8(7)
                encoded += int(np.count_nonzero((cpu & np.uint8(1)) != msb))
                cpu[:] = (cpu & np.uint8(0xFE)) | msb
                raw.copy_(torch.from_numpy(cpu.view(np.int8).reshape(tuple(raw.shape))).to(raw.device))
                check = raw.detach().cpu().contiguous().numpy().view(np.uint8).reshape(-1)
                assert np.all(((check >> np.uint8(7)) & np.uint8(1)) == (check & np.uint8(1))), name
                total_values += check.size
            print(f'SRLR_ENCODE|changed_lsb={encoded}|values={total_values}|rule=LSB<-MSB', flush=True)
    ''')).body
    correction = ast.parse(textwrap.dedent('''
        corrected_pairs = 0
        corrected_values = 0
        for name in sorted(config):
            module = model.get_submodule(name)
            raw = module.weight._data
            cpu = raw.detach().cpu().contiguous().numpy().view(np.uint8).reshape(-1)
            mismatch = (((cpu >> np.uint8(7)) & np.uint8(1)) ^ (cpu & np.uint8(1))) != 0
            count = int(np.count_nonzero(mismatch))
            if count:
                cpu[mismatch] = cpu[mismatch] | np.uint8(0x81)
                raw.copy_(torch.from_numpy(cpu.view(np.int8).reshape(tuple(raw.shape))).to(raw.device))
            check = raw.detach().cpu().contiguous().numpy().view(np.uint8).reshape(-1)
            assert not np.any((((check >> np.uint8(7)) & np.uint8(1)) ^ (check & np.uint8(1))) != 0), name
            corrected_pairs += count
            corrected_values += count
        print(f'SRLR_CORRECT|mismatch_pairs={corrected_pairs}|restored_values={corrected_values}|rule=mismatch->11', flush=True)
    ''')).body
    replaced_load = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'load_quantized_model' and not replaced_load:
            node.body = load_body[0].body
            replaced_load = True
        if isinstance(node, ast.FunctionDef) and node.name == 'inject_one_to_zero':
            for idx, statement in enumerate(node.body):
                if isinstance(statement, ast.Return):
                    node.body[idx:idx] = correction
                    break
    ast.fix_missing_locations(tree)
    program = ast.unparse(tree)
    compile(program, '<srlr_mmlu_worker>', 'exec')
    return program

worker_program = make_worker()
for trial, seed in enumerate(SEEDS, 1):
    result_path = OUT / f'MMLU_{seed}.json'
    if result_path.exists() and json.loads(result_path.read_text()).get('status') == 'completed':
        print('SKIP_COMPLETED', seed, flush=True)
        continue
    worker_path = OUT / f'MMLU_{seed}_worker.py'
    log_path = OUT / f'MMLU_{seed}.log'
    worker_path.write_text(worker_program, encoding='utf-8')
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES='0', PYTHONWARNINGS='ignore', USE_TF='0',
        USE_FLAX='0', TOKENIZERS_PARALLELISM='false'
    )
    env['PATH'] = str(PYTHON.parent) + os.pathsep + env.get('PATH', '')
    for key, folder in (
        ('TMPDIR', '.tmp'), ('HF_HOME', '.cache/huggingface'),
        ('TORCH_HOME', '.cache/torch'), ('TORCH_EXTENSIONS_DIR', '.cache/torch_extensions')
    ):
        path = ROOT / folder
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    record = {
        'task': 'MMLU', 'seed': seed, 'round': trial, 'status': 'running',
        'model': MODEL_DIR, 'gpu': 0, 'target_ber': TARGET_BER,
        'protection': 'LSB<-MSB; mismatch->11', 'matrices': 224,
        'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    result_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f'START MMLU {trial}/5 seed={seed}', flush=True)
    lines = []
    with log_path.open('w', buffering=1, encoding='utf-8') as log:
        proc = subprocess.Popen(
            [str(PYTHON), '-u', str(worker_path), str(seed)],
            cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in proc.stdout:
            print(line, end='', flush=True)
            log.write(line)
            lines.append(line.rstrip())
        rc = proc.wait()
    bits = next((x.split('|', 1)[1] for x in lines if x.startswith('ONEZERO_BITS|')), None)
    result_line = next((x.split('|', 1)[1] for x in lines if x.startswith('RESULT_JSON|')), None)
    encode_line = next((x for x in lines if x.startswith('SRLR_ENCODE|')), None)
    correct_line = next((x for x in lines if x.startswith('SRLR_CORRECT|')), None)
    if bits:
        changed, total = map(int, bits.split('/'))
        record.update(changed_bits=changed, total_bits=total, actual_ber=changed/total)
    if result_line:
        record.update(json.loads(result_line))
    record.update(encode_line=encode_line, correct_line=correct_line, exit_code=rc)
    record['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record['status'] = 'completed' if rc == 0 and result_line and bits else 'failed'
    result_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print('ROUND_STATUS', json.dumps(record), flush=True)
    if record['status'] != 'completed':
        raise RuntimeError(f'failed seed={seed}; inspect {log_path}')
print('ALL_5_MMLU_COMPLETED', flush=True)
