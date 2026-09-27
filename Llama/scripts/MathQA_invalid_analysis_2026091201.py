import os, sysconfig
from pathlib import Path
_nvroot = Path(sysconfig.get_paths()['purelib']) / 'nvidia'
os.environ['CPATH'] = os.pathsep.join((str(p) for p in _nvroot.glob('*/include')))
os.environ['LIBRARY_PATH'] = os.pathsep.join((str(p) for p in _nvroot.glob('*/lib')))
os.environ['MAX_JOBS'] = '2'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6;8.9'
GPU = 0
import os
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import gc, re, ast, textwrap, warnings
from pathlib import Path
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers.utils import logging as hf_logging
from optimum.quanto import quantize, freeze, qint8, quantization_map
from optimum.quanto.nn import QModuleMixin
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
warnings.filterwarnings('ignore', category=UserWarning)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
ROOT = Path('/tf/wjy')
MODEL_PATH = ROOT / 'models/Meta-Llama-3-8B-Instruct'
assert torch.cuda.is_available(), 'CUDA is required'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
EOS_IDS = list({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')})

def release_model():
    global model
    if 'model' in globals():
        del model
    gc.collect()
    torch.cuda.empty_cache()

def load_quantized_model(bits):
    global model
    assert bits == 8
    release_model()
    set_seed(42)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True, torch_dtype=torch.float16, device_map={'': GPU}, attn_implementation='sdpa').eval()
    expected = [name for name, module in model.named_modules() if name.startswith('model.layers.') and isinstance(module, torch.nn.Linear)]
    assert len(expected) == model.config.num_hidden_layers * 7
    qtype = qint8
    quantize(model, weights=qtype, activations=None, include=expected)
    freeze(model)
    config = quantization_map(model)
    assert set(config) == set(expected)
    assert all((c == {'weights': f'qint{bits}', 'activations': 'none'} for c in config.values()))
    modules = [model.get_submodule(name) for name in expected]
    assert all((m.frozen and m.weight.dtype == torch.float16 for m in modules))
    assert model.model.embed_tokens.weight.dtype == model.lm_head.weight.dtype == torch.float16
    seen = set()

    def check_input(module, args):
        assert args[0].dtype == torch.float16, 'Activation is not FP16'
        seen.add(id(module))
    hooks = [m.register_forward_pre_hook(check_input) for m in modules]
    try:
        sample = tokenizer('Hello', return_tensors='pt').to(model.device)
        with torch.inference_mode():
            logits = model(**sample, use_cache=False).logits
        assert len(seen) == len(modules) and torch.isfinite(logits).all()
    finally:
        for hook in hooks:
            hook.remove()
    gc.collect()
    torch.cuda.empty_cache()

@torch.inference_mode()
def generate_batch(messages, max_new_tokens):
    prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, add_special_tokens=False)
    assert inputs.input_ids.shape[1] + max_new_tokens <= model.config.max_position_embeddings
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    output = model.generate(**inputs, do_sample=False, temperature=None, top_p=None, max_new_tokens=max_new_tokens, eos_token_id=EOS_IDS, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.batch_decode(output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)

def run_evaluation():
    test = load_from_disk(ROOT / 'mathqa_fixed_1000')
    demos = load_from_disk(ROOT / 'mathqa_fixed_5shot')
    assert len(test) == 1000 and len(demos) == 5
    assert not {x['Problem'].strip() for x in test} & {x['Problem'].strip() for x in demos}

    def question(row):
        return f"Question: {row['Problem'].strip()}\nOptions: {row['options'].strip()}"

    def messages(row):
        msg = [{'role': 'system', 'content': 'Solve the multiple-choice math problem. Explain your reasoning concisely, then end with exactly: Final answer: <letter> where <letter> is a, b, c, d, or e.'}]
        for d in demos:
            msg += [{'role': 'user', 'content': question(d)}, {'role': 'assistant', 'content': d['Rationale'].strip() + '\nFinal answer: ' + d['correct'].strip().lower()}]
        return msg + [{'role': 'user', 'content': question(row)}]

    def parse_answer(text):
        matches = re.findall('(?im)^\\s*(?:\\*\\*)?final answer\\s*:\\s*(?:\\*\\*)?\\s*[\\(\\[]?([a-e])\\b', text)
        if matches:
            return matches[-1].lower()
        matches = re.findall('(?i)\\b(?:the\\s+)?(?:correct\\s+)?answer\\s*(?:is|:)\\s*(?:\\*\\*)?[\\(\\[]?([a-e])\\b', text)
        if matches:
            return matches[-1].lower()
        return text.strip().lower() if re.fullmatch('[a-e]', text.strip().lower()) else None
    correct = 0
    invalid = 0
    processed = 0

    pred_counts = {
        'a': 0,
        'b': 0,
        'c': 0,
        'd': 0,
        'e': 0
    }

    invalid_examples = []

    for start in range(0, len(test), 4):
        rows = [test[i] for i in range(start, min(start + 4, len(test)))]
        generated = generate_batch([messages(r) for r in rows], 1024)

        for offset, (text, row) in enumerate(zip(generated, rows)):
            pred = parse_answer(text)
            gold = row['correct'].strip().lower()

            if pred is None:
                invalid += 1

                if len(invalid_examples) < 20:
                    invalid_examples.append({
                        'index': start + offset,
                        'gold': gold,
                        'output': repr(text[:1500])
                    })
            else:
                pred_counts[pred] += 1

            if pred == gold:
                correct += 1

        processed += len(rows)

        print(
            'PROGRESS',
            processed,
            '/',
            len(test),
            'correct=',
            correct,
            'invalid=',
            invalid,
            flush=True
        )

    wrong_valid = processed - correct - invalid
    valid = processed - invalid

    print('\n' + '=' * 70)
    print('MATHQA OUTPUT ANALYSIS')
    print('=' * 70)

    print('Total:', processed)
    print('Correct:', correct)
    print('Wrong valid:', wrong_valid)
    print('Invalid:', invalid)

    print(f'Accuracy: {correct / processed * 100:.2f}%')
    print(f'Valid rate: {valid / processed * 100:.2f}%')
    print(f'Invalid rate: {invalid / processed * 100:.2f}%')

    print('\nPrediction distribution:')

    for letter in ['a', 'b', 'c', 'd', 'e']:
        count = pred_counts[letter]
        print(
            f'{letter.upper()}: {count} '
            f'({count / processed * 100:.2f}%)'
        )

    print(
        f'Invalid: {invalid} '
        f'({invalid / processed * 100:.2f}%)'
    )

    print('\n' + '=' * 70)
    print('FIRST 20 INVALID OUTPUTS')
    print('=' * 70)

    if not invalid_examples:
        print('No invalid outputs.')
    else:
        for item in invalid_examples:
            print('\n' + '-' * 70)
            print('Index:', item['index'])
            print('Gold:', item['gold'])
            print('Raw output:')
            print(item['output'])

    return (correct, len(test))

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
    print('BITS BEFORE', total_bits, 'ONES', total_ones, 'K', k, flush=True)
    chosen = uniform_subset(total_ones, k, seed)
    print('SAMPLING DONE', flush=True)
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
            print('INJECT VERIFIED', verified, '/', k, name, flush=True)
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
    print('LOAD original FP16 Quanto INT8', flush=True)
    original_loader(bits)
    cfg = quantization_map(model)
    expected = {f'model.layers.{i}.{part}' for i in range(32) for part in ('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj')}
    assert set(cfg) == expected
    assert all(p.dtype == torch.float16 for p in model.parameters() if p.is_floating_point())
    print('INJECT START: 224 matrices, no lm_head or embeddings', flush=True)
    changed, total = inject_one_to_zero(seed)
    print(f'ONEZERO_BITS|{changed}/{total}', flush=True)
    def finite_check(module, args, output):
        if not torch.isfinite(output.logits[:, -1, :]).all().item():
            raise RuntimeError('Nonfinite logits; score is invalid.')
    model.register_forward_hook(finite_check)


import json,datetime,time
print('WORKER PID',os.getpid(),'GPU',os.environ['CUDA_VISIBLE_DEVICES'],'SEED',seed,flush=True)
try:
    load_quantized_model(8)
    print('EVAL START',flush=True)
    correct,total=run_evaluation()
    print('RESULT_JSON|'+json.dumps({'correct':correct,'total_questions':total,'accuracy':correct/total}),flush=True)
finally:
    release_model()
