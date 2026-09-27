# HBM Supplementary Experiments

Supplementary code and selected benchmark results for Qwen3-8B and Llama-3-8B-Instruct experiments on memory fault protection.

## Contents

- **`Qwen/`** — five experiment groups: `fp16-clean`, `int8-clean`, `int8-ber003`, `specc`, and `srlr`.
- **`Llama/`** — experiment notebooks, scripts, notes, and result summaries. Experiment details are in `Llama/docs/EXPERIMENTS.md`.

## Qwen INT8 configuration

The INT8 experiments use symmetric W8A16 RTN quantization (group size 128) with BF16 computation. Fault injection targets eligible weight bits in 252 non-`lm_head` linear layers; embeddings, `lm_head`, scales, and biases are excluded. The injected fault model is a one-to-zero bit flip at BER 0.003.

The `specc` group adapts SPECC (Sparrow ECC) to INT8 weights: high-nibble values are Hamming(7,4)-encoded, while eligible raw low-nibble bits are subject to fault injection. Encoded high-nibble bits are excluded from injection; therefore, these results do not evaluate recovery from faults injected into those code bits.

## Selected benchmark accuracy

| Model | Experiment | MathQA | MMLU | HumanEval |
|---|---|---:|---:|---:|
| Qwen3-8B | FP16 clean | 84.90% | 72.30% | 82.32% |
| Qwen3-8B | INT8 clean | 84.90% | 72.20% | 83.54% |
| Qwen3-8B | INT8 BER=0.003 | 24.41% | 43.99% | 37.81% |
| Qwen3-8B | SPECC INT8 | 84.39% | 72.36% | 82.87% |
| Qwen3-8B | SRLR INT8 | 80.40% | 71.13% | 78.17% |
| Llama-3-8B-Instruct | FP16 clean | 48.80% | 48.80% | 60.98% |
| Llama-3-8B-Instruct | INT8 clean | 48.40% | 65.60% | 65.60% |
| Llama-3-8B-Instruct | INT8 BER=0.003 | 5.80% | 26.46% | 1.34% |
| Llama-3-8B-Instruct | SPECC INT8 | 46.49% | 64.82% | 56.83% |
| Llama-3-8B-Instruct | SRLR INT8 | 36.04% | 59.34% | 43.66% |
