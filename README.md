# HBM Supplementary Experiments

This repository collects supplementary experiment code and compact result summaries for Qwen3-8B and Llama-3-8B-Instruct.

## Repository structure

- `Qwen/`: five Qwen3-8B experiment groups: `fp16-clean`, `int8-clean`, `int8-ber003`, `specc`, and `srlr`. Each group contains its worker or evaluation code and configuration files.
- `Llama/`: Llama-3-8B-Instruct experiment notebooks, scripts, experiment notes, and compact result summaries.

## Configuration summary

The Qwen INT8 experiments use symmetric W8A16 RTN quantization (group size 128) with BF16 compute. The configured protection/injection scope is the 252 non-`lm_head` linear weight matrices; embeddings, `lm_head`, scales, and biases are excluded. Fault injection uses eligible 1→0 bit flips at BER 0.003.

The Qwen `specc` experiment is an INT8 adaptation of SPECC (Sparrow ECC): each high nibble is encoded with Hamming(7,4), and encoded high-nibble bits are excluded from fault injection. BER 0.003 one-to-zero faults are injected only into eligible raw low-nibble bits. As high-nibble codeword bits are not faulted, this experiment does not measure ECC recovery from high-nibble errors.

The Llama experiment scope and configuration are documented in `Llama/docs/EXPERIMENTS.md`; compact benchmark summaries are in `Llama/results_summary/`.

## Selected benchmark accuracy

### Qwen3-8B

| Experiment | MathQA | MMLU | HumanEval |
|---|---:|---:|---:|
| FP16 clean | 84.90% | 72.30% | 82.32% |
| INT8 clean | 84.90% | 72.20% | 83.54% |
| INT8 BER=0.003 | 24.41% | 43.99% | 37.81% |
| SPECC INT8 | 84.39% | 72.36% | 82.87% |
| SRLR INT8 | 80.40% | 71.13% | 78.17% |

### Llama-3-8B-Instruct

| Experiment | MathQA | MMLU | HumanEval |
|---|---:|---:|---:|
| FP16 clean | 48.80% | 48.80% | 60.98% |
| INT8 clean | 48.40% | 65.60% | 65.60% |
| INT8 BER=0.003 | 5.80% | 26.46% | 1.34% |
| SPECC INT8 | 46.49% | 64.82% | 56.83% |
| SRLR INT8 | 36.04% | 59.34% | 43.66% |

