# HBM 补充实验

本仓库收录 Qwen3-8B 与 Llama-3-8B-Instruct 在存储器故障保护方面的补充实验代码及部分基准结果。

## 目录结构

- **`Qwen/`**：五组实验，分别为 `fp16-clean`、`int8-clean`、`int8-ber003`、`specc` 和 `srlr`。
- **`Llama/`**：实验笔记本、脚本、说明文档和结果摘要。实验说明见 `Llama/docs/EXPERIMENTS.md`。

## Qwen INT8 配置

INT8 实验采用对称 W8A16 RTN 量化，group size 为 128，计算精度为 BF16。故障注入范围为 252 个非 `lm_head` 线性层中符合条件的权重位；不包含 embedding、`lm_head`、scale 和 bias。注错模型为 BER=0.003 的 1→0 位翻转。

`specc` 实验将 SPECC（Sparrow ECC）适配到 INT8 权重：高半字节采用 Hamming(7,4) 编码，符合条件的低半字节原始位参与注错；编码后的高半字节位不注错。因此，该实验不评估编码位发生注错时的纠错恢复能力。

## 部分基准准确率

| 模型 | 实验配置 | MathQA | MMLU | HumanEval |
|---|---|---:|---:|---:|
| Qwen3-8B | FP16 无注错 | 84.90% | 72.30% | 82.32% |
| Qwen3-8B | INT8 无注错 | 84.90% | 72.20% | 83.54% |
| Qwen3-8B | INT8，BER=0.003 | 24.41% | 43.99% | 37.81% |
| Qwen3-8B | SPECC INT8 | 84.39% | 72.36% | 82.87% |
| Qwen3-8B | SRLR INT8 | 80.40% | 71.13% | 78.17% |
| Llama-3-8B-Instruct | FP16 无注错 | 48.80% | 48.80% | 60.98% |
| Llama-3-8B-Instruct | INT8 无注错 | 48.40% | 65.60% | 65.60% |
| Llama-3-8B-Instruct | INT8，BER=0.003 | 5.80% | 26.46% | 1.34% |
| Llama-3-8B-Instruct | SPECC INT8 | 46.49% | 64.82% | 56.83% |
| Llama-3-8B-Instruct | SRLR INT8 | 36.04% | 59.34% | 43.66% |
