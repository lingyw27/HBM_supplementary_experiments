# Llama-3-8B-Instruct INT8 Fault-Injection Experiments

本目录只整理截图对应的 FP16、Quanto INT8、SpECC INT8 和 SRLR INT8 实验，不包含 AWQ INT4 内容。

## 实验对象

- 模型：Llama-3-8B-Instruct
- 评测集：MathQA、MMLU、HumanEval
- 故障模型：HBM 持久性 1→0 位翻转
- 目标 BER：0.003
- clean：未注错模型
- protected：SpECC 或 SRLR 保护后的 INT8 模型

故障注入从 clean INT8 模型重新加载，在指定权重范围内选择初始值为 1 的 bit，并将选中的 bit 翻转为 0。每轮记录随机种子、注错数量、实际 BER、GPU 和准确率。

## 汇总结果

| 模型/模式 | MathQA | MMLU | HumanEval |
|---|---:|---:|---:|
| FP16 clean | 48.80% | 48.80% | 60.98% |
| INT8 clean | 48.40% | 65.60% | 65.60% |
| INT8，BER=0.003 | 5.80% | 26.46% | 1.34% |
| INT8 + SpECC | 46.49% | 64.82% | 56.83% |
| INT8 + SRLR | 36.04% | 59.34% | 43.66% |

## 目录说明

- scripts/：INT8 故障注入、SpECC、SRLR 和评测控制程序。
- notebooks/：MMLU、MathQA、HumanEval 和 INT8 实验 notebook。
- results_summary/：小体积汇总结果。

模型权重、缓存、完整日志、原始 mask 和逐样本结果不纳入仓库。
