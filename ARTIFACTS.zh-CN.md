# 复现实验产物映射

语言 / Language: [中文](ARTIFACTS.zh-CN.md) | [English](ARTIFACTS.md)

## 核心数据集

- `data/clean_data_v4_imputed.csv`：多数阶段脚本使用的基线面板数据。
- `data/clean_data_v5_enhanced.csv`：增强特征版本，供仪表盘和比较模块使用。

## 主要定量输出

- `results/causal_forest_cate.csv`：`scripts/phase2_causal_forest.py` 生成的 CATE 结果。
- `results/iv_analysis_results.csv`：`scripts/phase4_iv_analysis.py` 的 IV 估计结果。
- `results/placebo_results.csv`：`scripts/phase4_placebo.py` 的 placebo 检验结果。
- `results/mechanism_analysis_results.csv`：`scripts/phase5_mechanism.py` 的机制分析输出。
- `results/external_validity_results.csv`：`scripts/phase6_external_validity.py` 的外部效度检验。
- `results/dynamic_effects.csv`：`scripts/phase7_dynamic_effects.py` 的动态效应结果。
- `results/sensitivity_analysis.csv`：`scripts/oster_sensitivity.py` 的 Oster 稳健性结果。
- `results/dragonnet_comparison.csv`：`scripts/dragonnet_comparison.py` 的模型比较结果。

## 图形与论文

- `results/figures/`：论文与报告使用的图形输出目录。
- `paper.pdf`、`paper_cn.pdf`：由 `compile_paper.sh` 编译得到的论文文件。

## 校验报告

- `results/academic_consistency_guard_report.md`：由 `scripts/academic_consistency_guard.py` 生成。
- `scripts/preflight_release_check.py` 输出的预发布检查状态。
