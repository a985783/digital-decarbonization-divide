# 数字脱碳鸿沟（增强研究复现包）

语言 / Language: [中文](README.zh-CN.md) | [English](README.md)

本仓库用于复现论文《The Digital Decarbonization Divide》的核心实证结果，覆盖 40 个经济体（2000-2023）并提供完整的可复现工作流。

## 一键复现

```bash
git clone https://github.com/a985783/digital-decarbonization-divide.git
cd digital-decarbonization-divide
bash reproduce.sh
```

## 分阶段复现

```bash
make setup
make test
make verify
make analysis
make paper
```

详细说明见 `REPRODUCIBILITY.zh-CN.md`（中文）或 `REPRODUCIBILITY.md`（英文）。

## 项目亮点

- 因果推断主方法：Causal Forest DML
- 稳健性与补充方法：IV、Placebo、Oster、DragonNet
- 输出完整：结果表、图表、论文 PDF（中英文）
- 可发布：含 CI 校验、CITATION、LICENSE、贡献指南

## 关键输出

- `results/causal_forest_cate.csv`
- `results/iv_analysis_results.csv`
- `results/placebo_results.csv`
- `results/sensitivity_analysis.csv`
- `results/dragonnet_comparison.csv`
- `paper.pdf`、`paper_cn.pdf`

## 引用

标准引用信息见 `CITATION.cff`。

## 作者与联系

- Qingsong Cui（独立研究者 / Independent Researcher）
- 邮箱 / Email: `qingsongcui9857@gmail.com`

## 许可证

本项目采用 MIT License，详见 `LICENSE`。
