# 可复现指南

语言 / Language: [中文](REPRODUCIBILITY.zh-CN.md) | [English](REPRODUCIBILITY.md)

本仓库已按论文复现包方式组织，可在新机器上按固定流程复现实证结果与论文产物。

## 1）系统要求

- Python 3.8+
- `pip`
- LaTeX 工具链（`xelatex`、`pdflatex`、`bibtex`）用于论文编译

## 2）快速开始（一条命令）

```bash
bash reproduce.sh
```

该命令会自动安装依赖、运行测试、执行发布前校验、运行核心分析管线并编译论文。

## 3）分阶段复现

```bash
make setup
make test
make verify
make analysis
make paper
```

## 4）预期核心输出

- `results/causal_forest_cate.csv`
- `results/iv_analysis_results.csv`
- `results/placebo_results.csv`
- `results/mechanism_analysis_results.csv`
- `results/external_validity_results.csv`
- `results/dynamic_effects.csv`
- `results/sensitivity_analysis.csv`
- `results/dragonnet_comparison.csv`
- `paper.pdf`
- `paper_cn.pdf`

## 5）发布前校验门禁

- `python3 scripts/preflight_release_check.py`
- `python3 scripts/academic_consistency_guard.py`

发布前这两项都应通过。

## 6）仪表盘运行

```bash
streamlit run app.py
```

目前数据加载已改为仓库相对路径，克隆后可直接运行。

## 7）常见问题

- 若论文编译失败，先安装缺失 TeX 包，再执行 `make paper`。
- 若依赖安装冲突，请使用全新虚拟环境并重新执行 `make setup`。
- 若校验失败，请按报错路径检查对应结果文件并重新生成。
