# Reproducibility Guide

Language / 语言: [English](REPRODUCIBILITY.md) | [中文](REPRODUCIBILITY.zh-CN.md)

This repository is structured as a paper replication package.

## 1) System Requirements

- Python 3.8+
- `pip`
- LaTeX toolchain (`xelatex`, `pdflatex`, `bibtex`) for paper compilation

## 2) Quick Start (One Command)

```bash
bash reproduce.sh
```

This command installs dependencies, runs focused tests, executes release guards,
runs the core analysis pipeline, and compiles the papers.

## 3) Stage-by-Stage Reproduction

```bash
make setup
make test
make verify
make analysis
make paper
```

## 4) Expected Core Outputs

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

## 5) Verification Gates

- `python3 scripts/preflight_release_check.py`
- `python3 scripts/academic_consistency_guard.py`

Both checks must pass before packaging a release.

## 6) Dashboard Execution

```bash
streamlit run app.py
```

Data loading now uses repository-relative paths and works after clone.

## 7) Troubleshooting

- If paper compilation fails, install missing TeX packages and rerun `make paper`.
- If dependency resolution fails, use a clean virtual environment and rerun `make setup`.
- If verification fails, inspect the reported file path in guard output and regenerate required artifacts.
