#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[1/6] Python version"
"${PYTHON_BIN}" --version

echo "[2/6] Install dependencies"
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -r requirements.txt

echo "[3/6] Run focused tests"
"${PYTHON_BIN}" -m pytest -q tests/test_analysis_config.py tests/test_docs_consistency.py tests/test_phase4_iv_analysis.py tests/test_phase6_external_validity.py

echo "[4/6] Run release guards"
"${PYTHON_BIN}" scripts/preflight_release_check.py
"${PYTHON_BIN}" scripts/academic_consistency_guard.py

echo "[5/6] Run core analysis pipeline"
"${PYTHON_BIN}" -m scripts.phase1_mvp_check
"${PYTHON_BIN}" -m scripts.phase2_causal_forest
"${PYTHON_BIN}" -m scripts.phase3_visualizations
"${PYTHON_BIN}" -m scripts.phase4_iv_analysis
"${PYTHON_BIN}" -m scripts.phase4_placebo
"${PYTHON_BIN}" -m scripts.phase5_mechanism
"${PYTHON_BIN}" -m scripts.phase6_external_validity
"${PYTHON_BIN}" -m scripts.phase7_dynamic_effects
"${PYTHON_BIN}" -m scripts.oster_sensitivity
"${PYTHON_BIN}" -m scripts.dragonnet_comparison

echo "[6/6] Compile papers"
bash compile_paper.sh

echo "Done. Reproduction artifacts are under results/ and paper PDFs at repository root."
