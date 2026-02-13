PYTHON ?= python3

.PHONY: setup test verify analysis paper all

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -q tests/test_academic_consistency_guard.py tests/test_phase4_iv_analysis.py tests/test_phase4_placebo.py tests/test_phase5_mechanism.py tests/test_phase6_external_validity.py

verify:
	$(PYTHON) scripts/preflight_release_check.py
	$(PYTHON) scripts/academic_consistency_guard.py

analysis:
	$(PYTHON) -m scripts.phase1_mvp_check
	$(PYTHON) -m scripts.phase2_causal_forest
	$(PYTHON) -m scripts.phase3_visualizations
	$(PYTHON) -m scripts.phase4_iv_analysis
	$(PYTHON) -m scripts.phase4_placebo
	$(PYTHON) -m scripts.phase5_mechanism
	$(PYTHON) -m scripts.phase6_external_validity
	$(PYTHON) -m scripts.phase7_dynamic_effects
	$(PYTHON) -m scripts.oster_sensitivity
	$(PYTHON) -m scripts.dragonnet_comparison

paper:
	bash compile_paper.sh

all: setup test verify analysis paper
