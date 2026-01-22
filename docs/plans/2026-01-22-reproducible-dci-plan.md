# Reproducible DCI Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the paper fully reproducible by aligning data, code, results, and manuscript around a real WDI-constructed DCI treatment.

**Architecture:** A config-driven pipeline (`analysis_spec.yaml`) controls indicators, treatments, and inference. DCI is constructed from real WDI indicators via PCA, MICE imputation runs within training folds only, and analysis scripts consume the same config to produce auditable outputs.

**Tech Stack:** Python 3.9, pandas, numpy, miceforest, scikit-learn, econml, pytest

---

### Task 1: Add config + loader

**Files:**
- Create: `analysis_spec.yaml`
- Create: `scripts/analysis_config.py`
- Test: `tests/test_analysis_config.py`

**Step 1: Write the failing test**

```python
from scripts.analysis_config import load_config

def test_load_config_has_required_fields(tmp_path):
    cfg = load_config("analysis_spec.yaml")
    assert cfg["treatment_main"] == "DCI"
    assert "moderators_X" in cfg
    assert "controls_W" in cfg
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_analysis_config.py -v`  
Expected: FAIL with `ModuleNotFoundError` or missing keys

**Step 3: Write minimal implementation**

```python
# scripts/analysis_config.py
import yaml

REQUIRED_KEYS = ["treatment_main", "treatment_secondary", "outcome",
                 "moderators_X", "controls_W", "years", "groups",
                 "cv", "bootstrap", "imputation", "dci_components",
                 "wdi_indicators"]

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in REQUIRED_KEYS:
        if key not in cfg:
            raise ValueError(f"Missing config key: {key}")
    return cfg
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_analysis_config.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add analysis_spec.yaml scripts/analysis_config.py tests/test_analysis_config.py
git commit -m "feat: add analysis config loader"
```

---

### Task 2: DCI construction module

**Files:**
- Create: `scripts/dci.py`
- Test: `tests/test_dci.py`

**Step 1: Write the failing test**

```python
import pandas as pd
from scripts.dci import build_dci

def test_build_dci_requires_components():
    df = pd.DataFrame({"Internet_users": [10, 20]})
    try:
        build_dci(df, ["Internet_users", "Fixed_broadband_subscriptions", "Secure_servers"])
    except ValueError:
        assert True
    else:
        assert False
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_dci.py -v`  
Expected: FAIL (module missing)

**Step 3: Write minimal implementation**

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def build_dci(df, components):
    missing = [c for c in components if c not in df.columns]
    if missing:
        raise ValueError(f"Missing DCI components: {missing}")
    X = df[components].astype(float)
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=1, random_state=42)
    dci = pca.fit_transform(X_scaled).reshape(-1)
    # Standardize to mean 0, sd 1
    dci = (dci - dci.mean()) / dci.std(ddof=0)
    return dci, pca.explained_variance_ratio_[0]
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_dci.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/dci.py tests/test_dci.py
git commit -m "feat: add DCI construction module"
```

---

### Task 3: WDI indicator mapping from config

**Files:**
- Create: `scripts/wdi_indicators.py`
- Modify: `scripts/solve_wdi_v4_expanded_zip.py`
- Test: `tests/test_wdi_indicators.py`

**Step 1: Write the failing test**

```python
from scripts.analysis_config import load_config
from scripts.wdi_indicators import load_indicators

def test_required_wdi_codes_present():
    cfg = load_config("analysis_spec.yaml")
    indicators = load_indicators(cfg)
    for code in ["IT.NET.USER.ZS", "IT.NET.BBND.P2", "IT.NET.SECR.P6", "BX.GSR.CCIS.ZS"]:
        assert code in indicators
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_wdi_indicators.py -v`  
Expected: FAIL (module missing)

**Step 3: Write minimal implementation**

```python
# scripts/wdi_indicators.py
def load_indicators(cfg):
    return cfg["wdi_indicators"]
```

Update `scripts/solve_wdi_v4_expanded_zip.py` to import `load_config` + `load_indicators`.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_wdi_indicators.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/wdi_indicators.py scripts/solve_wdi_v4_expanded_zip.py tests/test_wdi_indicators.py
git commit -m "feat: drive WDI indicators from config"
```

---

### Task 4: Fold-safe MICE imputation

**Files:**
- Create: `scripts/imputation.py`
- Modify: `scripts/impute_mice.py`
- Test: `tests/test_imputation_folded.py`

**Step 1: Write the failing test**

```python
import pandas as pd
from scripts.imputation import impute_folded

def test_impute_does_not_touch_y_t():
    df = pd.DataFrame({
        "country": ["A","A","B","B"],
        "year": [2000,2001,2000,2001],
        "Y": [1.0, None, 2.0, None],
        "T": [0.1, None, 0.2, None],
        "W1": [1.0, None, 3.0, None],
    })
    out = impute_folded(df, y_col="Y", t_col="T", w_cols=["W1"], group_col="country")
    assert out["Y"].isna().sum() == 2
    assert out["T"].isna().sum() == 2
    assert out["W1"].isna().sum() == 0
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_imputation_folded.py -v`  
Expected: FAIL (module missing)

**Step 3: Write minimal implementation**

Implement `impute_folded` using GroupKFold + miceforest:
- Fit `ImputationKernel` on train subset (W/X only)
- Use `impute_new_data` on test subset
- Merge results in original order

Update `scripts/impute_mice.py` to call this function and enforce no Y/T imputation.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_imputation_folded.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/imputation.py scripts/impute_mice.py tests/test_imputation_folded.py
git commit -m "feat: fold-safe MICE without Y/T imputation"
```

---

### Task 5: Analysis scripts use DCI + GroupKFold

**Files:**
- Modify: `scripts/phase1_mvp_check.py`
- Modify: `scripts/phase2_causal_forest.py`
- Modify: `scripts/dml_causal_v2.py`
- Modify: `scripts/rebuttal_analysis.py`
- Test: `tests/test_prepare_analysis_data.py`

**Step 1: Write the failing test**

```python
import pandas as pd
from scripts.analysis_config import load_config
from scripts.analysis_data import prepare_analysis_data

def test_treatment_is_dci():
    cfg = load_config("analysis_spec.yaml")
    df = pd.DataFrame({
        "country":["A"], "year":[2000],
        "DCI":[0.0], "ICT_exports":[5.0], "CO2_per_capita":[1.0]
    })
    Y, T, X, W = prepare_analysis_data(df, cfg)
    assert T[0] == 0.0
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_prepare_analysis_data.py -v`  
Expected: FAIL (module missing)

**Step 3: Write minimal implementation**

Create `scripts/analysis_data.py` to:
- Read `treatment_main` from config (DCI)
- Exclude DCI components from W
- Build X from `moderators_X`
Update analysis scripts to use `prepare_analysis_data` and GroupKFold by country.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_prepare_analysis_data.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/analysis_data.py scripts/phase1_mvp_check.py scripts/phase2_causal_forest.py scripts/dml_causal_v2.py scripts/rebuttal_analysis.py tests/test_prepare_analysis_data.py
git commit -m "feat: run analyses with DCI treatment and group CV"
```

---

### Task 6: Doc consistency + manifest updates

**Files:**
- Modify: `DATA_MANIFEST.md`
- Modify: `README.md`
- Modify: `paper.tex`
- Test: `tests/test_docs_consistency.py`

**Step 1: Write the failing test**

```python
from pathlib import Path

def test_no_simulated_dci_in_paper():
    text = Path("paper.tex").read_text(encoding="utf-8")
    assert "Simulated" not in text
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_docs_consistency.py -v`  
Expected: FAIL (Simulated still present)

**Step 3: Write minimal implementation**

Update documentation:
- Replace any "simulated" wording for DCI
- Add WDI codes, download date placeholder, license notes
- Ensure methods match fold-safe MICE + GroupKFold

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_docs_consistency.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add DATA_MANIFEST.md README.md paper.tex tests/test_docs_consistency.py
git commit -m "docs: align manuscript with real DCI pipeline"
```

---

### Task 7: Regenerate results + verify

**Files:**
- Modify: `results/*.csv`
- Modify: `results/figures/*`

**Step 1: Run pipeline**

Run: `python3 scripts/solve_wdi_v4_expanded_zip.py`  
Run: `python3 scripts/impute_mice.py`  
Run: `python3 scripts/phase1_mvp_check.py`  
Run: `python3 scripts/phase2_causal_forest.py`  
Run: `python3 scripts/phase3_visualizations.py`

**Step 2: Verify**

Run: `python3 -m pytest -v`  
Expected: PASS  
Check `results/` for updated timestamps and schema.

**Step 3: Commit**

```bash
git add results
git commit -m "data: regenerate results with real DCI treatment"
```

---

Plan complete. Ready for implementation.
