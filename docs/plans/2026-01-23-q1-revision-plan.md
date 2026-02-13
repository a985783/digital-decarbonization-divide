# Q1 Journal Revision Implementation Plan
> **Status**: Drafted
> **Target**: Q1 Journal (Major Revision)
> **Priorities**: Identification (IV/Placebo) > Mechanism (Renewables) > External Validity

## Overview
This plan addresses the reviewer's critical feedback regarding identification strategy, mechanism analysis, and external validity. The goal is to upgrade the current "working paper" quality to "Q1 publication" quality by adding rigorous robustness checks and deeper heterogeneity analysis.

## Phase 4: Identification Strategy (High Priority)
**Goal**: Address endogeneity concerns of the DCI variable.

### Task 4.1: Placebo Tests (Randomization Inference)
**Rationale**: Prove that the results are not driven by spurious trends or model overfitting.
**Files**:
- Create: `scripts/phase4_placebo.py`
- Output: `results/placebo_distribution.png`

**Steps**:
1.  **Random Treatment Permutation**: Shuffle `DCI` values across countries/years (breaking the correlation structure).
2.  **Re-run Causal Forest**: Train the model on the shuffled data (computationally expensive, maybe use 500 trees instead of 2000 for speed, or 100 iterations).
3.  **Distribution Plot**: Plot the distribution of Average Treatment Effects (ATE) from placebos vs. the True ATE.
    - *Success Criteria*: True ATE should fall outside the 95% range of the placebo distribution.
4.  **Space Placebo**: Randomize `DCI` across countries within the same year (preserving time trends but breaking cross-sectional logic).

### Task 4.2: IV Strategy (Double Machine Learning with IV)
**Rationale**: Address omitted variable bias.
**Files**:
- Create: `scripts/phase4_iv_analysis.py`
- Output: `results/iv_comparison_table.csv`

**Steps**:
1.  **Instrument Construction**:
    - Use **Lagged DCI (t-1, t-2)** as the instrument. (Standard GMM logic: past shocks predict current capacity but not current error term).
    - *Alternative*: If `wdi_expanded_raw.csv` contains "Terrain Ruggedness" or similar geographical fixed vars, use them. (Unlikely in standard WDI, so stick to Lags).
2.  **Model Implementation**:
    - Use `econml.dml.CausalForestDMLIV` or `LinearDML` with `model_z` (instrument model).
    - `Y = CO2`, `T = DCI`, `Z = DCI_lag1`, `X = Controls`.
3.  **Comparison**:
    - Compare `ATE_IV` vs `ATE_OLS` (or standard Causal Forest).
    - If `ATE_IV` is consistent (even if less precise), the main result holds.

## Phase 5: Mechanism Analysis (High Priority)
**Goal**: Explain the "counter-intuitive" finding that high renewable energy countries show weaker DCI effects.

### Task 5.1: The "Renewable Paradox" Deep Dive
**Files**:
- Create: `scripts/phase5_mechanism.py`
- Output: `results/figures/mechanism_renewable_curve.png`

**Steps**:
1.  **Non-linear Analysis**:
    - Load `causal_forest_cate.csv`.
    - Plot `CATE` (y-axis) vs `Renewable_energy_consumption_pct` (x-axis).
    - Fit a polynomial (quadratic) or GAM (Generalized Additive Model) line.
2.  **Hypothesis Testing**:
    - *Hypothesis 1 (Diminishing Returns)*: In clean grids, digital efficiency saves less carbon because the energy saved was already clean.
    - *Hypothesis 2 (Grid Stability)*: High renewable penetration + High digital load = Need for fossil peaker plants? (Harder to test without hourly data, but can discuss).
3.  **Interaction Check**:
    - Run `CATE ~ Renewable + Renewable^2 + GDP + Corruption`.
    - Check if the "weakening" effect is statistically significant controlling for GDP.

## Phase 6: External Validity (Medium Priority)
**Goal**: Defend the sample selection (40 countries).

### Task 6.1: Representativeness Analysis
**Files**:
- Create: `scripts/phase6_external_validity.py`
- Output: `results/figures/sample_representativeness.png`

**Steps**:
1.  **Global Comparison**:
    - Load `wdi_expanded_raw.csv`.
    - Calculate global averages for `GDP`, `CO2`, `DCI_components` (excluding the 40 sample countries).
    - Create a "Radar Plot" or "Bar Chart" comparing [Sample Mean] vs [Global Mean].
2.  **Coverage Statistic**:
    - Calculate % of Global GDP and % of Global CO2 covered by the 40 countries. (Likely >80%, which justifies the "Major Economies" focus).

## Phase 7: Writing & Visualization
**Goal**: Update the manuscript.

### Task 7.1: Update Figures
- Generate new plots from Phase 4, 5, 6.
- Update `results/figures/`.

### Task 7.2: Update LaTeX (Manual or separate task)
- Insert "Identification Strategy" section.
- Insert "Robustness Checks" section.
- Update "Discussion" with the Renewable Energy mechanism explanation.

## Execution Order
1. **Task 4.1 (Placebo)** - computationally intensive but straightforward.
2. **Task 5.1 (Mechanism)** - analytical, high value for "story".
3. **Task 4.2 (IV)** - technically risky (if IV is weak), do this after Placebo confirms signal.
4. **Task 6.1 (External Validity)** - low risk, easy win.

---
**Prepared by**: Sisyphus Agent
**Date**: 2026-01-23
