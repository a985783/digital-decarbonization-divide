# Academic Integrity Report: Paper-Results Consistency Audit

> ⚠️ Historical audit snapshot. This report captured an earlier inconsistency window and is retained for traceability.
> Current status must be evaluated using `results/academic_consistency_guard_report.md` and the latest regenerated `results/*.csv`.

**Date:** 2026-02-13
**Auditor:** consistency-auditor
**Scope:** Verify consistency between paper.tex claims and actual result files

---

## Executive Summary

This audit compares numerical claims in `paper.tex` against actual values in result files within `/Users/cuiqingsong/Documents/数字脱碳鸿沟-完结_存档/results/`.

**Overall Assessment:** **CRITICAL ISSUES IDENTIFIED**

- **1 CRITICAL inconsistency** (IV analysis results completely corrupted)
- **2 MAJOR inconsistencies** (placebo test incomplete, CATE range mismatch)
- **3 MINOR inconsistencies** (numerical rounding, labeling differences)
- **All figures referenced exist** in results/figures/

---

## 1. Detailed Claim Verification

### 1.1 IV Estimate: -1.91 (95% CI: [-2.37, -1.46])

| Aspect | Paper Claim | Actual Result File | Status |
|--------|-------------|-------------------|--------|
| IV Estimate | -1.91 | **0.5** (iv_analysis_results.csv) | **CRITICAL MISMATCH** |
| 95% CI Lower | -2.37 | **0.1** | **CRITICAL MISMATCH** |
| 95% CI Upper | -1.46 | **0.9** | **CRITICAL MISMATCH** |
| First-stage F | 247.63 | **-2.89** (nonsensical negative) | **CRITICAL MISMATCH** |
| First-stage R² | 0.946 | **-0.626** (impossible negative) | **CRITICAL MISMATCH** |

**Analysis:** The `iv_analysis_results.csv` file contains **corrupted/incorrect values**:
- ATE of 0.5 vs paper's -1.91 (wrong sign and magnitude)
- F-statistic is negative (-2.89), which is statistically impossible
- R² is negative (-0.626), which is mathematically impossible

**Severity:** **CRITICAL** - The IV analysis results file appears to be corrupted or contains placeholder values. The paper's reported values (-1.91, F=247.63) are NOT found in the results file.

---

### 1.2 First-Stage F-Statistic: 247.63

| Aspect | Paper Claim | Actual Result File | Status |
|--------|-------------|-------------------|--------|
| F-Statistic | 247.63 | **-2.89** (iv_analysis_results.csv) | **CRITICAL MISMATCH** |
| Alternative source | - | 98.21 (placebo_iv_results.csv, lag 2) | Partial match |

**Note:** The `placebo_iv_results.csv` shows F-statistics of 98.21 (lag 2) and 37.22 (lag 3), which are reasonable but don't match the paper's 247.63.

**Severity:** **CRITICAL** - The claimed F-statistic of 247.63 has no supporting evidence in result files.

---

### 1.3 Placebo Test p-value: < 0.001

| Aspect | Paper Claim | Actual Result File | Status |
|--------|-------------|-------------------|--------|
| Pseudo p-value | < 0.001 | **N/A** (insufficient data) | **MAJOR ISSUE** |
| Signal-to-Noise | ~23x | Cannot verify | Cannot verify |
| Placebo runs | N=100 | **Only 2 iterations** in file | **MAJOR ISSUE** |

**Analysis:** The `phase4_placebo_results.csv` contains only **2 placebo iterations** (both showing 0.4185), not the 100 claimed in the paper. This is insufficient to calculate a reliable p-value.

**Severity:** **MAJOR** - The placebo test file is incomplete. Cannot verify the p < 0.001 claim.

---

### 1.4 CATE Range: -4.35 to +0.33

| Aspect | Paper Claim | Actual Result File | Status |
|--------|-------------|-------------------|--------|
| Minimum CATE | -4.35 | **-4.30** (RUS in rebuttal_country_cates.csv) | MINOR mismatch |
| Maximum CATE | +0.33 | **-0.07** (FIN, all negative) | **MAJOR ISSUE** |

**Analysis:** From `rebuttal_country_cates.csv`:
- **Actual minimum:** -4.30 (RUS - Russia)
- **Actual maximum:** -0.07 (FIN - Finland)
- **All CATEs are negative** - there is no positive CATE

The paper claims a range extending to +0.33 (positive), but the actual data shows all effects are negative (ranging from -4.30 to -0.07).

**Severity:** **MAJOR** - The upper bound of +0.33 is not supported by data. All CATEs are negative.

---

### 1.5 Mediation Effect: 11.7%

| Aspect | Paper Claim | Actual Result File | Status |
|--------|-------------|-------------------|--------|
| Mediation proportion | 11.7% | **11.67%** (mediation_summary.csv) | MATCH |
| Sobel test p-value | < 0.001 | **7.6e-06** | MATCH |
| Mediator | Energy Efficiency | Energy_Efficiency | MATCH |

**Analysis:** The `mediation_summary.csv` shows:
- `proportion_mediated` = 0.11667 (11.67%)
- Rounded to 11.7% in paper - **CORRECT**

**Severity:** **CONSISTENT**

---

### 1.6 Sample Size: 840 observations, 40 countries

| Aspect | Paper Claim | Actual Result File | Status |
|--------|-------------|-------------------|--------|
| Observations | 840 | **840** (multiple files) | MATCH |
| Countries | 40 | **40** (rebuttal_country_cates.csv) | MATCH |

**Verification:**
- `mechanism_enhanced_results.csv`: sample_size = 840
- `rebuttal_country_cates.csv`: 40 countries listed
- `causal_forest_cate.csv`: 840 rows (after inspection)

**Severity:** **CONSISTENT**

---

### 1.7 Pointwise Significance: 79.2%

| Aspect | Paper Claim | Actual Result File | Status |
|--------|-------------|-------------------|--------|
| Pointwise significance | 79.2% | **Not directly found** | CANNOT VERIFY |

**Analysis:** No result file explicitly contains the "79.2%" figure for pointwise significance. This may be calculated from `causal_forest_cate.csv` (counting Significant=True / total), but the exact percentage is not pre-computed in any file.

**Severity:** **MINOR** - Cannot verify without recalculating from raw CATE data.

---

### 1.8 Triple Interaction p-value: < 0.001

| Aspect | Paper Claim | Actual Result File | Status |
|--------|-------------|-------------------|--------|
| Triple interaction p | < 0.001 | **0.000216** (mechanism_enhanced_results.csv) | MATCH |

**Analysis:** The `mechanism_enhanced_results.csv` shows:
- `triple_interaction_p` = 0.000216
- This is < 0.001, consistent with paper claim

**Severity:** **CONSISTENT**

---

## 2. Model Ladder Verification

| Model | Paper ATE | Paper SE | Paper CI | File ATE | File SE | File CI | Status |
|-------|-----------|----------|----------|----------|---------|---------|--------|
| L0 (TWFE) | -2.81 | 0.463 | [-3.73, -1.84] | -2.81 | 0.463 | [-3.73, -1.84] | MATCH |
| L1 (Linear DML) | -0.99 | 0.436 | [-2.07, -0.37] | -0.99 | 0.436 | [-2.07, -0.37] | MATCH |
| L2 (Interactive) | -1.22 | 0.393 | [-2.03, -0.46] | -1.22 | 0.393 | [-2.03, -0.46] | MATCH |
| L3 (Causal Forest) | -1.73 | 0.588 | [-2.88, -0.58] | -1.73 | 0.588 | [-2.88, -0.58] | MATCH |

**Source:** `model_ladder.csv`

**Severity:** **ALL CONSISTENT**

---

## 3. GATE Results Verification

| GDP Group | Paper Estimate | Paper CI Lower | Paper CI Upper | File Estimate | File CI Lower | File CI Upper | Status |
|-----------|----------------|----------------|----------------|---------------|---------------|---------------|--------|
| Low Income | -1.19 | -1.47 | -0.99 | -1.19 | -1.47 | -0.99 | MATCH |
| Lower-Mid | -2.17 | -2.66 | -1.76 | -2.17 | -2.66 | -1.76 | MATCH |
| Upper-Mid | -2.29 | -2.65 | -1.85 | -2.29 | -2.65 | -1.85 | MATCH |
| High Income | -1.26 | -1.67 | -0.81 | -1.26 | -1.67 | -0.81 | MATCH |

**Source:** `rebuttal_gate.csv`

**Severity:** **ALL CONSISTENT**

---

## 4. Interaction Term Results Verification

### GDP Interaction (from phase1b_gdp_results.csv)

| Coefficient | Paper Value | File Value | Status |
|-------------|-------------|------------|--------|
| Main Effect (DCI) | -3.365 | -3.365 | MATCH |
| Interaction (DCI × log(GDP)) | -0.126 | -0.126 | MATCH |
| p-value (interaction) | 0.326 | 0.326 | MATCH |

### Institution Interaction (from phase1_mvp_results.csv)

| Coefficient | Paper Value | File Value | Status |
|-------------|-------------|------------|--------|
| Main Effect (DCI) | -1.376 | -1.376 | MATCH |
| Interaction (DCI × Institution) | 0.765 | 0.765 | MATCH |
| p-value (interaction) | <0.001 | 6.23e-05 | MATCH |

**Severity:** **ALL CONSISTENT**

---

## 5. Policy Exceptions (Weakest Reductions) Verification

| Country | Paper CATE | Paper CI | File CATE | File CI | Status |
|---------|------------|----------|-----------|---------|--------|
| FIN | -0.19 | [-0.39, -0.07] | **-0.19** | [-0.39, -0.07] | MATCH |
| SWE | -0.46 | [-0.60, -0.30] | **-0.46** | [-0.60, -0.30] | MATCH |
| CHE | -0.50 | [-0.57, -0.44] | **-0.50** | [-0.57, -0.44] | MATCH |
| CAN | -0.52 | [-0.62, -0.44] | **-0.52** | [-0.62, -0.44] | MATCH |
| VNM | -0.90 | [-1.01, -0.82] | **-0.90** | [-1.01, -0.82] | MATCH |

**Source:** `rebuttal_country_cates.csv`

**Severity:** **ALL CONSISTENT**

---

## 6. Figure Existence Verification

| Figure Reference | File Path | Status |
|-----------------|-----------|--------|
| Figure 1: linear_vs_forest | results/figures/linear_vs_forest.png | EXISTS |
| Figure 2: off_diagonal_cis | results/figures/off_diagonal_cis.png | EXISTS |
| Figure 3: gate_plot | results/figures/gate_plot.png | EXISTS |
| Figure 4: mechanism_renewable_curve | results/figures/mechanism_renewable_curve.png | EXISTS |
| Scree plot | results/figures/pca_scree_plot.png | EXISTS |
| Power simulation | results/figures/power_simulation_distribution.png | EXISTS |

**Severity:** **ALL FIGURES EXIST**

---

## 7. Additional Findings

### 7.1 LOCO Stability Results
The paper mentions LOCO (Leave-One-Country-Out) analysis showing Global ATE range of -2.33 to -0.67. The `loco_stability.csv` file confirms this range:
- Minimum Global_ATE: -2.33 (dropping ARG)
- Maximum Global_ATE: -0.67 (dropping DEU)

**Status:** CONSISTENT

### 7.2 PCA Diagnostics
The paper mentions PCA first component explains ~70% of variance. The `pca_diagnostics.csv` shows:
- Explained_Variance_PC1: 0.701 (70.1%)

**Status:** CONSISTENT

### 7.3 Small Sample Robustness
The `small_sample_robustness.csv` shows concerning results:
- `bootstrap_converged`: False
- `sample_size_stable`: False
- `min_significant_pct`: 45.83%

This suggests the small sample robustness checks did NOT converge, which contradicts the paper's claim that "bootstrap convergence diagnostics... confirm that our estimates converge appropriately."

**Severity:** **MAJOR** - Results suggest bootstrap did NOT converge, contrary to paper claims.

---

## 8. Summary of Inconsistencies

| # | Issue | Severity | Location | Impact |
|---|-------|----------|----------|--------|
| 1 | IV results file corrupted (wrong values, negative F/R²) | **CRITICAL** | iv_analysis_results.csv | Core causal claim unsupported |
| 2 | Placebo test incomplete (only 2 iterations vs 100 claimed) | **MAJOR** | phase4_placebo_results.csv | Cannot verify p-value claim |
| 3 | CATE range upper bound incorrect (+0.33 claimed, all negative in data) | **MAJOR** | rebuttal_country_cates.csv | Misrepresents heterogeneity |
| 4 | Bootstrap convergence failed (False in file, claimed stable in paper) | **MAJOR** | small_sample_robustness.csv | Contradicts robustness claims |
| 5 | Pointwise significance 79.2% not pre-computed in files | MINOR | N/A | Minor verification issue |
| 6 | CATE minimum -4.35 vs -4.30 (minor rounding) | MINOR | rebuttal_country_cates.csv | Minor numerical difference |

---

## 9. Recommendations

### Immediate Actions Required:

1. **Regenerate IV Analysis Results** (CRITICAL)
   - The `iv_analysis_results.csv` file contains corrupted/placeholder data
   - Must re-run `scripts/phase4_iv_analysis.py` to generate correct IV estimates
   - Verify F-statistic is positive and R² is in [0,1] range

2. **Complete Placebo Test** (MAJOR)
   - Re-run placebo test with proper 100 iterations
   - Ensure random seed is set for reproducibility
   - Calculate proper p-value from distribution

3. **Correct CATE Range Claim** (MAJOR)
   - Paper claims range is -4.35 to +0.33
   - Actual range is -4.30 to -0.07 (all negative)
   - Update paper to reflect correct range

4. **Investigate Bootstrap Convergence** (MAJOR)
   - `small_sample_robustness.csv` shows bootstrap_converged=False
   - This contradicts paper's claim of convergence
   - Either fix convergence issues or revise paper claims

### Verification Actions:

5. **Add Pointwise Significance Calculation**
   - Pre-compute 79.2% figure and save to results file
   - Document calculation method in code

6. **Verify CATE Minimum**
   - Confirm -4.30 is correct minimum (vs -4.35 in paper)
   - Check if rounding or data update caused difference

---

## 10. Conclusion

While many results (Model Ladder, GATEs, Policy Exceptions, Interactions, Mediation) are **consistent** between the paper and result files, there are **critical issues** with:

1. **IV Analysis** - Results file is corrupted/invalid
2. **Placebo Test** - Incomplete data (2 vs 100 iterations)
3. **CATE Range** - Upper bound misreported (+0.33 vs actual all negative)
4. **Bootstrap Convergence** - Failed convergence contradicts paper claims

**Recommendation:** Do not submit paper until IV analysis is re-run and placebo test is completed. The core causal identification strategy (IV) lacks valid supporting results in the current files.

---

*Report generated by: consistency-auditor*
*Task: #3 - Paper-Results Consistency Audit*
