# The Digital Decarbonization Divide (Replication Package)

![Status](https://img.shields.io/badge/Status-Peer%20Review%20Ready-blue)

This repository contains the data and code to reproduce the findings of the paper:
**"The Digital Decarbonization Divide: Asymmetric Effects of Digital Capacity on CO₂ Emissions Across Socio-economic Capacity"**

---

## 🔑 Key Findings

Using a **Causal Forest DML** framework on 840 observations across 40 economies:

| Finding | Value |
|---------|-------|
| **IV Estimate (OrthoIV)** | **−1.91** metric tons/capita (95% CI: [-2.37, -1.46]) |
| **Naive Estimate (Linear)** | −1.54 metric tons/capita |
| **IV First-stage F-statistic** | **247.63** (Strong instrument) |
| **Placebo p-value** | **< 0.001** (Signal-to-Noise ratio ~23×) |
| Pointwise significant estimates | **79.2%** |
| CATE range | −4.35 to +0.33 metric tons/capita |
| CATE × Renewable Energy | **Positive correlation** (r = +0.56) |
| **Mediation (Energy Efficiency)** | **11.7%** of effect mediated |
| **Triple Interaction** | **p < 0.001** (DCI × Institution × Renewable) |
| Sample Coverage (Global) | 90% of GDP, ~100% of Emissions |

### The Core Insight: A "Divide" Exists

- **High-capacity economies**: DCI tends to reduce emissions
- **Low-capacity economies**: DCI shows weaker or indefinite effects
- **The divide is real**: Validated by highly significant interaction tests
- **Robustness**: Confirmed by IV strategy (Lagged DCI, F = 247.63) and Randomization Inference
- **Mechanism**: 11.7% of effect operates through improved energy efficiency
- **Policy Complementarity**: Triple interaction reveals renewable energy moderates institutional effects

## 📊 Main Visualizations

### The Digital Decarbonization Divide
![The Divide](results/figures/divide_plot_institution.png)

### Multi-Moderator Effects
![Moderator Panel](results/figures/moderator_effects_panel.png)

## 📂 Repository Structure

```
├── data/
│   ├── wdi_expanded_raw.csv           # Augmented WDI/WGI data (62 vars + country/year)
│   ├── clean_data_v4_imputed.csv      # Fold-safe MICE-imputed dataset (N=840)
├── scripts/
│   ├── analysis_config.py             # Config loader
│   ├── analysis_data.py               # Data preparation helpers
│   ├── dci.py                         # DCI construction (PCA)
│   ├── impute_mice.py                 # Fold-safe MICE imputation
│   ├── phase1_mvp_check.py            # ⭐ Heterogeneity verification
│   ├── phase1b_gdp_interaction.py     # GDP interaction check
│   ├── phase2_causal_forest.py        # ⭐ Causal Forest DML (main)
│   ├── phase3_visualizations.py       # ⭐ Publication-quality figures
│   ├── phase4_placebo.py              # Placebo Tests (Randomization Inference)
│   ├── phase4_iv_analysis.py          # ⭐ IV Strategy (Enhanced: Placebo IV + AR CI)
│   ├── phase5_mechanism.py            # Mechanism: Renewable Paradox
│   ├── phase5_mechanism_enhanced.py   # ⭐ Mediation + Triple Interaction
│   ├── phase6_external_validity.py    # Sample Representativeness
│   ├── phase7_dynamic_effects.py      # 🆕 Dynamic lag effects analysis
│   ├── pca_diagnostics.py             # 🆕 DCI measurement validation
│   ├── power_analysis.py              # 🆕 Monte Carlo power simulation
│   ├── rebuttal_analysis.py           # Model ladder + GATEs
│   ├── rebuttal_visualizations.py     # Rebuttal figures
│   ├── solve_wdi_v4_expanded_zip.py   # Data Download (WDI/WGI)
│   └── preflight_release_check.py     # Release sanity checks
├── analysis_spec.yaml                 # Single source of truth
├── results/
│   ├── causal_forest_cate.csv         # ⭐ Main results (CATE per obs)
│   ├── phase1_mvp_results.csv         # Interaction term results
│   ├── phase4_placebo_results.csv     # Placebo distribution
│   ├── iv_analysis_results.csv        # IV comparison + AR CI
│   ├── placebo_iv_results.csv         # 🆕 Placebo IV tests (t-2, t-3)
│   ├── pca_diagnostics.csv            # 🆕 PCA validation results
│   ├── dynamic_effects.csv            # 🆕 Lag effect estimates
│   ├── model_ladder.csv               # Model ladder summary
│   ├── rebuttal_gate.csv              # GATEs with cluster bootstrap
│   └── figures/
│       ├── divide_plot_institution.png # ⭐ Main figure
│       ├── divide_plot_gdp.png         # GDP moderation
│       ├── placebo_distribution.png    # Robustness: Placebo
│       ├── mechanism_renewable_curve.png # Mechanism: Renewable Curve
│       ├── sample_representativeness.png # External Validity
│       ├── cate_distribution.png       # CATE histogram
│       ├── country_average_cate.png    # Country comparison
│       └── moderator_effects_panel.png # ⭐ Multi-panel moderators
├── paper.tex                          # Paper (LaTeX, English)
├── paper_cn.tex                       # Paper (LaTeX, Chinese)
├── references.bib                     # Bibliography (Corrected & Verified)
├── DATA_MANIFEST.md                   # Variable definitions (62 vars)
└── requirements.txt                   # Dependencies
```

## 🚀 Reproduction Guide

### Prerequisites
Python 3.10+ recommended

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline

**Phase 1: Data Engineering**
```bash
python3 -m scripts.solve_wdi_v4_expanded_zip  # Download 62 vars
python3 -m scripts.impute_mice                # Fold-safe MICE Imputation
```

**Phase 2: Heterogeneity Verification**
```bash
python3 -m scripts.phase1_mvp_check           # ⭐ Interaction term test
```

**Phase 3: Causal Forest Analysis (Main)**
```bash
python3 -m scripts.phase2_causal_forest       # ⭐ Train Causal Forest (2000 trees)
```

**Phase 4: Identification & Robustness**
```bash
python3 -m scripts.phase4_placebo             # Placebo Tests (Randomization Inference)
python3 -m scripts.phase4_iv_analysis         # ⭐ IV Analysis + Placebo IV + AR CI
python3 -m scripts.small_sample_robustness    # Bootstrap convergence + Sample size sensitivity
```

**Phase 5: Mechanism Analysis**
```bash
python3 -m scripts.phase5_mechanism           # Mechanism Analysis (Renewable Paradox)
python3 -m scripts.phase5_mechanism_enhanced  # ⭐ Mediation + Triple Interaction
python3 -m scripts.phase6_external_validity   # External Validity Check
```

**Phase 6: Measurement Validation (Q1 Response)**
```bash
python3 -m scripts.pca_diagnostics            # 🆕 DCI construct validity
python3 -m scripts.power_analysis             # 🆕 Monte Carlo power analysis
python3 -m scripts.phase7_dynamic_effects     # 🆕 Dynamic lag effects
```

**Phase 7: Visualization**
```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp python3 -m scripts.phase3_visualizations  # Generate figures
```

## 📊 Data Summary

| Item | Details |
| :--- | :--- |
| **Source** | World Bank WDI & WGI |
| **Sample** | 40 economies, 2000–2023 |
| **Observations** | 840 (after excluding missing CO2 outcomes) |
| **Variables** | 62 variables (excluding country/year; includes OECD flag) |
| **Domains** | Institutions (6 WGI), Energy, Finance, Demographics |

*Note: `CO2_per_capita` is scaled by /100 when raw values exceed 100.*

## ⚠️ Methodology Notes

### Causal Forest Configuration
```python
CausalForestDML(
    model_y=XGBRegressor(),
    model_t=XGBRegressor(),
    n_estimators=2000,
    min_samples_leaf=10,
    max_depth=6,
    cv=GroupKFold(n_splits=5)  # Country-clustered cross-fitting
)
```

### Inference & Robustness
- **95% confidence intervals** via `effect_interval()`
- **Significance**: CI does not cross zero
- **IV Diagnostics**: First-stage F-statistic = 247.63 (strong instrument)
- **Small Sample**: Bootstrap convergence + sample size sensitivity analysis
- **Mechanisms**: Mediation analysis (Sobel test) + triple interaction tests

## 📄 Citation

```bibtex
@article{cui2026divide,
  title={The Digital Decarbonization Divide: Asymmetric Effects of ICT on CO₂ Emissions Across Institutional Regimes},
  author={Cui, Qingsong},
  journal={Working Paper},
  year={2026}
}
```

---

## References

- Athey, S. and Wager, S. (2019). Estimating treatment effects with causal forests. *Observational Studies*, 5(2), 37–51.
- Chernozhukov, V. et al. (2018). Double/debiased machine learning. *The Econometrics Journal*, 21(1), C1–C68.
- World Bank. (2025). *World Development Indicators*. Washington, D.C.

---

**Maintained by**: Qingsong Cui  
**Last Updated**: January 24, 2026
