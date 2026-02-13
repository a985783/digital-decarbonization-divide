# The Digital Decarbonization Divide - Enhanced Research Package

Language / 语言: [English](README.md) | [中文](README.zh-CN.md)

[![Version](https://img.shields.io/badge/Version-2.0--Enhanced-blue)](CHANGELOG.md)
[![Methods](https://img.shields.io/badge/Methods-Causal%20Forest%20%7C%20DragonNet%20%7C%20IV-green)](docs/theoretical_model.pdf)
[![App](https://img.shields.io/badge/App-Streamlit-orange)](app.py)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-GitHub%20Ready-success)](REPRODUCIBILITY.md)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

**A comprehensive research package examining how digital capacity affects CO₂ emissions across 40 economies (2000-2023), featuring advanced causal inference methods, interactive visualization, and policy tools.**

🚀 **New in v2.0 Enhanced Edition**: DragonNet comparison, Oster sensitivity analysis, formal theoretical model, interactive Streamlit dashboard, and complete policy toolkit.

## Reproduce the Paper (GitHub Ready)

For a clean, deterministic reproduction flow on a fresh machine:

```bash
git clone <your-repo-url>
cd 数字脱碳鸿沟-完结_存档_GitHub
bash reproduce.sh
```

If you want stage-by-stage execution:

```bash
make setup
make verify
make analysis
make paper
```

See `REPRODUCIBILITY.md` for full details, expected outputs, and troubleshooting.

## 🎯 Research Overview

This enhanced research package provides comprehensive analysis of the "Digital Decarbonization Divide" - the asymmetric effects of digital capacity on carbon emissions across development levels.

### Key Findings
- **IV Estimate**: see latest reproducible values in `results/iv_analysis_results.csv`
- **Sweet Spot**: Middle-income economies show strongest decarbonization potential
- **Mediation**: 11.7% of effect operates through energy efficiency
- **Robustness**: Confirmed by Oster sensitivity (δ=1.01) and DragonNet comparison

### 🔬 Advanced Methods

1. **Causal Forest DML** (Primary)
   - 2,000 trees with honest splitting
   - GroupKFold cross-validation
   - GATE analysis with country-cluster bootstrap

2. **Instrumental Variable Strategy**
    - Lagged DCI (t-1) as instrument
    - First-stage diagnostics are generated in `results/iv_analysis_results.csv`
    - Anderson-Rubin weak-IV robust CI

3. **Oster Sensitivity Analysis** ⭐ New
   - Breakdown point δ = 1.01
   - Moderate robustness to omitted variable bias
   - Sensitivity contour plots

4. **DragonNet Comparison** ⭐ New
   - Deep learning-based causal inference
   - R² = 0.989 (superior prediction)
   - ATE = -1.95 (consistent with CF)

5. **Formal Theoretical Model** ⭐ New
   - Representative agent framework
   - 4 testable propositions
   - Theory-empirical mapping

### 📊 Interactive Dashboard

The Streamlit dashboard provides four main modules for analyzing how digital connectivity drives carbon emission reductions:

1. **Data Explorer** - Explore time series trends and variable relationships
2. **Causal Effects** - Visualize CATE distributions and GATE analysis
3. **Policy Simulator** - Simulate CO2 reduction impacts from DCI policy changes
4. **Country Comparison** - Side-by-side country analysis with policy recommendations

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Required Packages

```bash
pip install streamlit pandas numpy plotly
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

### Requirements File

Use the repository `requirements.txt` directly:

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
scipy>=1.10.0
scikit-learn>=1.3.0
PyYAML>=6.0
matplotlib>=3.7.0
seaborn>=0.12.0
statsmodels>=0.14.0
xgboost>=1.7.0
econml>=0.15.0
joblib>=1.3.0
```

### Academic Consistency Guard

Before submission, run:

```bash
python3 -m scripts.academic_consistency_guard
```

## Usage

### Running the Dashboard

1. Navigate to the project directory:
```bash
cd 数字脱碳鸿沟-完结_存档_GitHub
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. The dashboard will open automatically in your default web browser at:
```
http://localhost:8501
```

### Dashboard Modules

#### Module 1: Data Explorer

- **Country Selector**: Dropdown menu with 40 countries
- **Time Series Plot**: View DCI, CO2, and GDP trends over time
- **Scatter Plot Matrix**: Explore relationships between variables
- **Summary Statistics**: Key metrics for selected country

**How to use:**
1. Select a country from the dropdown
2. View time series trends in the main panel
3. Toggle between single country and all countries view for scatter plots
4. Review summary statistics at the bottom

#### Module 2: Causal Effects

- **CATE Distribution**: Histogram of treatment effects across countries
- **GATE Heatmap**: Grouped Average Treatment Effects by GDP/Institution
- **Linear vs Forest**: Comparison of model predictions

**How to use:**
1. Navigate through the three tabs
2. Hover over plots for detailed values
3. Review summary statistics for CATE distribution
4. Examine group patterns in the heatmap

#### Module 3: Policy Simulator

- **DCI Slider**: Adjust target DCI level (0 to 2)
- **Real-time Predictions**: See CO2 reduction estimates
- **95% Confidence Intervals**: Uncertainty bounds
- **Policy Recommendations**: Tailored advice based on context

**How to use:**
1. Select a country for simulation
2. Review current status metrics
3. Adjust the DCI slider to set target level
4. View predicted CO2 reduction and confidence intervals
5. Read the policy recommendation and implementation timeline

#### Module 4: Country Comparison

- **Side-by-side Comparison**: Two countries compared
- **Classification Display**: Leader/Catch-up/Exception labels
- **Policy Recommendations**: Country-specific strategies
- **Visual Charts**: Bar charts and time series comparison

**How to use:**
1. Select two countries to compare
2. Review classification and key metrics for each
3. Compare policy priorities and investment strategies
4. Examine visual comparison charts

## Data Sources

The dashboard uses three main data sources:

1. **`data/clean_data_v5_enhanced.csv`** - Main dataset with:
   - Country-year panel data (2000-2023)
   - DCI (Digital Connectivity Index)
   - CO2 per capita emissions
   - GDP per capita
   - Control variables

2. **`policy_toolkit/country_classification.csv`** - Country classifications:
   - Classification labels (Leader/Catch-up/Exception/Struggling)
   - Policy priorities
   - Investment strategies
   - Latest year indicators

3. **`results/causal_forest_cate.csv`** - Causal inference results:
   - CATE (Conditional Average Treatment Effect) estimates
   - Confidence intervals
   - Covariates for each observation

## 📁 Complete File Structure

```
数字脱碳鸿沟-完结_存档_GitHub/
│
├── 📄 Paper Files
│   ├── paper.tex                    # Enhanced LaTeX paper
│   ├── paper_original.tex           # Original paper (backup)
│   ├── paper_cn.tex                 # Chinese version
│   └── references.bib               # Bibliography
│
├── 🖥️ Interactive Application
│   ├── app.py                       # Streamlit dashboard
│   ├── app/utils.py                 # Helper functions
│   └── requirements.txt             # Python dependencies
│
├── 📊 Data
│   ├── clean_data_v5_enhanced.csv   # Enhanced dataset (77 columns)
│   └── temp_downloads/              # Raw WDI data
│
├── 🔬 Analysis Scripts
│   ├── oster_sensitivity.py         # Oster (2019) sensitivity analysis
│   ├── dragonnet_comparison.py      # DragonNet comparison
│   ├── feature_engineering.py       # Feature engineering
│   ├── enhance_visualizations.py    # Enhanced figure generation
│   ├── phase4_iv_analysis.py        # IV analysis
│   ├── phase4_placebo.py            # Placebo tests
│   └── ...                          # Other phase scripts
│
├── 📈 Results
│   ├── sensitivity_analysis.csv     # Oster analysis results
│   ├── dragonnet_comparison.csv     # DragonNet comparison results
│   ├── feature_comparison.csv       # Feature engineering comparison
│   ├── causal_forest_cate.csv       # Main causal forest results
│   └── figures/
│       ├── enhanced/                # Enhanced figures (8 plots)
│       └── *.png                    # Standard figures
│
├── 🎯 Policy Toolkit
│   ├── country_classification.csv   # 40-country classification
│   ├── policy_simulator.py          # Policy simulation engine
│   ├── policy_lookup_table.csv      # 160-scenario lookup
│   ├── policy_recommendations.json  # Detailed recommendations
│   └── sdg_alignment_report.md      # SDG mapping
│
├── 📚 Documentation
│   ├── theoretical_model.tex        # Formal theory (LaTeX)
│   ├── theoretical_model.pdf        # Compiled theory (9 pages)
│   ├── theory_empirical_mapping.md  # Theory-empirical links
│   ├── policy_experiment_design.md  # RCT design framework
│   ├── implementation_roadmap.md    # 36-month roadmap
│   ├── ethics_checklist.md          # IRB compliance
│   ├── writing_style_guide.md       # Writing guidelines
│   └── writing_optimization_log.md  # Changes log
│
└── 📋 CHANGELOG.md                  # Version history
```

## Troubleshooting

### Port Already in Use

If port 8501 is already in use, specify a different port:
```bash
streamlit run app.py --server.port 8502
```

### Data Loading Errors

Ensure all data files are in the correct locations:
- Check that CSV files exist in their respective directories
- Verify file permissions allow reading
- Check for proper CSV formatting

### Missing Dependencies

If you encounter import errors:
```bash
pip install --upgrade streamlit pandas numpy plotly
```

### Browser Not Opening Automatically

Manually navigate to:
```
http://localhost:8501
```

## Customization

### Adding New Countries

To add new countries to the dashboard:
1. Update the data files with new country data
2. Ensure consistent country naming across all files
3. Restart the Streamlit app

### Modifying Visualizations

Edit `app.py` to customize:
- Color schemes
- Chart types
- Layout configurations
- Additional metrics

### Changing Default Values

Modify the default selections in the `st.selectbox` and `st.slider` calls in `app.py`.

## Technical Details

### CATE Calculation

The Conditional Average Treatment Effect (CATE) represents the expected CO2 reduction from a one-unit increase in DCI for a given country context.

Formula used in policy simulator:
```
CO2 Reduction (%) = |CATE| * DCI Change * 100
```

### Confidence Intervals

95% confidence intervals are calculated assuming 15% standard error:
```
CI = Estimate +/- 1.96 * SE
SE = |CATE| * 0.15
```

### Country Classifications

- **Leader**: High DCI, strong institutions, significant CO2 reduction potential
- **Catch-up**: Medium DCI, improving institutions, moderate potential
- **Exception**: High DCI but unique circumstances affecting outcomes
- **Struggling**: Low DCI, weak institutions, needs international support
- **Potential**: High DCI but untapped emission reduction potential

## Citation

Citation metadata is available in `CITATION.cff`.

If using this project in research, please cite:

```text
Cui, Qingsong. (2026). The Digital Decarbonization Divide: Enhanced Research Package.
GitHub repository.
```

## Contact

- Qingsong Cui (Independent Researcher)
- Email: `qingsongcui9857@gmail.com`

## License

This project is released under the MIT License. See `LICENSE`.
