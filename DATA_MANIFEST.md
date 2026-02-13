# Data Manifest (v5 - Enhanced Edition)

This enhanced dataset supports advanced causal inference analysis with **77 variables** (13 new engineered features). It covers 40 economies from 2000-2023 with comprehensive digital, economic, institutional, and environmental indicators.

## Version History
- **v5 (Current)**: Enhanced with 13 new features including interaction terms, polynomial terms, and institutional categories
- **v4**: High-dimensional dataset with 62 variables and DCI main treatment
- **v3**: Base dataset with imputed controls

## 1. Core Variables
| Variable | Description | Source |
| :--- | :--- | :--- |
| `CO2_per_capita` | CO₂ emissions (metric tons per capita) | WDI |
| `DCI` | Domestic Digital Capacity index (PCA of Internet users, Fixed broadband, Secure servers; mean 0, SD 1) | WDI (author computation) |
| `EDS` | ICT service exports (% of service exports) | WDI |

## 1.1 Units and Scaling
*   `CO2_per_capita`: Values in CSV may be stored at 100x WDI scale; analysis scripts rescale by `/100` when the mean exceeds 100 to restore metric tons per capita.
*   `DCI`: Standardized index (mean 0, SD 1) computed from WDI components during analysis (not stored in raw CSV).
*   `EDS`: Percentage points.

## 2. Institutional Quality (WGI)
*   `Control_of_Corruption`: Perceptions of public power exercised for private gain.
*   `Rule_of_Law`: Confidence in rules of society and contract enforcement.
*   `Political_Stability`: Likelihood of political instability/violence.
*   `Government_Effectiveness`: Quality of public services and policy formulation.
*   `Regulatory_Quality`: Ability to permit and promote private sector development.
*   `Voice_and_Accountability`: Freedom of expression and association.

## 3. Demographics & Social
*   `Age_dependency_ratio`: Ratio of dependents to working-age population.
*   `Tertiary_enrollment`: Gross enrollment ratio in tertiary education.
*   `Female_labor_participation`: % of female population ages 15+ in labor force.
*   `Population_total`: Total population.
*   `Urban_population_pct`: % of total population living in urban areas.
*   `Population_growth`: Annual population growth (%).
*   `Life_expectancy`: Life expectancy at birth, total (years).
*   `Mortality_rate_adult_male`: Mortality rate, adult, male (per 1,000 male adults).
*   `Govt_expenditure_education_pct_GDP`: Government expenditure on education (% of GDP).
*   `Current_health_expenditure_pct_GDP`: Current health expenditure (% of GDP).

## 4. Infrastructure & Digital
*   `Internet_users`: Individuals using the Internet (% of population).
*   `Mobile_cellular_subscriptions`: Subscriptions per 100 people.
*   `Fixed_telephone_subscriptions`: Subscriptions per 100 people.
*   `Fixed_broadband_subscriptions`: Fixed broadband subscriptions per 100 people.
*   `Secure_servers`: Secure Internet servers per 1 million people.
*   `Railways_route_km`: Length of railway lines.
*   `Container_port_traffic`: Twenty-foot equivalent units (TEUs).
*   `Air_freight_million_ton_km`: Air freight volume.
*   `Electric_power_losses`: Transmission and distribution losses (% of output).

## 5. Financial Depth
*   `Domestic_credit_to_private_sector`: Financial resources provided to private sector (% of GDP).
*   `Market_capitalization`: Market cap of listed domestic companies (% of GDP).
*   `Broad_money_pct_GDP`: Money supply (M3/GDP).
*   `Lending_interest_rate`: Rate charged by banks on prime customers.
*   `Inflation_consumer_prices`: Annual % change in CPI.
*   `Inflation_GDP_deflator`: GDP deflator (annual %).

## 6. Macroeconomic Structure
*   `GDP_per_capita_current`: Gross Domestic Product per capita (Current US$).
*   `GDP_per_capita_constant`: Gross Domestic Product per capita (Constant 2015 US$).
*   `GDP_growth`: Annual GDP growth (constant prices, %).
*   `Trade_openness`: (Exports + Imports) / GDP.
*   `FDI_net_inflows_pct_GDP`: Foreign Direct Investment, net inflows.
*   `Current_account_balance_pct_GDP`: Current account balance (% of GDP).
*   `Manufacturing_value_added_pct_GDP`: Manufacturing sector size.
*   `Services_value_added_pct_GDP`: Service sector size.
*   `Gross_capital_formation_pct_GDP`: Investment rate.
*   `Gross_savings_pct_GDP`: Gross savings (% of GDP).
*   `Exports_pct_GDP`: Exports of goods and services (% of GDP).
*   `Imports_pct_GDP`: Imports of goods and services (% of GDP).
*   `Industry_value_added_pct_GDP`: Industry value added (% of GDP).
*   `Agriculture_value_added_pct_GDP`: Agriculture, forestry, and fishing value added (% of GDP).
*   `Central_govt_debt_pct_GDP`: Debt-to-GDP ratio.

## 7. Energy & Environment
*   `Energy_use_per_capita`: Kg of oil equivalent per capita.
*   `Renewable_energy_consumption_pct`: % of total final energy consumption.
*   `Access_to_electricity_pct`: % of population with access.
*   `Methane_emissions_kt_CO2_eq`: Methane emissions.
*   `Nitrous_oxide_emissions_kt_CO2_eq`: N2O emissions.
*   `Forest_area_pct`: Forest area (% of land area).
*   `Agricultural_land_pct`: Agricultural land (% of land area).
*   `Arable_land_pct`: Arable land (% of land area).

## 8. Innovation
*   `Research_and_development_expenditure_pct_GDP`: R&D spending.
*   `High_tech_exports_pct_mfg_exports`: Tech intensity of exports.
*   `Scientific_journal_articles`: Scientific and technical journal articles.
*   `Patent_applications_residents`: Innovation output.
*   `Patent_applications_nonresidents`: Patent applications by nonresidents.

---

## 9. Enhanced Features (v5)

The enhanced dataset (`clean_data_v5_enhanced.csv`) includes **13 additional engineered features**:

### 9.1 Core Digital Connectivity Index
*   `DCI`: Computed via PCA from Internet_users, Fixed_broadband_subscriptions, and Secure_servers (explained variance: 70.15%)

### 9.2 Non-linear and Interaction Terms
*   `DCI_squared`: Quadratic term capturing diminishing marginal returns to digitalization
*   `log_GDP_per_capita`: Natural logarithm of GDP per capita
*   `DCI_x_Trade_Openness`: Interaction between digital capacity and trade openness
*   `DCI_x_Financial_Development`: Interaction with financial depth (Domestic_credit_to_private_sector)
*   `DCI_x_Education_Level`: Interaction with tertiary enrollment
*   `log_GDP_x_DCI_squared`: Three-way interaction capturing income-digitalization non-linearity

### 9.3 Institutional Quality Categories
*   `Institution_quality_index`: Average of 6 WGI indicators (mean 0, SD 1)
*   `Institution_quality_category`: Categorical (High/Medium/Low)
*   `Institution_quality_high`: Dummy for high institutional quality (top 33%)
*   `Institution_quality_medium`: Dummy for medium institutional quality (middle 33%)
*   `Institution_quality_low`: Dummy for low institutional quality (bottom 33%)

---

## 10. Sample Flags

*   `OECD`: Binary indicator for OECD membership in the sample.

---

## 11. Causal Forest Output (v4)

**File**: `results/causal_forest_cate.csv`

| Variable | Description |
|----------|-------------|
| `country` | Country code (ISO 3-letter) |
| `year` | Year of observation |
| `CATE` | Conditional Average Treatment Effect (DCI → CO₂) |
| `CATE_LB` | 95% confidence interval lower bound |
| `CATE_UB` | 95% confidence interval upper bound |
| `Significant` | Boolean: CI does not cross zero |
| `Effect_Direction` | "Reducing CO2" or "Increasing CO2" |

### Key Statistics
- **N**: 840 observations (CO2 missing outcomes excluded; no imputation on Y/T)
- **Significant heterogeneity**: 79.2%
- **CATE range**: −4.35 to +0.33 metric tons/capita (per 1 SD DCI)

---
**Note**: Control variables and moderators are processed via fold-safe MICE imputation. CATE predictions are generated by CausalForestDML with 2,000 trees.

---

## 12. Enhanced Analysis Results (v5)

### 12.1 Sensitivity Analysis
**File**: `results/sensitivity_analysis.csv`

Oster (2019) sensitivity analysis results:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Representative Delta | 1.01 | Omitted variables must be 1.01x as important as observed controls to explain away the result |
| IV Coefficient | -273.71 | Consistent with main analysis |
| R² (Controlled) | 0.78 | Model explains 78% of variance |

**Status**: δ > 1.0 indicates **moderate robustness** to omitted variable bias.

### 12.2 DragonNet Comparison
**File**: `results/dragonnet_comparison.csv`

Comparison of causal inference methods:

| Method | ATE | R²_Y | Key Advantage |
|--------|-----|------|---------------|
| DragonNet | -1.95 | 0.989 | Superior prediction (MSE=0.20) |
| CausalForestDML | -3.29 | - | Standard errors & CIs |
| LinearDML | -2.51 | - | Baseline linear model |

### 12.3 Feature Engineering Comparison
**File**: `results/feature_comparison.csv`

Baseline vs. enhanced model comparison:

| Metric | Baseline | Enhanced | Change |
|--------|----------|----------|--------|
| CATE Mean | -1.30 | -1.39 | Better alignment with IV |
| CATE Std | 0.90 | 2.97 | Captures more heterogeneity |
| Features | 6 | 15 | +9 engineered features |

### 12.4 Enhanced Visualizations
**Directory**: `results/figures/enhanced/`

Publication-ready figures following Nature/Science standards:
- `divide_plot_gdp_enhanced.png/pdf`: With confidence interval bands
- `gate_plot_enhanced.png/pdf`: Enhanced with error bars
- `gate_heatmap_*_enhanced.png/pdf` (3 files): Multidimensional heterogeneity
- `linear_vs_forest_enhanced.png/pdf`: With significance markers
- `mechanism_renewable_curve_enhanced.png/pdf`: Contour + bar chart
- `placebo_distribution_enhanced.png/pdf`: Optimized density curve

---

## 13. Key WDI Indicator Codes

| Indicator | WDI Code | Notes |
| :--- | :--- | :--- |
| CO2 per capita | `EN.ATM.CO2E.PC` | Outcome |
| ICT service exports (EDS) | `BX.GSR.CCIS.ZS` | Secondary dimension |
| Internet users | `IT.NET.USER.ZS` | DCI component |
| Fixed broadband subscriptions | `IT.NET.BBND.P2` | DCI component |
| Secure servers | `IT.NET.SECR.P6` | DCI component |

**DCI Construction**: PCA over the three DCI components after standardization, then re-standardized to mean 0 and SD 1.
