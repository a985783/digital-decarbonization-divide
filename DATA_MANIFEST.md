# Data Manifest (v3 - High Dimensional)

This dataset has been augmented to support High-Dimensional Causal Inference. It now contains **60 variables** (excluding country/year; includes the OECD flag) covering 8 thematic domains.

## 1. Core Variables
| Variable | Description | Source |
| :--- | :--- | :--- |
| `CO2_per_capita` | CO₂ emissions (metric tons per capita) | WDI |
| `ICT_exports` | ICT service exports (% of service exports) | WDI |

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

## 9. Sample Flags

*   `OECD`: Binary indicator for OECD membership in the sample.

---

## 10. Causal Forest Output (NEW)

**File**: `results/causal_forest_cate.csv`

| Variable | Description |
|----------|-------------|
| `country` | Country code (ISO 3-letter) |
| `year` | Year of observation |
| `CATE` | Conditional Average Treatment Effect (ICT → CO₂) |
| `CATE_LB` | 95% confidence interval lower bound |
| `CATE_UB` | 95% confidence interval upper bound |
| `Significant` | Boolean: CI does not cross zero |
| `Effect_Direction` | "Reducing CO2" or "Increasing CO2" |

### Key Statistics
- **N**: 960 observations
- **Significant heterogeneity**: 25.3%
- **CATE range**: −0.10 to +0.04 metric tons/capita

---
**Note**: All control variables are processed via MICE imputation. CATE predictions are generated by CausalForestDML with 2,000 trees.
