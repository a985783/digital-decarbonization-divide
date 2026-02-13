# Policy Experiment Design for Digital-Climate Interventions

## Executive Summary

This document presents three experimental design options for evaluating the causal impact of digital infrastructure investments on carbon emissions reduction. Based on feasibility, internal validity, and policy relevance, **Option B (Field Experiment)** is recommended for primary implementation, with Option A (Quasi-experiment) as a complementary validation strategy.

---

## Option A: Quasi-Experimental Design

### Overview
Leverage historical rollout of digital infrastructure (broadband, 5G, data centers) to construct a natural experiment using difference-in-differences (DiD) methodology.

### Design Specification

**Treatment Definition:**
- Treatment: Regions/countries that received significant digital infrastructure investment (>50% increase in DCI score)
- Control: Matched regions with similar pre-trends but no major investment
- Treatment timing: Staggered adoption based on actual rollout dates

**Identification Strategy:**
```
Y_it = α + β(DCI_it) + γ_i + δ_t + ε_it
```
Where:
- Y_it: CO2 emissions per capita in region i, time t
- DCI_it: Digital infrastructure treatment indicator
- γ_i: Region fixed effects
- δ_t: Time fixed effects

**Parallel Trends Assumption:**
- Verify pre-treatment trends in CO2 emissions are parallel
- Use event study specification to test for dynamic effects
- Implement Callaway-Sant'Anna estimator for staggered treatment timing

**Data Requirements:**
- Panel data: 2000-2023 for 150+ countries
- DCI measures: ITU database, World Bank Digital Development indicators
- Outcomes: CO2 emissions (EDGAR), energy consumption (IEA)
- Covariates: GDP, population, renewable energy share, institutional quality

**Strengths:**
- Large sample size (n>100 countries)
- No implementation costs
- Real-world policy relevance
- Can study long-term effects (10+ years)

**Limitations:**
- Selection bias: Digital investments may target high-growth regions
- Confounding: Correlated with other development policies
- Parallel trends assumption may be violated
- Limited control over treatment intensity

---

## Option B: Field Experiment (RECOMMENDED)

### Overview
Partner with 2-3 pilot countries to implement randomized digital infrastructure interventions, measuring direct causal effects on energy consumption and emissions.

### Design Specification

**Treatment Arms:**
1. **Control:** Business-as-usual (no intervention)
2. **Treatment A:** Smart grid infrastructure (advanced metering, demand response)
3. **Treatment B:** Digital agriculture extension (precision farming apps, IoT sensors)
4. **Treatment C:** Combined package (A + B)

**Unit of Randomization:**
- Primary: District/municipality level (n=120 units across 3 countries)
- Secondary: Households within treated districts (n=2,400 households)

**Randomization Strategy:**
```
Stratified by:
- Baseline CO2 emissions (terciles)
- GDP per capita (above/below median)
- Urbanization rate (high/low)
- Geographic region (fixed effects)

Within each stratum: Block randomization
```

**Outcome Variables:**

| Level | Primary Outcomes | Secondary Outcomes | Measurement |
|-------|------------------|-------------------|-------------|
| District | CO2 emissions (tons) | Energy intensity, renewable share | Administrative data |
| Firm | Energy consumption (kWh) | Productivity, output | Smart meters, surveys |
| Household | Electricity use (kWh) | DCI adoption, behavior change | Smart meters, apps |

**Timeline:**
- Month 0-6: Baseline measurement, randomization
- Month 6-12: Intervention rollout (pilot phase)
- Month 12-24: Full implementation
- Month 24-30: Follow-up measurement

**Sample Size Calculation:**
See `experiment_power_analysis.R` for detailed calculations.
- Target: 120 districts (40 per country)
- Power: 80% to detect 15% reduction in emissions
- Significance: α = 0.05 (two-tailed)
- ICC (intra-cluster correlation): 0.15
- Attrition allowance: 20%

**Analysis Plan:**
```
Intent-to-Treat (ITT):
Y_ij = β_0 + β_1*Treat_j + X'_ij*γ + δ_c + ε_ij

Treatment-on-Treated (TOT):
Instrument treatment receipt with assignment
```

**Spillover Controls:**
- Geographic buffer zones between treatment/control districts
- Monitor cross-district migration of firms
- Track technology diffusion patterns

**Strengths:**
- Gold standard causal identification
- Control over treatment design and intensity
- Direct measurement of mechanisms
- Policy scalability assessment

**Limitations:**
- High implementation cost ($2-5M)
- Political economy challenges
- External validity concerns (specific contexts)
- Long timeline (30 months)

---

## Option C: Discrete Choice Experiment

### Overview
Use conjoint survey design to elicit stakeholder preferences for digital-climate policies and simulate adoption behavior under different policy scenarios.

### Design Specification

**Survey Population:**
- Primary: Policymakers in 30 developing countries (n=300)
- Secondary: Firm managers, urban households (n=1,500 each)

**Conjoint Design:**
```
Attributes (6 dimensions, 2-4 levels each):
1. Policy type: Smart grid / Digital agriculture / E-mobility / Green IT
2. Investment scale: $10M / $50M / $100M / $500M
3. Financing: Domestic / International grant / Carbon market / PPP
4. Implementation: Government-led / Private sector / Mixed
5. Timeline: 2 years / 5 years / 10 years
6. Co-benefits: Jobs / Energy access / Health / None

Profiles: 2 per choice task
Tasks: 8 per respondent
Design: D-efficient, 100 unique sets
```

**Choice Questions:**
"Which policy package would you recommend for your country?"
- Option A: [Randomized profile]
- Option B: [Randomized profile]
- Neither

**Analysis Framework:**
```
Multinomial Logit:
U_ij = β*X_ij + ε_ij

Heterogeneous effects (Mixed Logit):
β_k ~ N(μ_k, σ_k²)

Marginal Willingness-to-Pay:
MWTP = -β_attribute / β_cost
```

**Simulation Component:**
- Agent-based model of technology adoption
- Calibrated using estimated preference parameters
- Simulate diffusion under different policy scenarios

**Strengths:**
- Low cost ($50K-100K)
- Rapid implementation (3-6 months)
- Can test counterfactual policies
- Scalable to many contexts

**Limitations:**
- Stated preferences vs. revealed behavior gap
- Hypothetical bias
- No direct causal estimates
- Limited external validity

---

## Recommended Approach: Hybrid Design

### Primary: Field Experiment (Option B)
Implement randomized controlled trial in 2-3 countries with:
- 120 districts randomized to treatment/control
- Smart grid and digital agriculture interventions
- 30-month timeline with comprehensive measurement

### Complementary: Quasi-Experiment (Option A)
- Validate findings using historical rollout data
- Assess external validity across contexts
- Estimate long-term effects (10+ years)

### Exploratory: Discrete Choice (Option C)
- Inform intervention design through preference elicitation
- Simulate policy adoption scenarios
- Guide scale-up strategy

---

## Detailed Design: Field Experiment (Option B)

### Treatment vs. Control Specification

**Treatment Group (n=72 districts):**
- Smart Grid Intervention (n=24):
  - Deploy advanced metering infrastructure (AMI)
  - Implement demand response programs
  - Real-time pricing signals
  - Grid optimization algorithms

- Digital Agriculture (n=24):
  - Precision farming mobile applications
  - IoT soil and weather sensors
  - Digital extension services
  - Market linkage platforms

- Combined (n=24):
  - Both interventions integrated
  - Cross-sectoral data sharing
  - Coordinated implementation

**Control Group (n=48 districts):**
- No intervention during study period
- Commitment to receive delayed treatment (30 months)
- Regular data collection only

### Stratified Randomization Protocol

**Stratification Variables:**
1. **Baseline Emissions** (terciles): Low / Medium / High
2. **Economic Development** (binary): Above/below median GDP/capita
3. **Urbanization** (binary): Above/below 50% urban
4. **Country** (fixed): 3 countries = 3 strata

**Randomization Procedure:**
```python
# Pseudo-code for stratified randomization
for country in [CountryA, CountryB, CountryC]:
    for emission_level in [Low, Medium, High]:
        for gdp_group in [Above, Below]:
            for urban_group in [Above, Below]:
                block = get_districts(country, emission_level, gdp_group, urban_group)
                if len(block) >= 4:
                    assign_randomly(block, ratios=[2:1:1:2])
                    # Control:TreatmentA:TreatmentB:TreatmentC
```

**Balance Checks:**
- Compare baseline characteristics across arms
- Standardized mean differences < 0.1
- Re-randomize if severe imbalance detected

### Outcome Variables Definition

**Primary Outcomes:**

1. **CO2 Emissions (District Level)**
   - Definition: Total metric tons CO2 equivalent per year
   - Sources: Energy, agriculture, transport, industry
   - Measurement: Satellite data (OCO-2, GOSAT) + ground sensors
   - Frequency: Monthly
   - Precision target: ±5%

2. **Energy Consumption (Firm Level)**
   - Definition: Total kWh consumed per firm per month
   - Measurement: Smart meters with 15-minute granularity
   - Subcategories: Peak/off-peak, by end-use
   - Sample: 50 firms per district

3. **Electricity Use (Household Level)**
   - Definition: Monthly kWh per household
   - Measurement: Smart meters + utility billing data
   - Sample: 20 households per district

**Secondary Outcomes:**

4. **Renewable Energy Share**
   - % of total energy from renewable sources
   - Source: Grid operator data

5. **Energy Intensity**
   - kWh per unit of economic output
   - Source: Firm surveys + administrative data

6. **Digital Technology Adoption**
   - % of firms/households using digital tools
   - Source: App analytics, surveys

7. **Behavioral Change Index**
   - Composite of energy-saving behaviors
   - Source: Household surveys

**Mechanism Variables:**

8. **Information Access**
   - Frequency of energy data checking
   - Source: App usage logs

9. **Price Responsiveness**
   - Elasticity of demand to price signals
   - Source: Smart meter data

10. **Social Norms**
    - Perceived community energy-saving norms
    - Source: Survey questions

### Timeline: 6-Month Pilot + 18-Month Main

**Phase 1: Pilot (Months 0-6)**

| Month | Activity | Deliverable |
|-------|----------|-------------|
| 0-1 | Site selection, partnership agreements | MOUs signed |
| 1-2 | Baseline data collection | Baseline dataset |
| 2 | Stratified randomization | Randomization list |
| 2-3 | Treatment arm preparation | Implementation plans |
| 3-6 | Pilot intervention rollout | Pilot evaluation report |
| 6 | Pilot assessment, design refinement | Revised protocol |

**Pilot Success Criteria:**
- >90% treatment compliance
- <10% attrition
- No major adverse events
- Feasible data collection protocols

**Phase 2: Main Experiment (Months 6-24)**

| Period | Activity |
|--------|----------|
| Months 6-9 | Full intervention rollout |
| Months 9-12 | Intensive monitoring period |
| Months 12-15 | Midline assessment |
| Months 15-21 | Sustained implementation |
| Months 21-24 | Endline data collection |

**Phase 3: Follow-up (Months 24-30)**
- Sustainability assessment
- Cost-effectiveness analysis
- Scale-up recommendations

### Data Collection Protocol

**Administrative Data (Monthly):**
- Grid-level electricity consumption
- CO2 emissions estimates (satellite)
- Economic indicators

**Survey Data:**
- Baseline (Month 2): Full household and firm surveys
- Midline (Month 12): Abbreviated surveys
- Endline (Month 24): Full surveys

**Sensor Data (Real-time):**
- Smart meter readings (15-min intervals)
- Weather stations (hourly)
- IoT sensor networks (agriculture)

**Quality Assurance:**
- 10% random audit of surveys
- Real-time data validation
- Weekly data quality reports
- Independent monitoring visits

---

## Budget Estimate

| Component | Cost (USD) |
|-----------|------------|
| Smart grid infrastructure | $1,200,000 |
| Digital agriculture platform | $800,000 |
| Data collection & sensors | $400,000 |
| Research team (3 years) | $600,000 |
| Government partnerships | $200,000 |
| Evaluation & analysis | $300,000 |
| Contingency (15%) | $525,000 |
| **Total** | **$4,025,000** |

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Political interference | Medium | High | Multi-country design, MOU protections |
| Technology failure | Low | Medium | Redundant systems, vendor contracts |
| Attrition | Medium | Medium | Oversampling, tracking protocols |
| Spillovers | Medium | Medium | Buffer zones, monitoring |
| Natural disasters | Low | High | Insurance, flexible timeline |

---

## References

1. Duflo, E., & Banerjee, A. (2017). Handbook of Field Experiments
2. Gerber, A., & Green, D. (2012). Field Experiments: Design, Analysis, and Interpretation
3. Callaway, B., & Sant'Anna, P. (2021). Difference-in-differences with multiple time periods
4. Athey, S., & Imbens, G. (2017). The econometrics of randomized experiments
5. Hainmueller, J., et al. (2014). Causal inference in conjoint analysis
