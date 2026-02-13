# Digital Carbon Divide Dashboard - Preview

## App Screenshot Description

The Streamlit dashboard has been created with 4 modules:

### Module 1: Data Explorer
- **Left sidebar**: Country selector dropdown with 40 countries (Argentina, Australia, Austria, Bangladesh, Belgium, Brazil, Canada, Switzerland, China, Colombia, Germany, Denmark, Egypt, Spain, Finland, France, UK, Indonesia, India, Ireland, Italy, Japan, South Korea, Mexico, Malaysia, Nigeria, Netherlands, Norway, New Zealand, Pakistan, Peru, Philippines, Russia, Saudi Arabia, Sweden, Thailand, Turkey, USA, Vietnam, South Africa)
- **Main panel**: Time series plots showing DCI, CO2 per capita, and GDP per capita trends
- **Scatter plot matrix**: Interactive visualization of DCI vs CO2 vs GDP relationships
- **Summary statistics**: Key metrics displayed as cards

### Module 2: Causal Effects
- **Tab 1 - CATE Distribution**: Histogram of Conditional Average Treatment Effects with mean line and zero reference line
- **Tab 2 - GATE Heatmap**: Color-coded heatmap showing average CATE by GDP groups (Low, Lower-Mid, Upper-Mid, High) and Institution levels
- **Tab 3 - Linear vs Forest**: Scatter plot of CATE vs DCI with trend line, colored by GDP per capita

### Module 3: Policy Simulator
- **Left panel**: 
  - Country selector
  - Current status metrics (DCI, CO2, CATE)
  - DCI slider (0 to 2.0)
- **Right panel**:
  - Simulation results with CO2 reduction percentage
  - 95% confidence interval bar chart
  - Policy recommendation box
  - Implementation timeline table

### Module 4: Country Comparison
- **Two country selectors** side by side
- **Country cards** showing:
  - Classification (Leader/Catch-up/Exception/Struggling/Potential)
  - DCI, CO2, GDP metrics
  - CATE estimate
  - Renewable share
- **Policy recommendations** for each country
- **Visual comparison** bar charts
- **Historical trends** time series comparison

## UI Styling
- Clean white background with plotly white template
- Color scheme: Blue (#1f77b4) for primary, Red (#d62728) for CO2, Green (#2ca02c) for GDP
- Card-based layout with rounded corners
- Responsive design with columns and tabs
- Custom CSS for metric cards and recommendation boxes

## How to View the Actual Screenshot

1. Install dependencies:
```bash
pip install streamlit pandas numpy plotly
```

2. Run the app:
```bash
cd /Users/cuiqingsong/Documents/数字脱碳鸿沟-完结_存档
streamlit run app.py
```

3. Open browser to http://localhost:8501

4. Take screenshots of each module
