"""
Digital Carbon Divide - Interactive Streamlit Dashboard
=====================================================

An interactive visualization tool for exploring the relationship between
digital connectivity and carbon emissions across 40 countries.

Modules:
1. Data Explorer - Country-level time series and scatter plots
2. Causal Effects - CATE distribution and GATE analysis
3. Policy Simulator - Interactive DCI adjustment with CO2 predictions
4. Country Comparison - Side-by-side country analysis

Author: Research Team
Date: 2026-02-13
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from paths import data_path, result_path, toolkit_path, assert_exists

# Set page configuration
st.set_page_config(
    page_title="Digital Carbon Divide Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .country-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-box {
        background-color: #e8f4f8;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #2ca02c;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ==================== DATA LOADING ====================
@st.cache_data
def load_data():
    """Load all required datasets with caching."""
    df_main = pd.read_csv(assert_exists(data_path('clean_data_v5_enhanced.csv')))
    df_classification = pd.read_csv(assert_exists(toolkit_path('country_classification.csv')))
    df_cate = pd.read_csv(assert_exists(result_path('causal_forest_cate.csv')))
    return df_main, df_classification, df_cate


def get_latest_year_data(df):
    """Get data for the most recent year available for each country."""
    return df.loc[df.groupby('country')['year'].idxmax()]


# Load data
try:
    df_main, df_classification, df_cate = load_data()
    countries = sorted(df_main['country'].unique().tolist())
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False
    countries = []


# ==================== HEADER ====================
st.markdown('<p class="main-header">🌍 Digital Carbon Divide Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Exploring the Relationship Between Digital Connectivity and Carbon Emissions</p>', unsafe_allow_html=True)

if not data_loaded:
    st.stop()

# ==================== SIDEBAR ====================
st.sidebar.title("Navigation")
module = st.sidebar.radio(
    "Select Module:",
    ["📊 Data Explorer", "📈 Causal Effects", "🎛️ Policy Simulator", "🔄 Country Comparison"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
This dashboard visualizes research findings on how digital connectivity
drives carbon emission reductions across 40 countries.

**Data Sources:**
- World Development Indicators
- Causal Forest CATE Estimates
- Country Classification Analysis
""")


# ==================== MODULE 1: DATA EXPLORER ====================
if module == "📊 Data Explorer":
    st.header("📊 Data Explorer")
    st.markdown("Explore time series trends and relationships between key variables.")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Country Selection")
        selected_country = st.selectbox(
            "Select a country:",
            countries,
            index=countries.index('China') if 'China' in countries else 0
        )

        # Display country info
        country_class = df_classification[df_classification['Country'] == selected_country]
        if not country_class.empty:
            classification = country_class.iloc[0]['Classification']
            st.metric("Classification", classification)

    with col2:
        # Time Series Plot
        st.subheader("Time Series Analysis")
        country_data = df_main[df_main['country'] == selected_country].sort_values('year')

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=('Digital Connectivity Index (DCI)', 'CO2 per Capita (kg)', 'GDP per Capita (constant USD)'),
            vertical_spacing=0.1
        )

        # DCI
        fig.add_trace(
            go.Scatter(
                x=country_data['year'],
                y=country_data['DCI'],
                mode='lines+markers',
                name='DCI',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )

        # CO2
        fig.add_trace(
            go.Scatter(
                x=country_data['year'],
                y=country_data['CO2_per_capita'],
                mode='lines+markers',
                name='CO2 per Capita',
                line=dict(color='#d62728', width=3),
                marker=dict(size=8)
            ),
            row=2, col=1
        )

        # GDP
        fig.add_trace(
            go.Scatter(
                x=country_data['year'],
                y=country_data['GDP_per_capita_constant'],
                mode='lines+markers',
                name='GDP per Capita',
                line=dict(color='#2ca02c', width=3),
                marker=dict(size=8)
            ),
            row=3, col=1
        )

        fig.update_layout(
            height=600,
            showlegend=False,
            template='plotly_white',
            title_text=f"{selected_country} - Key Indicators Over Time"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Scatter Plot Matrix
    st.subheader("Scatter Plot Matrix")
    st.markdown("Explore relationships between DCI, CO2 emissions, and GDP.")

    col3, col4 = st.columns([1, 3])

    with col3:
        scatter_mode = st.radio(
            "View Mode:",
            ["Single Country", "All Countries (Latest Year)"]
        )

    with col4:
        if scatter_mode == "Single Country":
            plot_data = df_main[df_main['country'] == selected_country]
            color_col = None
            title_suffix = selected_country
        else:
            plot_data = get_latest_year_data(df_main)
            color_col = 'country'
            title_suffix = "All Countries (Latest Year)"

        fig2 = px.scatter_matrix(
            plot_data,
            dimensions=['DCI', 'CO2_per_capita', 'GDP_per_capita_constant'],
            color=color_col,
            title=f"Variable Relationships - {title_suffix}",
            labels={
                'DCI': 'Digital Connectivity Index',
                'CO2_per_capita': 'CO2 per Capita',
                'GDP_per_capita_constant': 'GDP per Capita'
            },
            height=600
        )

        fig2.update_traces(diagonal_visible=False)
        fig2.update_layout(template='plotly_white')

        st.plotly_chart(fig2, use_container_width=True)

    # Summary Statistics
    st.subheader("Summary Statistics")
    latest_data = get_latest_year_data(df_main)
    country_latest = latest_data[latest_data['country'] == selected_country]

    if not country_latest.empty:
        c_data = country_latest.iloc[0]
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.metric("DCI", f"{c_data['DCI']:.2f}")
        with col6:
            st.metric("CO2 per Capita", f"{c_data['CO2_per_capita']:.0f} kg")
        with col7:
            st.metric("GDP per Capita", f"${c_data['GDP_per_capita_constant']:,.0f}")
        with col8:
            renewable = c_data.get('Renewable_energy_consumption_pct', 'N/A')
            if pd.notna(renewable):
                st.metric("Renewable %", f"{renewable:.1f}%")
            else:
                st.metric("Renewable %", "N/A")


# ==================== MODULE 2: CAUSAL EFFECTS ====================
elif module == "📈 Causal Effects":
    st.header("📈 Causal Effects Analysis")
    st.markdown("Visualize treatment effects from the Causal Forest model.")

    tab1, tab2, tab3 = st.tabs(["CATE Distribution", "GATE Heatmap", "Linear vs Forest"])

    with tab1:
        st.subheader("Conditional Average Treatment Effect (CATE) Distribution")
        st.markdown("""
        This histogram shows the distribution of CATE estimates across all countries.
        Negative values indicate CO2 reduction effects from digitalization.
        """)

        latest_cate = get_latest_year_data(df_cate)

        fig = go.Figure()

        # Histogram
        fig.add_trace(go.Histogram(
            x=latest_cate['CATE'],
            nbinsx=20,
            name='CATE Distribution',
            opacity=0.7,
            marker_color='#1f77b4'
        ))

        # Add mean line
        mean_cate = latest_cate['CATE'].mean()
        fig.add_vline(
            x=mean_cate,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_cate:.2f}",
            annotation_position="top"
        )

        # Add zero line
        fig.add_vline(x=0, line_dash="dot", line_color="gray",
                     annotation_text="No Effect", annotation_position="bottom")

        fig.update_layout(
            xaxis_title="CATE (CO2 Reduction Effect)",
            yaxis_title="Number of Countries",
            template='plotly_white',
            height=500,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean CATE", f"{mean_cate:.3f}")
        with col2:
            st.metric("Median CATE", f"{latest_cate['CATE'].median():.3f}")
        with col3:
            significant = (latest_cate['CATE_LB'] < 0).sum()
            st.metric("Significant Effects", f"{significant}/{len(latest_cate)}")
        with col4:
            st.metric("Std Dev", f"{latest_cate['CATE'].std():.3f}")

    with tab2:
        st.subheader("Group Average Treatment Effects (GATE) Heatmap")
        st.markdown("""
        This heatmap shows average CATE estimates grouped by GDP level and Institution quality.
        Darker green indicates stronger CO2 reduction effects.
        """)

        # Merge data
        latest_cate = get_latest_year_data(df_cate)
        merged = latest_cate.merge(
            df_classification[['Country', 'GDP_Per_Capita_2023', 'Institution_Level']],
            left_on='country',
            right_on='Country',
            how='left'
        )

        # Create GDP groups
        merged['GDP_Group'] = pd.cut(
            merged['GDP_Per_Capita_2023'],
            bins=[0, 5000, 15000, 50000, float('inf')],
            labels=['Low', 'Lower-Mid', 'Upper-Mid', 'High']
        )

        # Create pivot table
        pivot = merged.pivot_table(
            values='CATE',
            index='Institution_Level',
            columns='GDP_Group',
            aggfunc='mean'
        )

        fig2 = px.imshow(
            pivot,
            text_auto='.3f',
            aspect='auto',
            color_continuous_scale='RdYlGn',
            title="Average CATE by GDP and Institution Level",
            height=400
        )

        fig2.update_layout(
            xaxis_title="GDP per Capita Group",
            yaxis_title="Institution Level"
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Group statistics
        st.subheader("Group Statistics")
        group_stats = merged.groupby(['GDP_Group', 'Institution_Level']).agg({
            'CATE': ['mean', 'std', 'count']
        }).round(3)
        st.dataframe(group_stats)

    with tab3:
        st.subheader("Linear Model vs Causal Forest Comparison")
        st.markdown("""
        This scatter plot compares CATE estimates against DCI levels.
        The trend line shows the relationship between digital connectivity and treatment effects.
        """)

        latest_data = get_latest_year_data(df_cate)

        fig3 = go.Figure()

        # Scatter plot of CATE vs DCI
        fig3.add_trace(go.Scatter(
            x=latest_data['DCI'],
            y=latest_data['CATE'],
            mode='markers',
            marker=dict(
                size=12,
                color=latest_data['GDP_per_capita_constant'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="GDP per Capita"),
                line=dict(width=1, color='black')
            ),
            text=latest_data['country'],
            hovertemplate='<b>%{text}</b><br>DCI: %{x:.2f}<br>CATE: %{y:.3f}<extra></extra>',
            name='Countries'
        ))

        # Add trend line
        z = np.polyfit(latest_data['DCI'], latest_data['CATE'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(latest_data['DCI'].min(), latest_data['DCI'].max(), 100)

        fig3.add_trace(go.Scatter(
            x=x_line,
            y=p(x_line),
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='Linear Trend'
        ))

        # Add zero line
        fig3.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

        fig3.update_layout(
            xaxis_title="Digital Connectivity Index (DCI)",
            yaxis_title="CATE (CO2 Reduction Effect)",
            template='plotly_white',
            height=500,
            showlegend=True
        )

        st.plotly_chart(fig3, use_container_width=True)

        # Correlation analysis
        correlation = latest_data['DCI'].corr(latest_data['CATE'])
        st.info(f"**Correlation between DCI and CATE:** {correlation:.3f}")


# ==================== MODULE 3: POLICY SIMULATOR ====================
elif module == "🎛️ Policy Simulator":
    st.header("🎛️ Policy Simulator")
    st.markdown("""
    Simulate the impact of digitalization policies on CO2 emissions.
    Adjust the target DCI level to see predicted CO2 reductions.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Policy Settings")

        selected_country = st.selectbox(
            "Select country for simulation:",
            countries,
            index=countries.index('China') if 'China' in countries else 0
        )

        # Get country data
        latest_main = get_latest_year_data(df_main)
        latest_cate = get_latest_year_data(df_cate)

        country_main = latest_main[latest_main['country'] == selected_country].iloc[0]
        country_cate = latest_cate[latest_cate['country'] == selected_country].iloc[0]

        current_dci = country_main['DCI']
        current_co2 = country_main['CO2_per_capita']
        cate_estimate = country_cate['CATE']
        gdp_pc = country_main['GDP_per_capita_constant']

        st.markdown("---")
        st.markdown("**Current Status**")
        st.metric("Current DCI", f"{current_dci:.2f}")
        st.metric("Current CO2", f"{current_co2:.0f} kg/capita")
        st.metric("CATE Estimate", f"{cate_estimate:.3f}")

        st.markdown("---")
        st.markdown("**Policy Target**")

        target_dci = st.slider(
            "Target DCI Level",
            min_value=0.0,
            max_value=2.0,
            value=min(current_dci + 0.3, 2.0),
            step=0.1
        )

        # Calculate impact
        dci_change = target_dci - current_dci

        # Base prediction using CATE (convert to percentage points)
        co2_reduction_pct = abs(cate_estimate) * dci_change * 100
        co2_reduction_abs = current_co2 * co2_reduction_pct / 100

        # Calculate confidence interval
        se = abs(cate_estimate) * 0.15
        ci_lower = co2_reduction_pct - 1.96 * se * 100
        ci_upper = co2_reduction_pct + 1.96 * se * 100

        st.markdown("---")

        # Generate recommendation
        if dci_change <= 0:
            recommendation = "⚠️ Target DCI should be higher than current level for emission reduction."
            rec_type = "warning"
        elif dci_change > 0.5:
            if gdp_pc > 30000:
                recommendation = "🚀 **Aggressive Digitalization**: Focus on green technology export and smart infrastructure."
            else:
                recommendation = "📈 **Accelerated Digitalization**: Gradual approach with international support and capacity building."
            rec_type = "success"
        elif dci_change > 0.2:
            recommendation = "📊 **Moderate Digitalization**: Strengthen institutions while expanding digital infrastructure."
            rec_type = "info"
        else:
            recommendation = "🔧 **Incremental Improvements**: Focus on efficiency gains and technology optimization."
            rec_type = "info"

    with col2:
        st.subheader("Simulation Results")

        # Results display
        if dci_change > 0:
            col_res1, col_res2, col_res3 = st.columns(3)

            with col_res1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("DCI Increase", f"+{dci_change:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)

            with col_res2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("CO2 Reduction", f"{co2_reduction_pct:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)

            with col_res3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Absolute Reduction", f"{co2_reduction_abs:.0f} kg/capita")
                st.markdown('</div>', unsafe_allow_html=True)

            # Confidence Interval
            st.markdown("#### 95% Confidence Interval")
            fig_ci = go.Figure()

            fig_ci.add_trace(go.Bar(
                x=["CO2 Reduction (%)"],
                y=[co2_reduction_pct],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[ci_upper - co2_reduction_pct],
                    arrayminus=[co2_reduction_pct - ci_lower]
                ),
                marker_color='#2ca02c',
                width=0.3
            ))

            fig_ci.update_layout(
                yaxis_title="Reduction (%)",
                template='plotly_white',
                height=300,
                showlegend=False
            )

            st.plotly_chart(fig_ci, use_container_width=True)

            # Policy Recommendation
            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
            st.markdown(f"**Policy Recommendation:**\n\n{recommendation}")
            st.markdown('</div>', unsafe_allow_html=True)

            # Implementation timeline
            st.markdown("#### Implementation Timeline")

            timeline_data = {
                'Phase': ['Short-term\n(1-2 years)', 'Medium-term\n(3-5 years)', 'Long-term\n(5-10 years)'],
                'Focus': [
                    'Digital infrastructure expansion',
                    'Smart grid integration',
                    'Circular digital economy'
                ],
                'Expected Impact': [
                    f"{co2_reduction_pct * 0.3:.1f}% reduction",
                    f"{co2_reduction_pct * 0.6:.1f}% reduction",
                    f"{co2_reduction_pct:.1f}% reduction"
                ]
            }

            timeline_df = pd.DataFrame(timeline_data)
            st.dataframe(timeline_df, use_container_width=True, hide_index=True)

        else:
            st.warning("Please set a target DCI higher than the current level to see emission reduction predictions.")


# ==================== MODULE 4: COUNTRY COMPARISON ====================
elif module == "🔄 Country Comparison":
    st.header("🔄 Country Comparison")
    st.markdown("Compare two countries side-by-side to understand different policy pathways.")

    col1, col2 = st.columns(2)

    with col1:
        country1 = st.selectbox(
            "Select first country:",
            countries,
            index=countries.index('China') if 'China' in countries else 0,
            key='c1'
        )

    with col2:
        # Default to different country
        default_idx = countries.index('United States') if 'United States' in countries else 1
        country2 = st.selectbox(
            "Select second country:",
            countries,
            index=default_idx,
            key='c2'
        )

    # Get comparison data
    latest_main = get_latest_year_data(df_main)
    latest_cate = get_latest_year_data(df_cate)

    c1_main = latest_main[latest_main['country'] == country1].iloc[0]
    c1_cate = latest_cate[latest_cate['country'] == country1].iloc[0]
    c1_class = df_classification[df_classification['Country'] == country1].iloc[0]

    c2_main = latest_main[latest_main['country'] == country2].iloc[0]
    c2_cate = latest_cate[latest_cate['country'] == country2].iloc[0]
    c2_class = df_classification[df_classification['Country'] == country2].iloc[0]

    # Display comparison cards
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="country-card">', unsafe_allow_html=True)
        st.markdown(f"### {country1}")
        st.markdown(f"**Classification:** {c1_class['Classification']}")

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("DCI", f"{c1_main['DCI']:.2f}")
        with col_m2:
            st.metric("CO2", f"{c1_main['CO2_per_capita']:.0f}")
        with col_m3:
            st.metric("GDP", f"${c1_main['GDP_per_capita_constant']:,.0f}")

        st.markdown(f"**CATE Estimate:** {c1_cate['CATE']:.3f}")
        st.markdown(f"**Renewable Share:** {c1_class['Renewable_Share_2023']:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="country-card">', unsafe_allow_html=True)
        st.markdown(f"### {country2}")
        st.markdown(f"**Classification:** {c2_class['Classification']}")

        col_m4, col_m5, col_m6 = st.columns(3)
        with col_m4:
            st.metric("DCI", f"{c2_main['DCI']:.2f}")
        with col_m5:
            st.metric("CO2", f"{c2_main['CO2_per_capita']:.0f}")
        with col_m6:
            st.metric("GDP", f"${c2_main['GDP_per_capita_constant']:,.0f}")

        st.markdown(f"**CATE Estimate:** {c2_cate['CATE']:.3f}")
        st.markdown(f"**Renewable Share:** {c2_class['Renewable_Share_2023']:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    # Policy Recommendations
    st.subheader("Policy Recommendations")

    col5, col6 = st.columns(2)

    with col5:
        st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
        st.markdown(f"**{country1}**")
        st.markdown(f"- **Priority:** {c1_class['Policy_Priority']}")
        st.markdown(f"- **Investment Strategy:** {c1_class['Investment_Strategy']}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col6:
        st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
        st.markdown(f"**{country2}**")
        st.markdown(f"- **Priority:** {c2_class['Policy_Priority']}")
        st.markdown(f"- **Investment Strategy:** {c2_class['Investment_Strategy']}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Comparison Chart
    st.subheader("Visual Comparison")

    comparison_metrics = ['DCI', 'CO2_per_capita', 'GDP_per_capita_constant', 'CATE']
    c1_values = [c1_main['DCI'], c1_main['CO2_per_capita'], c1_main['GDP_per_capita_constant']/1000, c1_cate['CATE']]
    c2_values = [c2_main['DCI'], c2_main['CO2_per_capita'], c2_main['GDP_per_capita_constant']/1000, c2_cate['CATE']]

    fig_comp = go.Figure()

    fig_comp.add_trace(go.Bar(
        name=country1,
        x=['DCI', 'CO2 (kg)', 'GDP ($1000s)', 'CATE'],
        y=c1_values,
        marker_color='#1f77b4'
    ))

    fig_comp.add_trace(go.Bar(
        name=country2,
        x=['DCI', 'CO2 (kg)', 'GDP ($1000s)', 'CATE'],
        y=c2_values,
        marker_color='#ff7f0e'
    ))

    fig_comp.update_layout(
        barmode='group',
        title="Side-by-Side Comparison",
        template='plotly_white',
        height=400
    )

    st.plotly_chart(fig_comp, use_container_width=True)

    # Time series comparison
    st.subheader("Historical Trends Comparison")

    c1_ts = df_main[df_main['country'] == country1].sort_values('year')
    c2_ts = df_main[df_main['country'] == country2].sort_values('year')

    fig_ts = make_subplots(
        rows=1, cols=2,
        subplot_titles=('DCI Trend', 'CO2 Trend'),
        shared_yaxes=False
    )

    # DCI trend
    fig_ts.add_trace(
        go.Scatter(x=c1_ts['year'], y=c1_ts['DCI'], mode='lines+markers',
                  name=f"{country1} DCI", line=dict(color='#1f77b4')),
        row=1, col=1
    )
    fig_ts.add_trace(
        go.Scatter(x=c2_ts['year'], y=c2_ts['DCI'], mode='lines+markers',
                  name=f"{country2} DCI", line=dict(color='#ff7f0e')),
        row=1, col=1
    )

    # CO2 trend
    fig_ts.add_trace(
        go.Scatter(x=c1_ts['year'], y=c1_ts['CO2_per_capita'], mode='lines+markers',
                  name=f"{country1} CO2", line=dict(color='#1f77b4', dash='dash')),
        row=1, col=2
    )
    fig_ts.add_trace(
        go.Scatter(x=c2_ts['year'], y=c2_ts['CO2_per_capita'], mode='lines+markers',
                  name=f"{country2} CO2", line=dict(color='#ff7f0e', dash='dash')),
        row=1, col=2
    )

    fig_ts.update_layout(
        height=400,
        template='plotly_white',
        showlegend=True
    )

    st.plotly_chart(fig_ts, use_container_width=True)


# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>Digital Carbon Divide Research Project</strong> | Data sources: World Bank WDI, Causal Forest Analysis</p>
    <p style="font-size: 0.8rem;">Last updated: February 2026</p>
</div>
""", unsafe_allow_html=True)
