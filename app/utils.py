"""
Utility functions for the Digital Carbon Divide Streamlit Dashboard.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from paths import data_path, result_path, toolkit_path, assert_exists


# Data loading functions
@st.cache_data
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all required datasets."""
    df_main = pd.read_csv(assert_exists(data_path('clean_data_v5_enhanced.csv')))
    df_classification = pd.read_csv(assert_exists(toolkit_path('country_classification.csv')))
    df_cate = pd.read_csv(assert_exists(result_path('causal_forest_cate.csv')))
    return df_main, df_classification, df_cate


def get_country_list(df: pd.DataFrame) -> List[str]:
    """Get sorted list of unique countries."""
    return sorted(df['country'].unique().tolist())


def get_country_code_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """Create mapping from country name to country code."""
    return dict(zip(df['country'], df['country']))


# Data processing functions
def get_country_time_series(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """Get time series data for a specific country."""
    return df[df['country'] == country].sort_values('year')


def get_latest_year_data(df: pd.DataFrame) -> pd.DataFrame:
    """Get data for the most recent year available for each country."""
    return df.loc[df.groupby('country')['year'].idxmax()]


def calculate_policy_impact(
    current_dci: float,
    target_dci: float,
    cate_estimate: float,
    gdp_per_capita: float
) -> Dict:
    """
    Calculate predicted CO2 reduction from DCI policy change.

    Args:
        current_dci: Current DCI level
        target_dci: Target DCI level
        cate_estimate: Conditional Average Treatment Effect estimate
        gdp_per_capita: GDP per capita for context

    Returns:
        Dictionary with impact metrics
    """
    dci_change = target_dci - current_dci

    # Base prediction using CATE
    co2_reduction = abs(cate_estimate) * dci_change * 100  # Convert to percentage points

    # Calculate confidence interval (assuming 15% standard error)
    se = abs(cate_estimate) * 0.15
    ci_lower = co2_reduction - 1.96 * se * 100
    ci_upper = co2_reduction + 1.96 * se * 100

    # Policy recommendation based on context
    if dci_change > 0.5:
        if gdp_per_capita > 30000:
            recommendation = "Aggressive digitalization with focus on green technology export"
        else:
            recommendation = "Gradual digitalization with international support"
    elif dci_change > 0.2:
        recommendation = "Moderate digitalization with institutional strengthening"
    else:
        recommendation = "Maintain current trajectory with efficiency improvements"

    return {
        'dci_change': dci_change,
        'co2_reduction': co2_reduction,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'recommendation': recommendation
    }


def classify_policy_outcome(cate: float, classification: str) -> str:
    """Classify policy outcome based on CATE and country classification."""
    if cate < -1.5:
        return "High Impact"
    elif cate < -0.8:
        return "Moderate Impact"
    elif cate < -0.3:
        return "Low Impact"
    else:
        return "Minimal Impact"


# Visualization functions
def create_time_series_plot(df: pd.DataFrame, country: str) -> go.Figure:
    """Create time series plot for DCI, CO2, and GDP."""
    country_data = get_country_time_series(df, country)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=('Digital Connectivity Index (DCI)', 'CO2 per Capita', 'GDP per Capita'),
        vertical_spacing=0.1
    )

    # DCI
    fig.add_trace(
        go.Scatter(
            x=country_data['year'],
            y=country_data['DCI'],
            mode='lines+markers',
            name='DCI',
            line=dict(color='#1f77b4', width=2)
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
            line=dict(color='#d62728', width=2)
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
            line=dict(color='#2ca02c', width=2)
        ),
        row=3, col=1
    )

    fig.update_layout(
        height=600,
        showlegend=False,
        title_text=f"{country} - Time Series Analysis",
        template='plotly_white'
    )

    return fig


def create_scatter_matrix(df: pd.DataFrame, country: Optional[str] = None) -> go.Figure:
    """Create scatter plot matrix for DCI, CO2, and GDP."""
    if country:
        plot_data = df[df['country'] == country]
        title = f"{country} - Variable Relationships"
    else:
        plot_data = get_latest_year_data(df)
        title = "Cross-Country Variable Relationships"

    fig = px.scatter_matrix(
        plot_data,
        dimensions=['DCI', 'CO2_per_capita', 'GDP_per_capita_constant'],
        color='country' if not country else None,
        title=title,
        labels={
            'DCI': 'Digital Connectivity Index',
            'CO2_per_capita': 'CO2 per Capita',
            'GDP_per_capita_constant': 'GDP per Capita'
        }
    )

    fig.update_layout(
        height=600,
        template='plotly_white'
    )

    return fig


def create_cate_distribution(df_cate: pd.DataFrame) -> go.Figure:
    """Create interactive CATE distribution plot."""
    latest_data = get_latest_year_data(df_cate)

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=latest_data['CATE'],
        nbinsx=20,
        name='CATE Distribution',
        opacity=0.7,
        marker_color='#1f77b4'
    ))

    # Add mean line
    mean_cate = latest_data['CATE'].mean()
    fig.add_vline(
        x=mean_cate,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_cate:.2f}"
    )

    # Add zero line
    fig.add_vline(x=0, line_dash="dot", line_color="gray")

    fig.update_layout(
        title="Distribution of Conditional Average Treatment Effects (CATE)",
        xaxis_title="CATE (CO2 Reduction Effect)",
        yaxis_title="Count",
        template='plotly_white',
        height=400
    )

    return fig


def create_gate_heatmap(df_cate: pd.DataFrame, df_classification: pd.DataFrame) -> go.Figure:
    """Create GATE heatmap by GDP and Institution groups."""
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

    fig = px.imshow(
        pivot,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdYlGn',
        title="GATE Heatmap: Average CATE by GDP and Institution Level"
    )

    fig.update_layout(
        height=400,
        xaxis_title="GDP per Capita Group",
        yaxis_title="Institution Level"
    )

    return fig


def create_linear_vs_forest_comparison(df_cate: pd.DataFrame) -> go.Figure:
    """Create comparison plot between linear and forest models."""
    latest_data = get_latest_year_data(df_cate)

    fig = go.Figure()

    # Scatter plot of CATE vs DCI
    fig.add_trace(go.Scatter(
        x=latest_data['DCI'],
        y=latest_data['CATE'],
        mode='markers',
        marker=dict(
            size=10,
            color=latest_data['GDP_per_capita_constant'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="GDP per Capita")
        ),
        text=latest_data['country'],
        name='CATE Estimates'
    ))

    # Add trend line
    z = np.polyfit(latest_data['DCI'], latest_data['CATE'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(latest_data['DCI'].min(), latest_data['DCI'].max(), 100)

    fig.add_trace(go.Scatter(
        x=x_line,
        y=p(x_line),
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Linear Trend'
    ))

    fig.update_layout(
        title="CATE vs DCI Level (colored by GDP)",
        xaxis_title="Digital Connectivity Index (DCI)",
        yaxis_title="CATE (CO2 Reduction Effect)",
        template='plotly_white',
        height=500
    )

    return fig


def get_country_comparison_data(
    df_main: pd.DataFrame,
    df_classification: pd.DataFrame,
    df_cate: pd.DataFrame,
    country1: str,
    country2: str
) -> Dict:
    """Get comparison data for two countries."""
    # Get latest data
    latest_main = get_latest_year_data(df_main)
    latest_cate = get_latest_year_data(df_cate)

    # Country 1 data
    c1_main = latest_main[latest_main['country'] == country1].iloc[0]
    c1_cate = latest_cate[latest_cate['country'] == country1].iloc[0]
    c1_class = df_classification[df_classification['Country'] == country1].iloc[0]

    # Country 2 data
    c2_main = latest_main[latest_main['country'] == country2].iloc[0]
    c2_cate = latest_cate[latest_cate['country'] == country2].iloc[0]
    c2_class = df_classification[df_classification['Country'] == country2].iloc[0]

    return {
        'country1': {
            'name': country1,
            'dci': c1_main['DCI'],
            'co2': c1_main['CO2_per_capita'],
            'gdp': c1_main['GDP_per_capita_constant'],
            'cate': c1_cate['CATE'],
            'classification': c1_class['Classification'],
            'policy_priority': c1_class['Policy_Priority'],
            'investment_strategy': c1_class['Investment_Strategy']
        },
        'country2': {
            'name': country2,
            'dci': c2_main['DCI'],
            'co2': c2_main['CO2_per_capita'],
            'gdp': c2_main['GDP_per_capita_constant'],
            'cate': c2_cate['CATE'],
            'classification': c2_class['Classification'],
            'policy_priority': c2_class['Policy_Priority'],
            'investment_strategy': c2_class['Investment_Strategy']
        }
    }
