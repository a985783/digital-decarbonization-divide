#!/usr/bin/env python3
"""
Policy Simulator for Digital Decarbonization
============================================

This module provides tools to simulate the expected CO2 reduction from
Digital Capacity Index (DCI) investments based on country characteristics.

Based on: Causal Forest Analysis (N=40 countries, 2000-2023)
Key Finding: IV estimate of -1.91 tons CO2/capita per DCI unit

Author: Policy Research Team
Version: 1.0
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CountryProfile:
    """Country characteristics for policy simulation."""
    country_code: str
    country_name: str
    gdp_per_capita: float
    dci_current: float
    institution_score: float
    renewable_share: float
    co2_per_capita: float
    eds_pct: float = 0.0  # External Digital Specialization


@dataclass
class SimulationResult:
    """Results from policy simulation."""
    country_code: str
    current_classification: str
    dci_target: float
    expected_co2_reduction: float
    confidence_interval: Tuple[float, float]
    investment_priority: str
    timeline_years: int
    mechanism_breakdown: Dict[str, float]
    sdg_contribution: Dict[str, float]


class PolicySimulator:
    """
    Policy simulator for digital decarbonization investments.

    Uses the IV estimate (-1.91 tons CO2/capita per DCI unit) as baseline,
    with adjustments based on country characteristics.
    """

    # Key parameters from research findings
    BASELINE_EFFECT = -1.91  # IV estimate
    CI_LOWER = -2.37
    CI_UPPER = -1.46

    # Classification thresholds
    DCI_HIGH = 0.5
    DCI_LOW = -0.5
    INST_HIGH = 0.5
    INST_LOW = -0.5
    GDP_MIDDLE_LOW = 5000
    GDP_MIDDLE_HIGH = 25000

    # Adjustment factors based on heterogeneity analysis
    GDP_ADJUSTMENTS = {
        'low': 0.62,        # Low income: -1.19 / -1.91
        'lower_mid': 1.14,  # Lower-middle: -2.17 / -1.91
        'upper_mid': 1.20,  # Upper-middle: -2.29 / -1.91
        'high': 0.66        # High income: -1.26 / -1.91
    }

    RENEWABLE_ADJUSTMENT_FACTOR = 0.56  # Correlation with CATE
    INSTITUTION_ADJUSTMENT_FACTOR = 0.15
    EDS_DAMPENING_FACTOR = 0.20  # High EDS weakens effect

    def __init__(self, classification_csv: Optional[str] = None):
        """Initialize simulator with country classification data."""
        self.classification_data = self._load_classification(classification_csv)

    def _load_classification(self, csv_path: Optional[str]) -> pd.DataFrame:
        """Load country classification data."""
        if csv_path is None:
            csv_path = Path(__file__).parent / "country_classification.csv"
        return pd.read_csv(csv_path)

    def classify_country(self, profile: CountryProfile) -> str:
        """
        Classify country based on DCI and institutional quality.

        Returns:
            One of: 'leaders', 'catch_up', 'potential', 'struggling', 'exceptions'
        """
        dci_level = self._get_dci_level(profile.dci_current)
        inst_level = self._get_institution_level(profile.institution_score)

        # Check for exceptions (high DCI + high Inst but weak effects expected)
        if dci_level == 'high' and inst_level == 'high':
            if profile.eds_pct > 15 or profile.renewable_share > 60:
                return 'exceptions'
            return 'leaders'

        if dci_level == 'medium' and inst_level == 'medium':
            return 'catch_up'

        if dci_level == 'low' and inst_level == 'high':
            return 'potential'

        if dci_level == 'low' and inst_level == 'low':
            return 'struggling'

        # Mixed cases - default based on GDP
        if profile.gdp_per_capita < self.GDP_MIDDLE_LOW:
            return 'struggling'
        elif profile.gdp_per_capita < self.GDP_MIDDLE_HIGH:
            return 'catch_up'
        else:
            return 'leaders'

    def _get_dci_level(self, dci: float) -> str:
        """Categorize DCI level."""
        if dci > self.DCI_HIGH:
            return 'high'
        elif dci < self.DCI_LOW:
            return 'low'
        return 'medium'

    def _get_institution_level(self, inst: float) -> str:
        """Categorize institution level."""
        if inst > self.INST_HIGH:
            return 'high'
        elif inst < self.INST_LOW:
            return 'low'
        return 'medium'

    def _get_gdp_group(self, gdp: float) -> str:
        """Categorize GDP group."""
        if gdp < 5000:
            return 'low'
        elif gdp < 12000:
            return 'lower_mid'
        elif gdp < 30000:
            return 'upper_mid'
        return 'high'

    def calculate_effect_modifier(self, profile: CountryProfile) -> float:
        """
        Calculate effect modifier based on country characteristics.

        Adjusts the baseline -1.91 effect based on:
        - GDP level (sweet spot in middle income)
        - Renewable energy share (diminishing returns)
        - Institutional quality (enabling conditions)
        - EDS level (structural dampening)
        """
        # Base adjustment from GDP (sweet spot effect)
        gdp_group = self._get_gdp_group(profile.gdp_per_capita)
        adjustment = self.GDP_ADJUSTMENTS.get(gdp_group, 1.0)

        # Renewable energy adjustment (diminishing returns)
        # Countries with >60% renewables see weaker effects
        if profile.renewable_share > 60:
            renewable_adj = 0.7
        elif profile.renewable_share > 40:
            renewable_adj = 0.85
        elif profile.renewable_share > 20:
            renewable_adj = 1.0
        else:
            renewable_adj = 1.15  # Stronger effect in dirty grids

        # Institutional quality adjustment
        if profile.institution_score > 1.0:
            inst_adj = 1.1
        elif profile.institution_score > 0:
            inst_adj = 1.0
        elif profile.institution_score > -0.5:
            inst_adj = 0.85
        else:
            inst_adj = 0.70

        # EDS dampening
        if profile.eds_pct > 20:
            eds_adj = 0.8
        elif profile.eds_pct > 10:
            eds_adj = 0.9
        else:
            eds_adj = 1.0

        return adjustment * renewable_adj * inst_adj * eds_adj

    def simulate(
        self,
        profile: CountryProfile,
        dci_target: float,
        timeline_years: int = 5
    ) -> SimulationResult:
        """
        Simulate CO2 reduction from DCI investment.

        Args:
            profile: Country characteristics
            dci_target: Target DCI level (standardized PCA score)
            timeline_years: Implementation timeline

        Returns:
            SimulationResult with expected outcomes
        """
        classification = self.classify_country(profile)
        dci_increase = dci_target - profile.dci_current

        # Calculate effect with modifiers
        modifier = self.calculate_effect_modifier(profile)
        base_effect = self.BASELINE_EFFECT * modifier

        # Expected reduction per DCI unit
        expected_reduction = base_effect * dci_increase

        # Confidence interval
        ci_width = (self.CI_UPPER - self.CI_LOWER) / 2 * abs(dci_increase) * modifier
        ci_center = expected_reduction
        ci = (ci_center - ci_width, ci_center + ci_width)

        # Mechanism breakdown (based on mediation analysis)
        mechanism_breakdown = {
            'energy_efficiency': expected_reduction * 0.117,
            'structural_change': expected_reduction * (-0.095),
            'innovation': expected_reduction * (-0.083),
            'direct_effect': expected_reduction * (1 - 0.117 + 0.095 + 0.083)
        }

        # SDG contribution quantification
        sdg_contribution = {
            'sdg_7_clean_energy': abs(expected_reduction) * 0.117,  # Via efficiency
            'sdg_9_industry': abs(expected_reduction) * 0.30,       # Industrial digitalization
            'sdg_12_consumption': abs(expected_reduction) * 0.15,   # Resource efficiency
            'sdg_13_climate': abs(expected_reduction)               # Total CO2 reduction
        }

        # Investment priority based on classification
        priorities = {
            'leaders': 'Technology export + absolute decoupling',
            'catch_up': 'Sweet spot investment - maximize returns',
            'potential': 'Infrastructure first - unlock capacity',
            'struggling': 'International aid + capacity building',
            'exceptions': 'Absolute decoupling focus'
        }

        return SimulationResult(
            country_code=profile.country_code,
            current_classification=classification,
            dci_target=dci_target,
            expected_co2_reduction=expected_reduction,
            confidence_interval=ci,
            investment_priority=priorities.get(classification, 'Custom strategy'),
            timeline_years=timeline_years,
            mechanism_breakdown=mechanism_breakdown,
            sdg_contribution=sdg_contribution
        )

    def get_investment_pathway(
        self,
        profile: CountryProfile,
        target_dci: float
    ) -> Dict[str, List[str]]:
        """
        Generate optimal investment pathway.

        Returns phased recommendations based on country classification.
        """
        classification = self.classify_country(profile)

        pathways = {
            'short_term': {
                'all': [
                    'Deploy smart metering in high-consumption areas',
                    'Launch digital government services',
                    'Establish public WiFi in urban centers'
                ],
                'catch_up': [
                    'Conduct industrial efficiency audits',
                    'Deploy agricultural digital extension services'
                ],
                'leaders': [
                    'Implement grid flexibility services',
                    'Deploy carbon tracking platforms'
                ],
                'struggling': [
                    'Focus on basic connectivity',
                    'Climate adaptation digital tools'
                ]
            },
            'medium_term': {
                'all': [
                    'Integrate smart grid infrastructure',
                    'Develop sectoral digital platforms',
                    'Build data infrastructure'
                ],
                'catch_up': [
                    'Implement Manufacturing 4.0 pilots',
                    'Launch smart city demonstrations'
                ],
                'leaders': [
                    'Deploy AI-powered optimization',
                    'Build circular economy platforms'
                ]
            },
            'long_term': {
                'all': [
                    'Achieve full sectoral digitalization',
                    'Establish cross-border digital cooperation'
                ],
                'catch_up': [
                    'Develop export-oriented digital services',
                    'Establish green tech manufacturing'
                ],
                'leaders': [
                    'Deploy next-generation infrastructure',
                    'Lead global green tech markets'
                ]
            }
        }

        # Select relevant actions
        result = {
            'short_term': pathways['short_term']['all'] +
                         pathways['short_term'].get(classification, []),
            'medium_term': pathways['medium_term']['all'] +
                          pathways['medium_term'].get(classification, []),
            'long_term': pathways['long_term']['all'] +
                        pathways['long_term'].get(classification, [])
        }

        return result

    def generate_lookup_table(self) -> pd.DataFrame:
        """
        Generate CSV lookup table for all countries.

        Creates a reference table with expected effects for different scenarios.
        """
        scenarios = []

        # Mapping from CSV classification to internal classification
        classification_mapping = {
            'Leader': 'leaders',
            'Catch-up': 'catch_up',
            'Potential': 'potential',
            'Struggling': 'struggling',
            'Exception': 'exceptions'
        }

        for _, row in self.classification_data.iterrows():
            country_code = row['Country_Code']
            csv_classification = row['Classification']
            internal_classification = classification_mapping.get(csv_classification, 'catch_up')

            # Create profiles for different DCI targets
            for target_increase in [0.5, 1.0, 1.5, 2.0]:
                current_dci = 0.0  # Baseline
                target_dci = current_dci + target_increase

                profile = CountryProfile(
                    country_code=country_code,
                    country_name=row['Country'],
                    gdp_per_capita=row['GDP_Per_Capita_2023'],
                    dci_current=current_dci,
                    institution_score=0.5 if row['Institution_Level'] == 'High' else
                                     (0.0 if row['Institution_Level'] == 'Medium' else -0.5),
                    renewable_share=row['Renewable_Share_2023'],
                    co2_per_capita=row['CO2_Per_Capita_2023'],
                    eds_pct=row.get('EDS_Pct', 10.0)
                )

                result = self.simulate(profile, target_dci)

                # Override classification with the one from CSV
                scenarios.append({
                    'country_code': country_code,
                    'country_name': row['Country'],
                    'classification': internal_classification,
                    'dci_increase': target_increase,
                    'expected_reduction_tons': round(result.expected_co2_reduction, 2),
                    'ci_lower': round(result.confidence_interval[0], 2),
                    'ci_upper': round(result.confidence_interval[1], 2),
                    'timeline_years': result.timeline_years,
                    'priority': result.investment_priority
                })

        return pd.DataFrame(scenarios)


def main():
    """Example usage of the policy simulator."""
    simulator = PolicySimulator()

    # Example: Simulate for a middle-income country
    example_profile = CountryProfile(
        country_code='BRA',
        country_name='Brazil',
        gdp_per_capita=8917.67,
        dci_current=0.0,
        institution_score=0.0,
        renewable_share=48.26,
        co2_per_capita=2018.22,
        eds_pct=12.0
    )

    result = simulator.simulate(example_profile, dci_target=1.0)

    print("=" * 60)
    print("POLICY SIMULATION RESULTS")
    print("=" * 60)
    print(f"Country: {result.country_code}")
    print(f"Classification: {result.current_classification}")
    print(f"DCI Target: {result.dci_target}")
    print(f"Expected CO2 Reduction: {result.expected_co2_reduction:.2f} tons/capita")
    print(f"95% CI: [{result.confidence_interval[0]:.2f}, {result.confidence_interval[1]:.2f}]")
    print(f"Investment Priority: {result.investment_priority}")
    print(f"Timeline: {result.timeline_years} years")
    print("\nMechanism Breakdown:")
    for mechanism, value in result.mechanism_breakdown.items():
        print(f"  - {mechanism}: {value:.2f} tons/capita")
    print("\nSDG Contribution:")
    for sdg, value in result.sdg_contribution.items():
        print(f"  - {sdg}: {value:.2f} tons CO2 equivalent")

    # Generate investment pathway
    print("\n" + "=" * 60)
    print("INVESTMENT PATHWAY")
    print("=" * 60)
    pathway = simulator.get_investment_pathway(example_profile, target_dci=1.0)
    for phase, actions in pathway.items():
        print(f"\n{phase.upper()}:")
        for action in actions:
            print(f"  - {action}")

    # Generate lookup table
    print("\n" + "=" * 60)
    print("GENERATING LOOKUP TABLE...")
    print("=" * 60)
    lookup_table = simulator.generate_lookup_table()
    output_path = Path(__file__).parent / "policy_lookup_table.csv"
    lookup_table.to_csv(output_path, index=False)
    print(f"Lookup table saved to: {output_path}")
    print(f"Total scenarios: {len(lookup_table)}")


if __name__ == "__main__":
    main()
