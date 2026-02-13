import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

def test_validity_logic_structure():
    try:
        from scripts import phase6_external_validity
    except ImportError:
        pytest.fail("Could not import scripts.phase6_external_validity")
    assert hasattr(phase6_external_validity, "run_external_validity")

@patch("scripts.phase6_external_validity.pd.read_csv")
@patch("scripts.phase6_external_validity.plt.savefig")
def test_validity_run(mock_savefig, mock_read_csv):
    df_mock = pd.DataFrame({
        "country": ["A", "B", "C", "D"],
        "year": [2019]*4,
        "GDP_per_capita_current": [1000, 2000, 100, 100],
        "Population_total": [1000, 1000, 1000, 1000],
        "CO2_per_capita": [5, 5, 1, 1]
    })
    
    sample_df = pd.DataFrame({"country": ["A", "B"]})
    
    def side_effect(filepath):
        if "wdi_expanded_raw" in filepath:
            return df_mock
        elif "clean_data" in filepath:
            return sample_df
        return pd.DataFrame()
        
    mock_read_csv.side_effect = side_effect
    
    from scripts import phase6_external_validity
    phase6_external_validity.run_external_validity()
    
    assert mock_savefig.called
