import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock

def test_mechanism_logic_structure():
    try:
        from scripts import phase5_mechanism
    except ImportError:
        pytest.fail("Could not import scripts.phase5_mechanism")
    assert hasattr(phase5_mechanism, "analyze_renewable_mechanism")

@patch("scripts.phase5_mechanism.pd.read_csv")
@patch("scripts.phase5_mechanism.plt.savefig")
def test_mechanism_run(mock_savefig, mock_read_csv):
    df_mock = pd.DataFrame({
        "CATE": np.linspace(-2, 1, 20),
        "Renewable_energy_consumption_pct": np.linspace(0, 100, 20),
        "GDP_per_capita_constant": np.random.rand(20) * 10000,
        "Control_of_Corruption": np.random.rand(20),
        "Significant": [True, False] * 10
    })
    mock_read_csv.return_value = df_mock
    
    from scripts import phase5_mechanism
    
    phase5_mechanism.analyze_renewable_mechanism()
    
    assert mock_savefig.called
    args, _ = mock_savefig.call_args
    assert "mechanism_renewable_curve.png" in args[0]
