import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


def test_placebo_logic_structure():
    try:
        from scripts import phase4_placebo
    except ImportError:
        pytest.fail("Could not import scripts.phase4_placebo")

    assert hasattr(phase4_placebo, "run_placebo_analysis"), "Module missing run_placebo_analysis function"

@patch("scripts.phase4_placebo.prepare_analysis_data")
@patch("scripts.phase4_placebo.CausalForestDML")
@patch("scripts.phase4_placebo.load_config")
@patch("scripts.phase4_placebo.pd.read_csv")
def test_placebo_run_mocked(mock_read_csv, mock_load_config, mock_cf, mock_prepare_data, tmp_path):
    mock_df = pd.DataFrame({"col": [1, 2, 3], "CO2_per_capita": [1.0, 2.0, 3.0]})
    mock_read_csv.return_value = mock_df
    
    mock_load_config.return_value = {
        "outcome": "CO2_per_capita",
        "treatment_main": "DCI",
        "controls_W": ["w1"],
        "moderators_X": ["x1"],
        "groups": "country"
    }

    N = 10
    mock_prepare_data.return_value = (
        np.linspace(0.2, 1.2, N),
        np.linspace(0.1, 1.1, N),
        np.column_stack([np.linspace(-1, 1, N), np.linspace(1, 2, N)]),
        np.column_stack([np.linspace(0, 1, N), np.linspace(2, 3, N)]),
        pd.DataFrame({"country": [1] * N}),
    )
    
    mock_est = MagicMock()
    mock_cf.return_value = mock_est
    mock_est.effect.return_value = np.linspace(-2.0, -1.0, N)
    
    from scripts import phase4_placebo
    
    results = phase4_placebo.run_placebo_analysis(
        n_iterations=2,
        n_estimators=10,
        output_csv_path=str(tmp_path / "placebo_results.csv"),
        output_figure_path=str(tmp_path / "placebo_distribution.png"),
        random_state=123,
    )
    
    assert mock_cf.call_count >= 2 + 1 
    assert mock_est.fit.call_count >= 2 + 1
    
    assert "placebo_ates" in results
    assert "true_ate" in results
    assert len(results["placebo_ates"]) == 2
