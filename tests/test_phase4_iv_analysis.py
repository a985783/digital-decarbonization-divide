import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

def test_iv_logic_structure():
    try:
        from scripts import phase4_iv_analysis
    except ImportError:
        pytest.fail("Could not import scripts.phase4_iv_analysis")
    assert hasattr(phase4_iv_analysis, "run_iv_analysis")


def test_first_stage_f_statistic_non_negative():
    from scripts.phase4_iv_analysis import first_stage_f_statistic

    n = 12
    groups = np.array(["A"] * 6 + ["B"] * 6)
    z = np.linspace(-1.0, 1.0, n)
    w = np.column_stack([
        np.linspace(0.0, 1.0, n),
        np.linspace(1.0, 2.0, n),
    ])
    t = 0.7 * z + 0.2 * w[:, 0]

    f_stat, r2 = first_stage_f_statistic(t, z, w, groups)
    assert np.isfinite(f_stat)
    assert f_stat >= 0
    assert 0 <= r2 <= 1


def test_anderson_rubin_ci_returns_finite_bounds():
    from scripts.phase4_iv_analysis import anderson_rubin_ci

    n = 40
    z = np.linspace(-1.0, 1.0, n)
    w = np.column_stack([
        z ** 2,
        np.cos(np.linspace(0.0, np.pi, n)),
    ])
    t = 0.8 * z + 0.2 * w[:, 0] + 0.05 * w[:, 1]
    y = -1.5 * t + 0.3 * w[:, 1] + 0.1 * w[:, 0]

    lb, ub, stat0, pval = anderson_rubin_ci(y, t, z, w)
    assert np.isfinite(lb)
    assert np.isfinite(ub)
    assert lb <= ub
    assert np.isfinite(stat0)
    assert 0 <= pval <= 1

@patch("scripts.phase4_iv_analysis.load_config")
@patch("scripts.phase4_iv_analysis.pd.read_csv")
@patch("scripts.phase4_iv_analysis.LinearDML")
@patch("scripts.phase4_iv_analysis.OrthoIV")
@patch("scripts.phase4_iv_analysis.prepare_analysis_data")
def test_iv_run(mock_prep, mock_ortho, mock_linear, mock_read_csv, mock_load_config, tmp_path):
    N = 20
    df_mock = pd.DataFrame({
        "country": ["A"]*10 + ["B"]*10,
        "year": list(range(2000, 2010)) * 2,
        "DCI": np.linspace(0.1, 2.0, 20),
        "CO2_per_capita": np.linspace(1.0, 3.0, 20),
        "x1": np.linspace(-1.0, 1.0, 20),
        "w1": np.linspace(2.0, 4.0, 20),
    })
    mock_read_csv.return_value = df_mock
    
    mock_load_config.return_value = {
        "outcome": "CO2_per_capita",
        "treatment_main": "DCI",
        "controls_W": ["w1"],
        "moderators_X": ["x1"],
        "groups": "country"
    }
    
    mock_prep.return_value = (
        np.zeros(N), np.zeros(N), np.zeros((N, 2)), np.zeros((N, 2)), df_mock
    )
    
    mock_est_iv = MagicMock()
    mock_ortho.return_value = mock_est_iv
    mock_est_iv.effect.return_value = np.zeros(N) 
    mock_est_iv.ate.return_value = 0.5
    mock_est_iv.ate_interval.return_value = (0.1, 0.9)
    
    mock_est_linear = MagicMock()
    mock_linear.return_value = mock_est_linear
    mock_est_linear.ate.return_value = 0.4
    
    from scripts import phase4_iv_analysis
    phase4_iv_analysis.run_iv_analysis(
        iv_output_path=str(tmp_path / "iv_results.csv"),
        placebo_output_path=str(tmp_path / "placebo_iv_results.csv"),
    )
    
    assert mock_est_iv.fit.called
    assert mock_est_linear.fit.called
