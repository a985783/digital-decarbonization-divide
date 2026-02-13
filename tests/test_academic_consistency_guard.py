import pandas as pd

from scripts.academic_consistency_guard import run_guard


def test_guard_pass_with_valid_inputs(tmp_path):
    iv_file = tmp_path / "iv.csv"
    placebo_file = tmp_path / "placebo.csv"
    small_sample_file = tmp_path / "small_sample.csv"
    final_cert_file = tmp_path / "final.md"
    integrity_file = tmp_path / "integrity.md"

    pd.DataFrame(
        [
            {
                "Model": "IV (OrthoIV)",
                "ATE": -1.9,
                "CI_Lower": -2.3,
                "CI_Upper": -1.4,
                "F_Statistic": 12.0,
                "First_stage_R2": 0.4,
            },
            {
                "Model": "AR Robust CI",
                "ATE": float("nan"),
                "CI_Lower": -2.2,
                "CI_Upper": -1.2,
                "F_Statistic": float("nan"),
                "First_stage_R2": float("nan"),
                "AR_Statistic": 15.0,
                "AR_PValue": 0.01,
            },
        ]
    ).to_csv(iv_file, index=False)

    pd.DataFrame({"iteration": list(range(1, 101)), "placebo_ate": [0.0] * 100}).to_csv(
        placebo_file, index=False
    )
    pd.DataFrame(
        [{"bootstrap_converged": True, "sample_size_stable": True}]
    ).to_csv(small_sample_file, index=False)

    final_cert_file.write_text("archive", encoding="utf-8")
    integrity_file.write_text("ok", encoding="utf-8")

    report = run_guard(
        iv_file=iv_file,
        placebo_file=placebo_file,
        small_sample_file=small_sample_file,
        final_cert_file=final_cert_file,
        integrity_report_file=integrity_file,
        min_placebo_iterations=100,
    )

    assert report.failures == []


def test_guard_detects_invalid_iv_and_short_placebo(tmp_path):
    iv_file = tmp_path / "iv.csv"
    placebo_file = tmp_path / "placebo.csv"
    small_sample_file = tmp_path / "small_sample.csv"
    final_cert_file = tmp_path / "final.md"
    integrity_file = tmp_path / "integrity.md"

    pd.DataFrame(
        [
            {
                "Model": "IV (OrthoIV)",
                "ATE": 0.5,
                "CI_Lower": 0.1,
                "CI_Upper": 0.9,
                "F_Statistic": -0.7,
                "First_stage_R2": -0.2,
            },
            {
                "Model": "AR Robust CI",
                "ATE": float("nan"),
                "CI_Lower": float("nan"),
                "CI_Upper": float("nan"),
                "F_Statistic": float("nan"),
                "First_stage_R2": float("nan"),
                "AR_Statistic": float("nan"),
                "AR_PValue": float("nan"),
            },
        ]
    ).to_csv(iv_file, index=False)

    pd.DataFrame({"iteration": [1, 2], "placebo_ate": [0.1, -0.1]}).to_csv(
        placebo_file, index=False
    )
    pd.DataFrame(
        [{"bootstrap_converged": False, "sample_size_stable": "False"}]
    ).to_csv(small_sample_file, index=False)

    final_cert_file.write_text("达到完美标准", encoding="utf-8")
    integrity_file.write_text("CRITICAL ISSUES IDENTIFIED", encoding="utf-8")

    report = run_guard(
        iv_file=iv_file,
        placebo_file=placebo_file,
        small_sample_file=small_sample_file,
        final_cert_file=final_cert_file,
        integrity_report_file=integrity_file,
        min_placebo_iterations=100,
    )

    joined = "\n".join(report.failures)
    assert "Invalid first-stage F statistic" in joined
    assert "Invalid first-stage R2" in joined
    assert "Invalid AR confidence interval bounds" in joined
    assert "Invalid AR p-value" in joined
    assert "Insufficient placebo iterations" in joined
    assert "Governance contradiction" in joined
    assert len(report.warnings) == 2
