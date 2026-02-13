from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class GuardReport:
    failures: List[str]
    warnings: List[str]


def _read_iv_row(iv_file: Path) -> pd.Series:
    if not iv_file.exists():
        raise FileNotFoundError(f"Missing file: {iv_file}")

    df = pd.read_csv(iv_file)
    if df.empty:
        raise ValueError(f"Empty file: {iv_file}")

    if "Model" in df.columns and (df["Model"] == "IV (OrthoIV)").any():
        return df.loc[df["Model"] == "IV (OrthoIV)"].iloc[0]
    return df.iloc[0]


def _read_ar_row(iv_file: Path) -> pd.Series | None:
    if not iv_file.exists():
        return None
    df = pd.read_csv(iv_file)
    if df.empty or "Model" not in df.columns:
        return None
    ar_rows = df.loc[df["Model"] == "AR Robust CI"]
    if ar_rows.empty:
        return None
    return ar_rows.iloc[0]


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def run_guard(
    iv_file: Path,
    placebo_file: Path,
    small_sample_file: Path,
    final_cert_file: Path,
    integrity_report_file: Path,
    min_placebo_iterations: int = 100,
) -> GuardReport:
    failures: List[str] = []
    warnings: List[str] = []

    iv_row = _read_iv_row(iv_file)

    f_stat = float(iv_row.get("F_Statistic", float("nan")))
    r2 = float(iv_row.get("First_stage_R2", float("nan")))
    ate = float(iv_row.get("ATE", float("nan")))
    ci_lower = float(iv_row.get("CI_Lower", float("nan")))
    ci_upper = float(iv_row.get("CI_Upper", float("nan")))

    if not pd.notna(f_stat) or f_stat < 0:
        failures.append(f"Invalid first-stage F statistic: {f_stat}")
    if not pd.notna(r2) or r2 < 0 or r2 > 1:
        failures.append(f"Invalid first-stage R2: {r2}")
    if not pd.notna(ci_lower) or not pd.notna(ci_upper) or ci_lower > ci_upper:
        failures.append(f"Invalid confidence interval bounds: [{ci_lower}, {ci_upper}]")
    if pd.notna(ate) and pd.notna(ci_lower) and pd.notna(ci_upper):
        if ate < ci_lower or ate > ci_upper:
            failures.append(f"ATE {ate} is outside confidence interval [{ci_lower}, {ci_upper}]")

    ar_row = _read_ar_row(iv_file)
    if ar_row is not None:
        ar_lower = float(ar_row.get("CI_Lower", float("nan")))
        ar_upper = float(ar_row.get("CI_Upper", float("nan")))
        ar_pvalue = float(ar_row.get("AR_PValue", float("nan")))
        if not pd.notna(ar_lower) or not pd.notna(ar_upper) or ar_lower > ar_upper:
            failures.append(f"Invalid AR confidence interval bounds: [{ar_lower}, {ar_upper}]")
        if not pd.notna(ar_pvalue) or ar_pvalue < 0 or ar_pvalue > 1:
            failures.append(f"Invalid AR p-value: {ar_pvalue}")

    if not placebo_file.exists():
        failures.append(f"Missing placebo file: {placebo_file}")
    else:
        placebo_df = pd.read_csv(placebo_file)
        if len(placebo_df) < min_placebo_iterations:
            failures.append(
                f"Insufficient placebo iterations: {len(placebo_df)} < {min_placebo_iterations}"
            )

    if small_sample_file.exists():
        ss_df = pd.read_csv(small_sample_file)
        if not ss_df.empty:
            row = ss_df.iloc[0]
            if not _parse_bool(row.get("bootstrap_converged", False)):
                warnings.append("small-sample bootstrap convergence is false")
            if not _parse_bool(row.get("sample_size_stable", False)):
                warnings.append("small-sample sample-size stability is false")
    else:
        warnings.append(f"small-sample robustness file missing: {small_sample_file}")

    final_text = final_cert_file.read_text(encoding="utf-8") if final_cert_file.exists() else ""
    integrity_text = (
        integrity_report_file.read_text(encoding="utf-8")
        if integrity_report_file.exists()
        else ""
    )
    if "达到完美标准" in final_text and "CRITICAL ISSUES IDENTIFIED" in integrity_text:
        failures.append(
            "Governance contradiction: final certification says 'perfect' while integrity report flags critical issues"
        )

    return GuardReport(failures=failures, warnings=warnings)


def write_report(report: GuardReport, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    status = "PASS" if not report.failures else "FAIL"

    lines = [
        "# Academic Consistency Guard Report",
        "",
        f"Status: **{status}**",
        "",
        "## Failures",
    ]
    if report.failures:
        lines.extend([f"- {item}" for item in report.failures])
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Warnings")
    if report.warnings:
        lines.extend([f"- {item}" for item in report.warnings])
    else:
        lines.append("- None")

    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run academic consistency guard checks")
    parser.add_argument("--iv-file", default="results/iv_analysis_results.csv")
    parser.add_argument("--placebo-file", default="results/phase4_placebo_results.csv")
    parser.add_argument("--small-sample-file", default="results/small_sample_robustness.csv")
    parser.add_argument("--final-cert-file", default="final_certification.md")
    parser.add_argument("--integrity-report-file", default="academic_integrity_report.md")
    parser.add_argument("--output", default="results/academic_consistency_guard_report.md")
    parser.add_argument("--min-placebo-iterations", type=int, default=100)
    args = parser.parse_args()

    report = run_guard(
        iv_file=Path(args.iv_file),
        placebo_file=Path(args.placebo_file),
        small_sample_file=Path(args.small_sample_file),
        final_cert_file=Path(args.final_cert_file),
        integrity_report_file=Path(args.integrity_report_file),
        min_placebo_iterations=args.min_placebo_iterations,
    )
    write_report(report, Path(args.output))

    if report.failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
