from pathlib import Path
import re

import pandas as pd


BANNED_RELATIVE = [
    "paper.md",
    "data/clean_data_v3_imputed.csv",
    "scripts/solve_wdi_v3_zip.py",
]

REQUIRED_COLUMNS = {"country", "year", "DCI", "EDS", "CATE"}


def _select_root():
    repl = Path("dist") / "replication"
    return repl if repl.exists() else Path(".")


def _assert_exists(path, message):
    if not path.exists():
        raise SystemExit(f"[FAIL] {message}: {path}")


def _check_banned(root):
    for rel in BANNED_RELATIVE:
        if (root / rel).exists():
            raise SystemExit(f"[FAIL] Banned legacy file found: {rel}")


def _check_results(root):
    results_path = root / "results" / "causal_forest_cate.csv"
    _assert_exists(results_path, "Missing causal_forest_cate.csv")
    df = pd.read_csv(results_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SystemExit(f"[FAIL] causal_forest_cate.csv missing columns: {missing}")
    if len(df) != 840:
        raise SystemExit(f"[FAIL] Unexpected N in causal_forest_cate.csv: {len(df)}")


def _check_paper(root):
    paper_path = root / "paper.tex"
    _assert_exists(paper_path, "Missing paper.tex")
    tex = paper_path.read_text(encoding="utf-8")
    if "N=840" not in tex and "N = 840" not in tex:
        raise SystemExit("[FAIL] paper.tex does not mention N=840")

    # Allow literature/references mentions but block results/interpretation leftovers.
    scrubbed = re.sub(r"rebound effect hypothesis", "", tex, flags=re.IGNORECASE)
    scrubbed = re.sub(r"rebound effects and ICT", "", scrubbed, flags=re.IGNORECASE)
    if re.search(r"rebound", scrubbed, flags=re.IGNORECASE):
        raise SystemExit("[FAIL] paper.tex contains rebound wording beyond literature context")


def main():
    root = _select_root()
    _assert_exists(root / "analysis_spec.yaml", "Missing analysis_spec.yaml")
    _check_banned(root)
    _check_results(root)
    _check_paper(root)
    print("[OK] Preflight release check passed.")


if __name__ == "__main__":
    main()
