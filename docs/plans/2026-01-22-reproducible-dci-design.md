# Reproducible DCI Pipeline Design

**Goal:** Make the paper fully reproducible and academically compliant by ensuring data, code, results, and narrative are strictly aligned, with DCI built from real WDI indicators and treatment set to DCI.

## Decision
- Use Route A: construct DCI from real WDI indicators and set treatment to DCI across all analysis scripts.
- Use `paper.tex` as the single source of truth for the manuscript.

## Single Source of Truth
Introduce `analysis_spec.yaml` to define:
- Outcome, treatment(s), moderators, controls
- Years, country list, and grouping key
- Cross-fitting (GroupKFold by country)
- Bootstrap settings (country cluster, B=1000)
- Imputation rules (MICE for W/X only, within-train folds)

All scripts read this config; no hardcoded `ICT_exports` or `DCI` in analysis logic.

## Data Pipeline
1. **Fetch**
   - New fetch script pulls WDI indicators:
     - IT.NET.USER.ZS (Internet users)
     - IT.NET.BBND.P2 (Fixed broadband)
     - IT.NET.SECR.P6 (Secure servers)
     - BX.GSR.CCIS.ZS (ICT service exports / EDS)
   - Save raw snapshot to `data/raw/wdi_<date>.csv` with metadata (codes, dates, hash).
2. **Clean + Impute**
   - No imputation for Y/T.
   - MICE for W/X only, fit inside training folds, apply to test folds.
3. **Construct DCI**
   - Standardize 3 indicators, PCA (PC1), re-standardize to mean 0, sd 1.
   - Persist PCA loadings and explained variance.
4. **Analysis**
   - Treatment = DCI.
   - GroupKFold by country.
   - Country-cluster bootstrap for GATE CIs.
   - Results output includes `country`, `year`, `DCI`, `EDS`, and group labels.

## Error Handling / Guardrails
- Hard fail if DCI components are missing or simulated.
- Hard fail if any Y/T imputation is detected.
- Assert GroupKFold uses `country`.
- Assert outputs contain expected columns for table reproducibility.

## Testing Strategy (TDD)
- Unit tests:
  - Config loader resolves fields consistently.
  - DCI construction uses real WDI columns only.
  - PCA output standardized and deterministic with seed.
- Integration tests:
  - Fold-level MICE never sees test data.
  - Y/T remain un-imputed.
  - Analysis outputs contain required columns and stable schema.

## Deliverables
- `analysis_spec.yaml`
- Updated fetch/clean/analysis scripts
- Updated `DATA_MANIFEST.md`, `README.md`, `paper.tex`
- Regenerated results and figures

## Out of Scope
- Model redesign or new estimators beyond required alignment and compliance.
