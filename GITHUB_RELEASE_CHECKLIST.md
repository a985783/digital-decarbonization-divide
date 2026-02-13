# GitHub Release Checklist (Replication Package)

Language / 语言: [English](GITHUB_RELEASE_CHECKLIST.md) | [中文](GITHUB_RELEASE_CHECKLIST.zh-CN.md)

## Repository Metadata

- [x] `README.md` includes reproducibility quick start.
- [x] `LICENSE` present (MIT).
- [x] `CITATION.cff` present.
- [x] `CONTRIBUTING.md` present.

## Reproducibility Entry Points

- [x] `reproduce.sh` one-command workflow.
- [x] `Makefile` stage-by-stage workflow.
- [x] `REPRODUCIBILITY.md` detailed runbook.
- [x] `ARTIFACTS.md` output mapping.
- [x] `environment.yml` included.

## Verification Gates

- [x] `scripts/preflight_release_check.py` passes.
- [x] `scripts/academic_consistency_guard.py` passes.
- [x] Focused regression tests pass.
- [x] CI workflow at `.github/workflows/reproducibility.yml`.

## Portability

- [x] Dashboard data loading uses repository-relative paths.
- [x] Hardcoded local absolute paths removed from app entry files.

## Before Tagging a Release

- [ ] Confirm `CITATION.cff` repository URL and authors are final.
- [ ] Confirm release artifacts to include/exclude from repository history.
- [ ] Add release notes summarizing methods and replication scope.
