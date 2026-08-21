# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2026-08-21

### Added
- **Reject inference (v3): a rigorous, negative-result investigation** (`notebooks/16`-`20`,
  `docs/reject_inference_roadmap.md`). Tests whether reject inference can correct the model's
  documented selection-bias limitation. Thin model (Logistic Regression, 3 shared features)
  scores 0.5620 ± 0.0027 AUC out-of-sample (4-fold CV); the profit metric was independently
  re-implemented and validated byte-exact against the published production result
  ($242,230,710.89 @ threshold 0.31). Two independent lines of evidence — a direct
  impossibility argument (no reject-outcome labels exist anywhere in the dataset, no true
  population rate to check inferred rates against, and a measured, monotonic training-rate
  inflation across the multiplier sweep) and a Bayesian bias-aware evaluation that reproduces
  its own prior (slope ≈ 0.99) rather than the data — both conclude RI is not validatable on
  this dataset. Includes an independent cross-environment validation of the ingestion
  pipeline on Databricks Free Edition. The production model (`src/api.py`) is unchanged;
  this is a methodology chapter, not a product change.

## [2.0.2] - 2026-08-18

### Fixed
- README's stale "not yet served behind an API" framing corrected (see README.md and
  docs/MODEL_CARD.md section 10) -- technical_report.md had the same stale claim, fixed
  here. technical_report.md's Limitations section (§11) now describes the model as
  served and monitored, with the still-valid caveat that it must not drive a real
  credit decision; its Recommendations section (§13) drops the already-shipped
  deployment item and adds automated retraining execution as a next step.
- docs/SETUP.md corrected: it said requirements.txt has 124 packages, the file actually
  pins 127 (fastapi, uvicorn, pytest, added in v2.0.0 for the API and its test suite).
  requirements.txt itself reordered so those three packages sit in alphabetical
  position instead of appended at the end.

## [2.0.1] - 2026-08-06

### Fixed
- SETUP guide: removed the stale "No Makefile, Docker, or CI ... [TODO: add if a
  reviewer requires one]" note in section 8 and reconciled the wording with reality --
  Docker (section 9) and the GitHub Actions CI now exist. Version note bumped to 2.0.1.

## [2.0.0] - 2026-08-06

### Added
- Production serving: FastAPI inference service (src/api.py) with POST /score
  (returns probability_default + approve/reject at the 0.31 threshold) and GET /health;
  input schema with per-field required/default decisions, 422 on invalid input, term
  restricted to 36.
- Production cleaning port (src/cleaning.py): notebook 03's missing-data treatment as
  reusable code for single records, verified bit-for-bit against loans_clean.parquet.
- Drift monitor (src/monitor.py): reuses the Phase-1 PSI engine against the clean
  training baseline, with a measured sample floor and a real-vs-artifact verdict
  (initial_list_status policy drift vs issue_d definitional artifact).
- Containerization: Dockerfile, .dockerignore, and a lean requirements-api.txt.
- CI: GitHub Actions workflow (.github/workflows/docker.yml) building the image and
  smoke-testing /health, a real /score, and the 422 path on every push.
- Model Card production section (section 10) and a defined retraining trigger.
- models/xgb_final.joblib is now versioned in the repository (4.1 MB) so the container
  and a clean checkout can serve without regenerating it.

### Changed
- Model Card version bumped to 2.0.0; deployment and drift monitoring moved from
  "next steps" into a real production section.

### Removed
- Model Card claims that the model had "no API, no monitoring, no drift detection" and
  that deployment was an unfinished next step -- now false.
