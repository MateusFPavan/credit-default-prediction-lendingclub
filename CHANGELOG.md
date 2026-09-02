# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.1.0] - 2026-08-31

### Removed

- **BREAKING (request schema) — `application_type` dropped from the feature set and from
  the API contract.** The column was constant in the training split, so one-hot encoding
  with `drop_first` left it with **zero trained columns**: the model never had access to it
  and never could, while the API required it of every caller and validated it. It is now
  out of `FEATURE_SET` (79 -> 78), out of `CATEGORICAL_COLS` (5 -> 4) and out of
  `ScoreRequest`. Callers that still send it are unaffected — Pydantic ignores unknown
  fields.

### Note on published results

**No number changed, and that was measured rather than assumed.** An ablation retraining
without the feature reproduces $242,230,710.89 **to the cent**, keeps the same **90**
feature columns (it never contributed one), and the paired bootstrap of the profit
difference is `[$0, $0]` — an identity, not a sampling result. No retraining is required
and `models/xgb_final.joblib` is unchanged: its 90 trained column names never included an
`application_type_*` column, so the serving matrix is byte-identical.

The same ablation tested `emp_length_anos` as a second candidate, on the hypothesis that
its signal lived in the missingness flag rather than the years. **That hypothesis was
wrong**: removing it costs **$1,735,339** (-0.72%), with the bootstrap CI entirely
negative. It stays. Notably its AUC cost is only 0.0007 (0.684563 -> 0.683846) — a second,
independent instance of this project's central finding that AUC and the business metric
diverge. Selecting features by AUC would have discarded $1.7M.

## [3.0.0] - 2026-08-31

### Fixed

- **Inference encoding no longer depends on batch composition.** `prepare_X` used
  `pd.get_dummies(..., drop_first=True)`, whose output columns depend on which categories
  are present *in that call*. On a single-row request every categorical has exactly one
  category, `drop_first` removes it, and **zero** one-hot columns are produced; the
  subsequent `reindex(fill_value=0)` then filled all 16 trained columns with 0. Because
  the API is single-record, **every applicant was scored as the base category** —
  `home_ownership`, `purpose`, `verification_status` and `initial_list_status` were
  accepted, validated and ignored. Measured spread across categories was
  `0.0000000000` on all four; it is now 0.1380 for `purpose` (`small_business` moves from
  p=0.12 to p=0.26), 0.0164 for `home_ownership`, 0.0016 and 0.0001 for the others. The
  same record also scored differently depending on who else was in the batch
  (0.1341794431 alone vs 0.1289446801 in a batch; now identical). Fixed by passing
  `drop_first=False` at inference, which combined with the reindex reproduces the training
  encoding exactly, base category included.
- **`emp_length` is now converted to `emp_length_anos`.** The API accepted
  `emp_length: Optional[str]` ("10+ years") while the model uses the numeric
  `emp_length_anos`, and nothing converted one to the other — so every request was scored
  as "employment length unknown", on a feature ranked 26th of 88 by weight. 4.36% of
  training rows are genuinely missing; 100% of served rows were. The parser is copied
  verbatim from `notebooks/02_cleaning.ipynb`, not re-decided.
- **Defensive numeric coercion works again on pandas 3.** The guard tested
  `df[c].dtype == object`, but pandas ≥ 3.0 gives string columns a dedicated `StringDtype`,
  so the comparison was False and the coercion never ran. The condition was correct when
  written and became wrong on its own, without this file being touched.
- **Numerical-stability guards** on the training matrix and on the served probability, and
  a guard on `verify_pipeline`'s reindex, which used the same construction as the bug above
  and was safe only by a property of the data, not of the code.

### Changed

- **BREAKING — input contract narrowed.** `annual_inc` and `total_acc` now require `> 0`
  (previously `>= 0`). The training split has zero rows with either at 0, and
  `build_features` divides by both, so a 0 produced `Inf` on a feature the model has never
  seen — while still returning a well-formed probability. Same rationale the API already
  applied to `term=36`: refuse what the model cannot score rather than extrapolate.

### Added

- 68 tests across `tests/test_features.py`, `tests/test_scoring.py`,
  `tests/test_cleaning.py`, `tests/test_paridade_treino_inferencia.py` and
  `tests/test_estabilidade_numerica.py`. None of the pre-existing checks caught any of the
  above, because every failure mode produced a **well-formed, in-range answer**. The tests
  that do catch them are sensitivity tests (changing a feature must change the score) and
  a train-vs-inference parity assertion against the real 90-column artifact.

### Note on published results

**No published number changed.** `verify_pipeline` reproduces $242,230,710.89 to the cent
and `run_facts` regenerates every CSV in `reports/` byte-identically. The offline path
encodes whole splits, where every category is present and the encoding was always correct.
Only the live serving path (`src/api.py`, `src/monitor.py`) was affected.

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
