# PSI reports (Project 4, Phase 1)

All numbers in this directory come from `src/run_psi.py` (`python -m src.run_psi`) run
against `models/xgb_final.joblib` and `data/processed/{train,validation,test}.parquet`.
Nothing here was typed by hand.

## Files

- `psi_splits_raw.csv` - PSI of every FEATURE_SET column (79, per `docs/FACTS.md` section 7
  and `src/data.py`; the task brief referenced 77, this sheet reports the actual count) plus
  `model_score`, baseline = full train (issue_d 2007-06 to 2013-12), compared against
  validation (2014) and test (2015).
- `psi_splits_clean.csv` - identical comparison, but the baseline is train restricted to
  `era_pre_2012 == 0` (2012-2013 vintages only).
- `psi_quarterly.csv` - PSI of `model_score` and the 6 top-SHAP features (`docs/FACTS.md`
  section 5: `fico_range_low`, `installment_to_income`, `annual_inc`, `dti`, `revol_util`,
  `inq_last_6mths`), per `issue_d` quarter across train+validation+test, always against the
  same train (raw) baseline bins - never re-fit per quarter, so the curve is comparable
  across time.

## Read the raw view's high PSI as a known artifact, not real drift

`psi_splits_raw.csv` shows many features at "unstable" band. This is expected and was
predicted before this analysis ran, in `docs/scope.md` section 10 ("Sentinel-driven
shift"): 19 of the 20 features with the largest train/test standardized mean difference
are exactly the 2012 bureau-rollout block, because those columns carry sentinel values
(-1, 999) in ~29.9% of train and 0% of validation/test - a mechanical consequence of
keeping pre-2012 vintages in training (decision D3), not a change in the world. The
`psi_splits_clean.csv` view, which drops the pre-2012 rows from the baseline, is the
sensitivity check scope.md section 10 calls the "first experiment, not an extension" -
comparing the two views isolates how much of the raw PSI is sentinel artifact versus
residual real shift.

## Two more per-column caveats worth reading before scanning the CSVs

- `issue_d` itself shows an extreme PSI (~13, far past the 0.25 "unstable" ceiling) in
  every view. This is definitional, not drift: train/validation/test are cut *by*
  `issue_d`, so by construction train's issue_d values and test's never overlap. Its PSI
  says "the splits are time-ordered," which we already know - it is not an actionable
  monitoring signal and is kept in the CSV only for completeness.
- `initial_list_status` stays "unstable" even in the clean view (0.48). Per
  `docs/scope.md` section 10 ("Genuine policy shift"), this is real: the platform moved
  from mostly fractional ("f") to mostly whole ("w") loan listings between train and
  test (82% -> 41% "f"). Unlike the sentinel columns, this one isn't an artifact to
  discount - it's a documented, expected policy change.

## Bands

< 0.10 stable, 0.10-0.25 attention, >= 0.25 unstable.

## Summary (feature-level bands, one row per FEATURE_SET column, excluding model_score)

| Variant | Comparison | Stable | Attention | Unstable |
|---|---|---|---|---|
| Raw | validation (2014) | 40 | 3 | 36 |
| Raw | test (2015) | 40 | 4 | 35 |
| Clean (era_pre_2012==0 baseline) | validation (2014) | 75 | 3 | 1 |
| Clean (era_pre_2012==0 baseline) | test (2015) | 75 | 2 | 2 |

## Score PSI (model_score, train baseline vs. test)

| Variant | PSI | Band |
|---|---|---|
| Raw | 0.0015 | stable |
| Clean | 0.0014 | stable |

The PSI engine and its rules (bins fit once on baseline and reused, decile bins for
numerics with sentinel point-masses carved into their own bin, new categories kept as
their own bin, epsilon=1e-6 for empty bins) are documented in `src/psi.py`.
