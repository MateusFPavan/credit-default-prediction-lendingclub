# Reject inference: can it be validated on Lending Club?

The production model is trained on approved loans only, so it estimates
P(default | approved) and has never seen a rejected application. Reject inference (RI) is
the standard technique for correcting that selection bias. This chapter asks whether RI
can be validated on this dataset. **It cannot**, and two independent lines of evidence
show why. The production model, the API, the 0.31 threshold and the feature set are
unchanged by this work.

Implementation: `notebooks/16` through `20` (and `21` for the vintage cross-check).
Outputs: `reports/reject/`.

## Why ask the question this way

The rejected population is large: **27,648,741** applications against **673,314**
approved loans in the analytical population, about **41x**. Practice and research
disagree on what to do about it. Lenders and regulators expect reject inference to be
done and documented; the academic literature (Hand & Henley, 1993, among others) argues
that when rejects are missing not at random, no inference can be tested without
accepting a sample of rejects and observing them. The design followed from that: run RI
as an experiment, and measure whether its result can be validated before claiming any
improvement.

## Phase 1: ingestion and profiling (`notebooks/16`)

- **Source**: `rejected_2007_to_2018Q4.csv.gz` (255,470,782 bytes), Kaggle
  `wordsforthewise/lending-club`, CC0-1.0, the same package as the approved file. Nine
  columns; no FICO and no grade.
- **Parsing**: one malformed line out of 27,648,741, recovered with explicit CSV options
  (`multiLine`, `quote`, `escape`). `PERMISSIVE` mode with a `_corrupt_record` column
  audits the result: 0 corrupt rows after the fix.
- **Architecture**: PySpark ingests the single, non-splittable gzip and writes Parquet
  partitioned by application year; DuckDB does the local profiling over that Parquet. On
  Windows, Spark's native Parquet write needs Hadoop's `winutils`, so the local branch
  writes through pyarrow instead of installing a third-party binary.
- **Same code, two environments**: the notebook also ran natively on Databricks Free
  Edition (serverless), with identical results.

| Check | Local (Windows) | Databricks (Linux) |
|---|---|---|
| Rows | 27,648,741 | 27,648,741 |
| Corrupt rows after parsing | 0 | 0 |
| `risk_score` present | 33.1% | 33.1% |
| Top state | CA 3,242,169 | CA 3,242,169 |

Profiling findings that shaped the model:

- **`risk_score` coverage is temporal, not uniform**: about 86% through 2014, 17.85% in
  2015, about 54% in 2017 and 6.83% in 2018. Under walk-forward validation a raw
  `risk_score` would be rich in training years and nearly empty in test years, so it is
  not used.
- **`dti` is not on the same scale in both populations**: the approved `dti` excludes the
  mortgage and the Lending Club loan; the rejected `dti` is a text field of the denied
  application. Means are 26.58 (rejected) against 17.55 (approved); the two are compared
  in shape, not in level.
- **Employment length uses the same format in both**, and the requested amount does not
  separate the populations (mean 13,133 rejected against 13,091 approved).
- **`state` is complete** (50 states and DC); Iowa appears only 456 times because Lending
  Club did not operate there.

## Phase 2a: thin model, parcelling and profit (`notebooks/17` to `19`)

**Shared features, treated by mechanism.** A model that scores both populations can only
use what both have: amount, `dti` and employment length.

| Feature | Mechanism found | Treatment |
|---|---|---|
| `dti` = -1% (4.35%) | "not reported" sentinel | flag + median imputation |
| `dti` = 100% (4.93%) | right-censoring (spike about 170x its neighbours) | flag, value kept at 100 |
| `dti` = 9999 / 99999 / 199998% (0.3%) | redundant sentinel | dropped |
| `dti` real tail 100-1000 | genuine extreme debt | kept |
| employment length `< 1 year` | real signal, the main driver of rejection | kept as 0 |
| amount <= 0 (1,288 rows) | impossible value | dropped |

This keeps 1,927,007 rejected applications that a blanket `dti > 100` filter would have
discarded; 27,564,714 remain. The flags are not model features: they are always 0 among
approved loans, so a model trained on approvals cannot learn a weight for them.

**Thin model.** Logistic regression on the three shared features. Out-of-sample AUC from
4-fold stratified cross-validation on approved loans: **0.5620 ± 0.0027**. The benchmark
is the same model with no reject information ("ignore rejects").

**Parcelling**, implemented by hand (there is no maintained Python library; the R package
`scoringTools` is the reference). Rejects are scored, grouped into score bands, and given
the band's expected bad rate times a multiplier. The multiplier is swept over 1.0-3.0
rather than fixed, because reject labels are unknown: with an approved default rate of
12.43%, a plausible rejected rate of 18-31% implies a factor of about 1.5-2.5. At 4-5x
the inferred bad rate saturates at 1.0 in the worst bands and reaches about 70% of the
whole rejected population, which is not plausible for this data.

**Profit metric.** Expected profit per loan,
`PD·A·(1+i)·(1−LGD) + (1−PD)·A·(1+i) − A` (Kozodoi et al., arXiv:2407.13009), with LGD
swept from 0.5 to 0.9. Validated on approved loans before any use: it reproduces the
published **$242,230,710.89** at the 0.31 threshold, and an optimal threshold of 0.26 at
LGD 0.5. Strategies are compared at the same acceptance rate, not at a common numeric
threshold, because the score scales differ (XGB 78 features vs thin model 3 features): at
0.26, every reject would be accepted. An extra multiplier for censored `dti` was tested
this way and removed: the profit difference was 0.00 in all 24 combinations, because
those applications never reach the accepted group.

## Why the result cannot be validated

1. **The shared signal is weak.** An AUC of 0.5620 is barely above chance; labels
   inferred from it are close to noise.
2. **No rejected application has an outcome.** Kickout/AUK, the standard RI evaluation,
   needs labeled rejects (Kozodoi et al., arXiv:1909.06108, had an unbiased sample of
   1,967 rejects accepted on purpose). There is no such column anywhere in this data.
3. **There is no true population default rate to check against.** The *Illusion of
   Improvement* (Scarone & Baeza-Yates, arXiv:2606.18479, ECML PKDD 2026) shows that
   evaluating RI on approved loans rewards extreme decision boundaries; the check it
   proposes needs the population's real default rate. Here the symptom is visible but not
   interpretable: the training default rate after RI grows with the multiplier, and
   nothing in the data says whether that is a correction or a new bias.

| Multiplier | Training default rate after RI | Inflation vs approved (12.43%) |
|---|---|---|
| 1.0 | 0.1417 | +0.0173 |
| 1.5 | 0.2121 | +0.0878 |
| 2.0 | 0.2825 | +0.1581 |
| 2.5 | 0.3530 | +0.2287 |
| 3.0 | 0.4235 | +0.2992 |

4. **Phase 2b: the most sophisticated evaluation inherits the limitation**
   (`notebooks/20`). A bias-aware Bayesian evaluation of the expected portfolio default
   rate, following Kozodoi et al. (arXiv:2407.13009), returns close to the prior it is
   given: the estimate moves with the prior at a slope of about 0.99, with the same range
   (about 0.396) at acceptance rates of 10%, 30% and 50%. Rejects outnumber the approved
   training rows about 159 to 1, and a 0.56 AUC cannot push them out of the accepted set,
   so the unlabeled majority decides the metric.

## What would make it testable

Controlled exploration: approve a small random share of would-be rejects (2-5% is the
range discussed in the literature) and observe their outcomes. That is a decision a
lender makes going forward; it cannot be reconstructed from historical data.

## Vintage cross-check (`notebooks/21`)

A vintage analysis of the approved book found that a true roll rate cannot be computed
(`loan_status` has only two terminal values and the `mths_since_*` columns are snapshots,
not a monthly series). It also traced why 60-month loans disappear in 2014-2015 and
arrived, independently, at the maturity cutoff already documented in `docs/scope.md`: a
cross-check of the cleaning, not a new finding. Because 60-month loans default about
twice as often as 36-month ones, the apparent improvement in 2014-2015 can be a change in
mix rather than in credit quality.

## Reproducing

`requirements-reject.txt` holds the extra dependencies (PySpark, DuckDB). Run
`notebooks/16_reject_ingestion_profile.py` first (local branch, or the Databricks branch
on a Unity Catalog volume; `notebooks/16_reject_validation_databricks.ipynb` is the
recorded Databricks run), then `17` to `21` from the repository root. The diagnostics
behind each treatment decision are in `notebooks/scratch/`.

## References

- Hand, D. J. & Henley, W. E. (1993). Can reject inference ever work? *IMA Journal of
  Mathematics Applied in Business and Industry*, 5(1), 45.
- Kozodoi, N. et al. Shallow self-learning for reject inference in credit scoring.
  arXiv:1909.06108.
- Kozodoi, N. et al. Fighting sampling bias: a framework for training and evaluating
  credit scoring models. arXiv:2407.13009.
- Scarone, B. & Baeza-Yates, R. The Illusion of Improvement: Reject Inference Strategies
  in Credit Scoring. arXiv:2606.18479 (ECML PKDD 2026).
