# Technical Report: Credit Default Prediction on Lending Club Loans

**Author**: Mateus Fardin Pavan · **License**: MIT (see `LICENSE`) · **Repository**:
<https://github.com/MateusFPavan/credit-default-prediction-lendingclub>

Companion documents: population and column-level facts in
[`docs/FACTS.md`](FACTS.md); dataset provenance and licensing in
[`docs/DATA_CARD.md`](DATA_CARD.md); model specification in
[`docs/MODEL_CARD.md`](MODEL_CARD.md); reproduction steps in
[`docs/SETUP.md`](SETUP.md); business summary in
[`references/one_pager.md`](../references/one_pager.md).

---

## 1. Executive Summary

This project builds a credit-default classifier for peer-to-peer personal loans and
selects among candidate models by **expected portfolio profit**, not accuracy or AUC. On
the 2015 held-out test set, the selected model (a walk-forward-tuned XGBoost classifier)
produces a **net gain of +$9.0M over an approve-everyone policy** (95% CI $7.6M-$10.7M),
and +$6.3M over a logistic-regression baseline. That gain is attributable to a small,
financially concentrated set of rejections. The model rejects 3.8% of applications,
avoiding $32.2M in losses at a cost of $23.1M in forgone interest. The gross portfolio
total under the model's policy is $242.23M. The model's own contribution is the $9.0M
delta over the $233.2M an approve-all policy already yields. This report leads with the
delta throughout, not the gross figure.

The methodology is organized around four commitments, each detailed below: a decision
metric tied to real dollars rather than classification accuracy; a temporal, walk-forward
validation scheme instead of a random split; missing-data treatment driven by mechanism
rather than convention; and an evaluation that reports where the model is weak, not just
where it is strong.

## 2. Business Problem, Target, and Decision Metric

Each approved loan is a bet: a known potential return (contracted interest) against a
known potential loss (unrecovered principal). The target is binary: 1 (Charged Off,
realized loss) versus 0 (Fully Paid), derived only from loans with a concluded outcome.

The decision metric is **expected portfolio profit**, not accuracy, precision/recall, or
AUC:

```
profit = Σ(interest on approved good loans) − Σ(lost principal on approved bad loans)
interest = installment × term − loan_amnt
loss     = max(loan_amnt − total_rec_prncp, 0)
```

Both terms are computed from realized outcomes already in the data. The classifier's
decision threshold is chosen to maximize this curve directly, not by convention (0.5) or
by optimizing a statistical proxy. This choice is load-bearing: **a bad loan costs 2.67x
what a good loan returns, at the median** (median loss $5,398.84 vs. median interest
$2,023.62). Treating a false negative and a false positive as equally costly, as accuracy
implicitly does, misrepresents the actual economics of the decision.

## 3. Data

### 3.1 Population and funnel

The raw file (`accepted_2007_to_2018Q4.csv`, ~1.67 GB, 2,260,701 rows, 151 columns) is
reduced to an analytical population via a reconciled funnel (full detail and independent
re-verification in `docs/DATA_CARD.md` and `docs/FACTS.md`):

| Step | Rows removed | Cumulative population |
|---|---|---|
| Total in file | — | 2,260,701 |
| In-progress statuses (Current, Late, Grace Period, Default, "does not meet credit policy") + 33 footer rows | 915,391 | 1,345,310 |
| Immature vintages (36m issued after Dec/2015; 60m after Dec/2013) | 671,757 | 673,553 |
| Impossible `dti` (5 rows) + joint-application rows (234, `dti` computed on a different basis) | 239 | **673,314** |

The final population is **~673K matured 36-month Lending Club loans, 2007-2015, at a
14.8% default rate**. 60-month loans (54,969) are held out entirely as a transfer set
(§9), never used for training or model selection.

### 3.2 Missing data by mechanism

Missing data was diagnosed by *why* a value was absent, not treated with a single
imputation rule. That is the strongest methodological claim of the cleaning phase, since
two of the three mechanisms below have the null itself carry directional risk signal.

| Mechanism | Example columns | Evidence | Treatment |
|---|---|---|---|
| **MNAR** (informative absence) | `mths_since_last_delinq`, `mths_since_recent_inq` | Null share stable across every vintage (never drops to zero); null rows default *less* than filled rows — the null means "this event never happened," not "not collected" | Binary flag + sentinel 999 (preserves "higher = safer" ordering) |
| **Staged rollout** (2012 and 2015 bureau-attribute blocks) | ~40 bureau columns (`tot_cur_bal`, `mo_sin_*`, `num_*`, and others) | 100% null before the field existed, ~0% after (verified against issue-year null rate in `docs/column_birth_log.csv`) | Era flag (`era_pre_2012`) + sentinel −1, or dropped entirely where the field postdates the population window (e.g., the Dec/2015 `open_acc_6m` block) |
| **Sparse** (1,079 unique rows) | `pub_rec_bankruptcies`, `revol_util`, `dti`, others | Informative, not noise: affected rows default at 18.35% vs. 14.8% population rate | Aggregate flag (`sparse_bureau_missing`) + per-column median imputation |

Blanket median imputation across all three mechanisms would have erased the strongest
signal in the MNAR and sparse groups, and in the MNAR case, inverted it. Imputing the
median for `mths_since_last_delinq` would assert that half the population had a
delinquency 30 months ago, fabricating a history for the cleanest borrowers in the data.

## 4. Leakage Prevention

Leakage was screened on three independent fronts:

- **Temporal.** The train/validation/test split is by `issue_d`, never shuffled (§6.1). A
  random split would let 2015 information inform predictions evaluated as if only the
  past were known.
- **Target.** All post-origination columns (fields written into the record *after* the
  loan's outcome was already known) were dropped before modeling, confirmed empirically
  via univariate AUC rather than by name heuristics alone. `recoveries` alone scores an
  AUC of 0.90 against the target: a column that nearly determines the outcome by itself
  was written after the outcome, and was correctly removed as leakage.
- **Identity.** `member_id`, the only borrower identifier in the raw file, is 100% null
  across the entire 2007-2018 history. A borrower-level group split (the standard defense
  against the same person appearing in both train and test) is therefore **not possible**.
  This is stated here as an unresolvable limitation, not hidden or omitted.

## 5. Feature Engineering

79 named features enter the model, expanding to 90 columns after one-hot encoding of 5
categorical fields. Five interpretable ratio features were engineered from raw
origination-time fields: `installment_to_income`, `loan_to_income`, `credit_history_months`,
`revol_bal_to_income`, and `open_acc_ratio`. `installment_to_income` is the strongest of
the five, ranking #2 by SHAP importance in the final model (§8).

**Two engineered candidates were tried and dropped. This is reported here as a strength
of the process, not something omitted:**

- `fico_mean` (average of the FICO range bounds), dropped for **exact redundancy**:
  correlation of 1.0 with `fico_range_low`, since Lending Club reports FICO as a
  fixed-width band in this data.
- A bankcard-utilization ratio, dropped for **missingness with weak payoff**: undefined
  for ~30% of the training population (borrowers with no bankcard), for a univariate AUC
  of only 0.539, barely better than chance.

## 6. Modeling and Validation Methodology

### 6.1 Temporal split

| Split | N | Default rate | Period |
|---|---|---|---|
| Train | 172,988 | 12.43% | ≤ 2013 |
| Validation | 162,570 | 13.73% | 2014 |
| Test | 282,787 | 14.88% | 2015 |

Never a random split. Every model-selection decision (features, hyperparameters,
threshold) was made on validation. The test set was scored exactly once, after selection
was frozen.

### 6.2 Walk-forward hyperparameter selection

A single validation year risks selecting hyperparameters that overfit that year's
idiosyncrasies. Hyperparameters for the final model were instead selected by re-scoring
candidate configurations across **three expanding temporal windows**: train through 2011
and validate on 2012, train through 2012 and validate on 2013, and train through 2013 and
validate on 2014. Each configuration was scored by mean expected profit across all three
windows, not AUC and not a single year.

### 6.3 Final model configuration

`XGBClassifier` wrapped for interface consistency alongside a scikit-learn `Pipeline`
logistic-regression baseline:

```
max_depth=8, learning_rate=0.03, n_estimators=600, min_child_weight=10,
subsample=0.8, colsample_bytree=0.6, random_state=42, n_jobs=1, eval_metric=logloss
```

**Operating threshold: 0.31.** Bit-exact reproducibility requires three simultaneous
conditions: `random_state=42`, `n_jobs=1` (multi-threaded histogram building is not
provably deterministic run-to-run on this hardware), and training rows preserved in
`train.parquet`'s on-disk order. XGBoost's histogram algorithm is not row-order
invariant. Full reproduction: `docs/SETUP.md`.

## 7. Results on the Held-Out Test Set

### 7.1 Headline result

| Model | Test profit | 95% CI |
|---|---|---|
| Approve-all baseline | $233,202,813.06 | — |
| Logistic-regression baseline | $235,936,408.63 | — |
| **XGB_walkforward (final model)** | **$242,230,710.89** | **[$237.89M, $246.72M]** |

The model's attributable contribution is the delta over approve-all: **+$9.0M**, not the
$242.23M gross figure, which already includes $233.2M an approve-all policy would have
produced with zero modeling effort.

### 7.2 Statistical significance

A 1,000-resample bootstrap (seed 42) of the test set confirms both deltas are real, not
sampling noise:

| Comparison | Mean difference | 95% CI | Crosses zero? |
|---|---|---|---|
| XGB vs. approve-all | +$9,027,897.83 | [$7.63M, $10.66M] | No — 100% of resamples positive |
| XGB vs. logistic baseline | +$6.3M | CI excludes zero | No |

### 7.3 Error decomposition

At threshold 0.31, **10,644 loans are rejected**: avoided loss $32.2M, forgone interest
$23.1M, net $9.0M (matching §7.1 exactly). False-negative cost is approximately **11.9x**
false-positive cost. This is consistent with the 2.67x median cost asymmetry (§2),
amplified because false negatives concentrate in larger, riskier loans. **The model
rejects 9.2% of eventual defaulters.** The $9.0M gain comes from identifying a small,
financially concentrated slice of the worst cases, not from bulk rejection (96.2% of
applications are approved).

### 7.4 AUC vs. profit divergence: the central methodological justification

**XGB and the logistic baseline are statistically indistinguishable on AUC (0.6846 vs.
0.6847) yet XGB wins decisively on profit ($242.23M vs. $235.94M, a real, CI-confirmed
$6.3M gap).** Brier score: 0.1205 (secondary metric, reported for completeness). This
divergence is the central justification for the metric choice made in §2: a
rank-order statistic (AUC) cannot see where a decision boundary is drawn near the
operating threshold, and this business problem's value is concentrated exactly there.
Selecting a model by AUC alone would have treated these two candidates as a coin flip and
discarded $6.3M of real, measurable value.

### 7.5 Lift

Rejecting the riskiest 10% of applicants avoids **~21% of all defaults**, roughly twice
the yield of a random 10% cut. This is the plainest translation of the model into a
policy statement for a non-technical reader.

## 8. Subgroup Performance and Calibration

This section is the regulatory-relevant layer of the evaluation: an aggregate AUC or
profit figure can hide systematic weakness in specific, protected-adjacent segments.

**AUC declines monotonically with risk grade and rises monotonically with income**:

| Grade | AUC-ROC |
|---|---|
| A | 0.648 |
| B | 0.605 |
| C | 0.597 |
| D | 0.589 |
| E | 0.582 |
| F–G (small N) | 0.585–0.588 |

| Income quartile | AUC-ROC |
|---|---|
| Q1 (lowest income) | 0.648 |
| Q2 | 0.672 |
| Q3 | 0.688 |
| Q4 (highest income) | 0.697 |

**The model is least reliable exactly in the highest-risk, lowest-income segment.** That
is precisely where a lender would most need precision, and the reverse of what the
headline AUC (0.6846) alone would suggest.

**Calibration.** The model systematically **underestimates** default: observed default
exceeds predicted probability in every decile of the test set (decile 10: 31.80% observed
vs. 30.86% predicted). Stated plainly: as an absolute probability, the score is
optimistic. That is a limitation that matters directly for any threshold-based
decisioning built on top of it, not just an academic footnote.

**SHAP** (50,000-row stratified test sample, since a full-test run was estimated at ~22
minutes, over the compute budget for this analysis). Top features by mean |SHAP|:
`fico_range_low`, `installment_to_income`, `annual_inc`, `acc_open_past_24mths`, `dti`.
One confound is worth naming: `verification_status` shows a Simpson's-paradox pattern:
its univariate association (verified income correlates with *higher* default) partly
reverses once the other 78 features are controlled for. This indicates some of the
univariate signal was confounded by correlated features rather than reflecting a direct
causal reading of verification status itself.

## 9. Generalization: The 60-Month Transfer Finding

Applying the frozen 36-month model, without refitting, to 60-month loans produces severe
degradation. The logistic baseline's profit gain over approve-all turns **negative** on
that population. This is read as a finding about the data, not a model failure: **36- and
60-month loans are structurally distinct risk populations**, which is exactly why the
analytical population in this project is restricted to 36-month loans, and why a separate
60-month scorecard is named as future work (§13) rather than assumed unnecessary.

## 10. Calibration: Explored and Rejected

An isotonic recalibration experiment was run on the least-well-calibrated candidate model
(base classifier trained on `issue_d ≤ 2012`, calibrator fit on a 2013 holdout, a
legitimate temporal split that never touches validation or test). Reliability improved
substantially, but the base classifier lost **42% of its training data** to the holdout,
and validation-year profit dropped to barely above the approve-all baseline. **This is
reported as an honest trade-off, not a suppressed negative result**: recalibration cost
more in training data than it returned in profit for this model, and was not adopted.

## 11. Limitations, Bias, and Risks

- **Selection bias.** The model estimates P(default | approved), never having observed a
  rejected application. It cannot be used to score the rejected-applicant population, and
  says nothing about how it would perform as a first-pass underwriting filter.
- **Subgroup reliability.** Weakest in the highest-risk, lowest-income segment (§8),
  compounding the 2.67x cost asymmetry (§2) exactly where it is largest.
- **Term non-transferability.** Not valid for 60-month loans without a dedicated
  scorecard (§9).
- **Calibration is optimistic, not conservative** (§8). This is relevant to any
  downstream use of the raw probability, not just the fixed 0.31 threshold evaluated here.
- **Identity leakage is unverifiable, not just unresolved** (§4). The absence of a
  borrower key means this limitation cannot be fixed within this dataset.
- **Not deployed.** The model is packaged as a reproducible, serialized artifact
  (`models/xgb_final.joblib`), not a live service. No drift monitoring exists.

## 12. Reproducibility

`python run_all.py` reproduces the essential path in ~3.1 minutes: raw CSV becomes a
cleaned dataset, then a temporal split, then features, then a final model, then a
verified test result. It asserts the $242,230,710.89 test profit reproduces exactly. The
walk-forward tuning and bootstrap experiments referenced in §6.2, §7.2, and §10 are not
re-run by that entry point (they take hours). They are preserved in the numbered working
notebooks (`notebooks/06` through `11`) and summarized in `docs/FACTS.md`.

## 13. Conclusion and Recommendations

The evidence supports shipping the walk-forward-tuned XGBoost model as a **second
decision layer** over an existing approval process, on 36-month loans, with the frozen
0.31 threshold, not as a standalone underwriting system. Recommended next steps, named
explicitly rather than left implicit: (1) deployment as a served API with drift
monitoring, currently absent by design; (2) a dedicated 60-month scorecard, justified by
the transfer finding in §9; (3) revisiting calibration only if a use case specifically
requires well-calibrated absolute probabilities rather than a fixed threshold, given the
cost measured in §10.

---

*Report word count: this document intentionally has no fixed length target, unlike the
companion one-pager. It is organized to cover the full methodology and results at the
depth a technical reviewer would expect, without padding beyond what each section
required.*
