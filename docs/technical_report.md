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

78 named features enter the model, expanding to 90 columns after one-hot encoding of 4
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

### 7.6 Fit-to-fit noise: what survives it, and what a feature ablation measures

The frozen configuration draws at random twice per tree: `colsample_bytree=0.6` grows
each of the 600 trees on 60% of the columns, and `subsample=0.8` on 80% of the rows. The
published figure is one draw of that randomness (`random_state=42`). Refitting the same
configuration eight times, with the same rows and columns and only the seed changed, gives
a **standard deviation of $0.76M and a range of $2.10M** in test profit. The published
seed ranks third of eight, 0.94 SD above the mean. It remains the published figure: no
seed was chosen after seeing results, and the sweep measures uncertainty around it rather
than selecting anything (`scripts/ablation_noise_floor.py`; output in
`reports/ablation_noise_floor.txt`).

**The headline survives.** The logistic baseline has no random draw, so the comparison in
§7.4 holds against the whole distribution: XGB beats it by $5.58M at the eight-seed mean,
**7.4 SD** of fit-to-fit noise, and by $4.55M even at the worst seed. The §7.4 conclusion
does not depend on a favourable draw.

**Single-feature ablations do not survive it, and the reason is mechanical.** The column
draw depends on how many columns there are, so removing *any* column re-randomises all 600
trees. A leave-one-feature-out delta is therefore the feature's effect plus a fresh draw.
Two measured consequences:

- Removing a column the model never splits on still moved profit by $90K. Two such
  ablations produced bit-identical ensembles, down to the tree dumps, because the two
  89-column matrices differed only in a column that neither fit used.
- Because the baseline is one draw sitting above the mean, a fresh draw is expected to
  land lower. Regression to the mean alone predicts about **-$0.71M** per ablation; the
  observed median of the 78 ablation deltas is **-$0.91M**. Centred on zero, 20 of the 78
  deltas exceed 2 SD and all 20 are negative, which is the signature of a ranking measuring
  the shift rather than the effects. Centred on their median, 6 of 78 exceed 2 SD, and three
  of those are positive: removing `avg_cur_bal`, `bc_util` or `tot_hi_cred_lim` raised
  profit by more than the noise explains.

The clearest illustration is a contrast already in this repository. Removing
`application_type` changed nothing: it never contributed a column, the count stayed at 90,
nothing was re-drawn, and profit reproduced to the cent. Removing `emp_length_anos` took
the count to 89 and carries the full re-draw term; its $1.74M delta sits 1.1 SD from the
typical ablation, inside the noise. The feature stays in the model, which is the low-risk
side of the decision, but that ablation does not establish the size of its contribution.

The seed sweep redraws rows as well as columns, while removing a column redraws columns
only, so $0.76M is an **upper bound** on the re-draw noise. A delta beyond it is signal
with room to spare; a delta inside it is not thereby shown to be noise.

## 8. Subgroup Performance and Calibration

This section is the regulatory-relevant layer of the evaluation: an aggregate AUC or
profit figure can hide systematic weakness in specific, protected-adjacent segments.

**AUC is highest in grade A and lower in every riskier grade; it rises monotonically with
income.** The grade pattern is not a steady slope: a large drop from A to B, a gentler
decline through E, the lowest grade, and no further decline in F and G, whose estimates
rest on few loans. The logistic baseline is shown alongside, for the reason below:

| Grade | XGB | Logistic baseline |
|---|---|---|
| A | 0.6477 | 0.6411 |
| B | 0.6048 | 0.5992 |
| C | 0.5965 | 0.5854 |
| D | 0.5889 | 0.5735 |
| E | 0.5818 | 0.5675 |
| F (1,358 loans) | 0.5878 | 0.5679 |
| G (244 loans) | 0.5854 | 0.5828 |

| Income quartile | XGB | Logistic baseline |
|---|---|---|
| Q1 (lowest income) | 0.6480 | 0.6535 |
| Q2 | 0.6722 | 0.6765 |
| Q3 | 0.6881 | 0.6883 |
| Q4 (highest income) | 0.6974 | 0.6885 |

**How much of this is the model.** A within-segment AUC depends on the segment as well as
on the model: cutting by a variable that is itself a risk score, as the grade is, leaves
each slice more uniform and harder for any model to rank. The test is to put a second,
unrelated model through the same cuts (`scripts/subgroup_auc_vs_baseline.py`; output in
`reports/subgroup_auc_vs_baseline.txt`):

- **By grade, the gradient belongs to the population.** The baseline drops from A to B by
  the same amount as the XGB (0.042 against 0.043; difference +0.001, 95% CI -0.006 to
  +0.008). Within every grade the XGB's point estimate is ahead of the baseline's, and the
  lead survives a Bonferroni correction for the seven comparisons in grades C and D
  (+0.011 and +0.015).
- **By income, part of the gradient is the model's.** The baseline's AUC also rises with
  income, by 0.035 from Q1 to Q4; the XGB's rises by 0.049. The difference, +0.015 (95% CI
  +0.009 to +0.021), comes from both ends: the XGB is ahead of the baseline in the highest
  quartile (+0.009) and **behind it in the lowest** (-0.006, below zero after the
  correction for four comparisons). Refitting the XGB with eight seeds does not change
  this: in the lowest quartile every refit scores below the baseline.

**So the model is not weaker where risk is higher; it is weaker, relative to a simpler
model, for the lowest-income borrowers.** That is a narrower and more specific limitation
than the raw table suggests, and it is the one this report carries forward.

**Calibration.** The model systematically **underestimates** default: observed default
exceeds predicted probability in every decile of the test set (decile 10: 31.80% observed
vs. 30.86% predicted). Stated plainly: as an absolute probability, the score is
optimistic. That is a limitation that matters directly for any threshold-based
decisioning built on top of it.

**SHAP** (50,000-row stratified test sample, since a full-test run was estimated at ~22
minutes, over the compute budget for this analysis). Top features by mean |SHAP|:
`fico_range_low`, `installment_to_income`, `annual_inc`, `acc_open_past_24mths`, `dti`.
One confound is worth naming: `verification_status` shows a Simpson's-paradox pattern:
its univariate association (verified income correlates with *higher* default) partly
reverses once the other 77 features are controlled for. This indicates some of the
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

**Second pass, 2026-08-31: the post-processing route, and the cause.** The experiment
above tests *retraining*. The complementary question was left open: what does calibration
cost and return if the model is left untouched and only the score is transformed?
Post-processing costs no training data at all, so the trade-off that killed the first
attempt does not apply. Isotonic regression and Platt scaling were fitted on the score
without leakage (calibrator fit on one half of the test set, evaluated on the other).

| | Brier | log-loss | AUC | mean bias |
|---|---|---|---|---|
| raw | 0.1211 | 0.3985 | 0.6818 | -0.0251 |
| isotonic | **0.1203** | 0.3950 | 0.6814 | -0.0007 |
| Platt | 0.1213 | 0.3987 | 0.6818 | -0.0006 |

Both drive the mean bias to ~0. Isotonic improves Brier by **0.6%** — which is
essentially the entire gain theoretically available, since the squared bias is only
0.00063. Platt removes the bias but makes Brier slightly *worse*, indicating the
miscalibration is not sigmoid-shaped. AUC moves by 0.0004 in both, confirming the
transform is monotone. Re-mapping the 0.31 threshold into the calibrated space changes
profit by $106,162 on $242.2M — **0.04%, an order of magnitude inside the profit
confidence interval** reported in §7, and driven by 162 applicants sitting on an isotonic
plateau where the calibrated model genuinely cannot separate them.

**The cause, which neither pass had established.** The model's mean prediction on the 2015
test set is **0.1240**; the base rate of the split it was *trained* on is **0.1243**. It
reproduces its own training period's default rate to three decimal places, while 2015
defaulted at **0.1488**. The miscalibration is base-rate shift, not a defective estimator:
the model is well calibrated for 2007-2013 and is being asked about 2015.

| split | N | default rate |
|---|---|---|
| train (2007-2013) | 172,988 | 0.1243 |
| validation (2014) | 162,570 | 0.1373 |
| test (2015) | 282,787 | **0.1488** |

And the shift is **cyclical rather than trending**: within the training window the annual
default rate runs 17.9% (2007) → 9.9% (2010) → 13.6% (2012) → 12.3% (2013), tracking the
credit cycle. This is what makes the second rejection rest on more than the 0.6% figure:
**there is no stable target to calibrate to.** Validation (2014) sits between train and
test, so calibrating to it would be less wrong rather than right, and a calibrator fitted
on any past window is already stale for the next one. Honest recalibration is therefore an
operational commitment to refit periodically, which a project with no production traffic
cannot make.

Documented instead: a consumer needing an absolute PD for a 2015-like population can apply
the measured **+2.45-point** offset, and knows why it is there. The two passes are
complementary — the first shows the retraining route is too expensive, the second that the
cheap route buys almost nothing, and the cause explains both.

## 11. Limitations, Bias, and Risks

- **Selection bias.** The model estimates P(default | approved), never having observed a
  rejected application. It cannot be used to score the rejected-applicant population, and
  says nothing about how it would perform as a first-pass underwriting filter.
- **Subgroup reliability.** Within-grade AUC is lower in the riskier grades, but a logistic
  baseline shows the same pattern, so it is a property of the population (§8). The
  model-specific weakness is narrower: in the lowest-income quartile the XGB ranks
  borrowers slightly worse than the baseline (§8), compounding the cost asymmetry of §2.
- **Term non-transferability.** Not valid for 60-month loans without a dedicated
  scorecard (§9).
- **Calibration is optimistic, not conservative** (§8). This is relevant to any
  downstream use of the raw probability, not just the fixed 0.31 threshold evaluated here.
- **Identity leakage is unverifiable, not just unresolved** (§4). The absence of a
  borrower key means this limitation cannot be fixed within this dataset.
- **Served and monitored, not for real lending.** The model is served behind a FastAPI
  endpoint, containerized and CI-tested on every push, with a PSI-based drift monitor
  (`docs/MODEL_CARD.md` §10). This proves the artifact is servable and observable; it does
  not change the selection-bias limitation above, and the model must still not drive a
  real credit decision.
- **Feature-level attribution by ablation is not established.** Removing a single
  feature re-randomises the ensemble, so most single-feature ablation deltas fall inside
  the fit-to-fit noise (§7.6). Model-level comparisons are unaffected; statements of the
  form "feature X is worth $Y" are not supported by the ablations run here.

## 12. Reproducibility

`python run_all.py` reproduces the essential path in ~3.1 minutes: raw CSV becomes a
cleaned dataset, then a temporal split, then features, then a final model, then a
verified test result. It asserts the $242,230,710.89 test profit reproduces exactly. The
walk-forward tuning and bootstrap experiments referenced in §6.2, §7.2, and §10 are not
re-run by that entry point (they take hours). They are preserved in the numbered working
notebooks (`notebooks/06` through `11`) and summarized in `docs/FACTS.md`.
`python scripts/subgroup_auc_vs_baseline.py` reproduces the baseline comparison in §8 the
same way, also in about ten minutes.
`python scripts/ablation_noise_floor.py` reproduces §7.6 in about ten minutes; it checks
the baseline against the published figure before measuring anything, and stops if it
does not match.

## 13. Conclusion and Recommendations

The evidence supports shipping the walk-forward-tuned XGBoost model as a **second
decision layer** over an existing approval process, on 36-month loans, with the frozen
0.31 threshold, not as a standalone underwriting system. Recommended next steps, named
explicitly rather than left implicit: (1) a dedicated 60-month scorecard, justified by
the transfer finding in §9; (2) automated retraining execution (scheduling and alerting
on top of the retraining-trigger policy already defined in `docs/MODEL_CARD.md` §10-11);
(3) revisiting calibration only if a use case specifically requires well-calibrated
absolute probabilities rather than a fixed threshold, given the cost measured in §10.
Deployment as a served API with drift monitoring, listed here in earlier versions of this
report, shipped in v2.0.0 (`docs/MODEL_CARD.md` §10; `CHANGELOG.md`).

---

*Report word count: this document intentionally has no fixed length target, unlike the
companion one-pager. It is organized to cover the full methodology and results at the
depth a technical reviewer would expect, without padding beyond what each section
required.*
