---
model_name: credit-default-prediction-lendingclub / XGB_walkforward
model_version: 2.0.0
version_date: 2026-08-06
task: binary-classification (credit default prediction)
library: scikit-learn, xgboost
language: en
license: MIT (see LICENSE at repository root)
data_license: CC0-1.0
---

# Model Card: XGB_walkforward (Lending Club Credit Default)

**Model version: 2.0.0 — 2026-08-06.** Version history in §12.

Related docs: data card at [`docs/DATA_CARD.md`](DATA_CARD.md); setup and reproduction at
[`docs/SETUP.md`](SETUP.md); full verified facts sheet at [`docs/FACTS.md`](FACTS.md).

## 1. Model summary

A gradient-boosted tree classifier (XGBoost, alongside a scikit-learn `Pipeline` baseline)
that scores probability of default on Lending Club peer-to-peer personal loans, selected
and evaluated to maximize **expected portfolio profit** rather than accuracy or AUC.

## 2. Model details

- **Architecture**: `XGBClassifier` (gradient-boosted trees) as the final estimator; a
  `StandardScaler` + `LogisticRegression` `Pipeline` serves as the compared baseline.
- **Hyperparameters** (frozen): `max_depth=8`, `learning_rate=0.03`, `n_estimators=600`,
  `min_child_weight=10`, `subsample=0.8`, `colsample_bytree=0.6`, `random_state=42`,
  `n_jobs=1`, `eval_metric=logloss`. Operational decision threshold: **0.31**.
- **Features**: 79 named features, expanding to 90 columns after one-hot encoding 5
  categorical fields (from `train.parquet`'s 89 columns; the rest are target, `EVAL_ONLY`,
  excluded-by-design, and one dropped redundant column).
- **Versions**: scikit-learn 1.9.0, xgboost 3.3.0, Python 3.12.10.
- **Trained**: 2026-07-11. Serialized at `models/xgb_final.joblib` (versioned in-repo,
  4.1 MB); hyperparameters, feature list, and a SHA256 of the exact `train.parquet` are
  recorded in `models/model_meta.json`.
- **License**: MIT (see LICENSE at repository root). Underlying data is CC0-1.0
  (Kaggle `wordsforthewise/lending-club`; see `docs/DATA_CARD.md`).
- **Author**: Mateus Fardin Pavan. Repository:
  <https://github.com/MateusFPavan/credit-default-prediction-lendingclub>. Contact:
  GitHub or LinkedIn (<https://www.linkedin.com/in/mateus-fardin-pavan/>).

## 3. Intended uses

- **Primary use case**: a *second decision layer* estimating default probability for
  already-funded-style personal loans, ranked and thresholded to maximize expected
  portfolio profit rather than classification accuracy.
- **Intended users**: credit-risk and lending-portfolio analysts, and reviewers of
  credit-scoring methodology (model validators, risk committees) evaluating the approach
  rather than consuming live scores.
- **Domain**: unsecured personal installment loans, 36-month term, U.S. peer-to-peer
  lending, vintages 2007-2015.
- **Educational/portfolio use**: this project is a methodology demonstration (temporal
  validation, profit-based evaluation, documented bias, served + monitored — §10). See
  `docs/DATA_CARD.md` §7 for the same framing on the data side.

## 4. Out-of-scope and misuse

- **Must not be used to score the rejected-applicant population.** The model estimates
  P(default | approved), never having seen a rejected application. See §9.
- **Must not drive real credit decisions.** The service in §10 packages the model for
  demonstration and methodology review, not live underwriting. This limit is about
  *modeling*, not engineering: the model is now served and monitored, but the selection
  bias above still means its score must not drive a real lending decision. Deployment
  proves the artifact is servable and observable — not that it is fit to underwrite.
- **Degrades in the highest-risk segment.** AUC falls from grade A (0.648) to G (0.585)
  and is lowest for the lowest-income quartile — least reliable where a lender most needs
  it (§8).
- **Not transferable to 60-month loans.** Applied without refitting, performance degrades
  severely (AUC 0.6846 → 0.6433; the logistic baseline's profit gain turns negative). The
  two terms are structurally distinct risk pools; the API enforces `term=36`.
- **Probabilities are optimistic, not conservative.** The model systematically
  underestimates default (§8). Do not treat the raw score as a conservative lower bound.

## 5. How to use

Programmatic use (batch/offline scoring):

```python
import joblib
from src.data import load_split, FEATURE_SET, CATEGORICAL_COLS
from src.features import build_features, prepare_X

model = joblib.load("models/xgb_final.joblib")

# df must have the same raw/cleaned schema as data/processed/*.parquet
df = load_split("test")  # or your own already-cleaned dataframe
df_feat = build_features(df)
X = prepare_X(df_feat, FEATURE_SET, CATEGORICAL_COLS)
X = X.reindex(columns=model.get_booster().feature_names, fill_value=0)  # aligns one-hot columns to training (verified: returns the 90 trained column names)

y_prob = model.predict_proba(X)[:, 1]
decision = ["reject" if p >= 0.31 else "approve" for p in y_prob]
```

For HTTP request/response scoring and containerized serving, see §10.

## 6. Training data

Lending Club accepted personal loans, 2007-2013 vintages, 36-month term (`train.parquet`,
N=172,988). Full provenance, the population funnel, per-column missingness mechanisms,
and licensing are documented in `docs/DATA_CARD.md` and `docs/FACTS.md`, and are not
repeated here.

## 7. Training procedure

Categorical fields are one-hot encoded and datetime fields converted to days-since-epoch
via `src.features.prepare_X`. Five ratio features (e.g., `installment_to_income`, `dti`)
are engineered via `src.features.build_features`. Missing values are resolved by
mechanism (MNAR, staged bureau-data rollout, sparse) into sentinels plus binary flags,
and are never blanket-imputed (`docs/DATA_CARD.md` §4).

**Validation is temporal / walk-forward, never a random split**: train ≤2013 (N=172,988),
validation 2014 (N=162,570), test 2015 (N=282,787), all 36-month. Hyperparameters were
selected across three expanding windows (train→2011/val 2012, →2012/val 2013, →2013/val
2014), each optimizing expected profit, not AUC; no single validation-year fit decided the
choice. Reproducibility is bit-exact only under `random_state=42`, `n_jobs=1`, and
`train.parquet`'s on-disk row order (XGBoost histogram is not row-order invariant). Full
steps: `docs/SETUP.md`.

## 8. Evaluation

**Test set**: 2015 vintages, 36-month term, N=282,787, touched once, after model
selection. **Primary metric: expected portfolio profit** (sum of interest on approved
good loans minus principal lost on approved defaults), at the frozen 0.31 threshold.

| Model | Test profit | 95% CI |
|---|---|---|
| Approve-all baseline | $233,202,813.06 | — |
| Logistic-regression baseline | $235,936,408.63 | — |
| **XGB_walkforward (this model)** | **$242,230,710.89** | **[$237.89M, $246.72M]** |

Net gain over approve-all: **+$9,027,897.83** [$7.63M, $10.66M], paired bootstrap does not
cross zero (100% of resamples favor this model). Net gain over the logistic baseline:
**+$6.32M**, CI does not cross zero.

**Secondary metrics**: AUC-ROC 0.6846, Brier 0.1205. Key insight: this model and the
logistic baseline have statistically indistinguishable AUC (0.6846 vs. 0.6847) yet this
model wins decisively on profit. AUC and the business metric diverge, which is why
profit, not AUC, is the reported decision metric.

**Error decomposition** (threshold 0.31): 10,644 loans rejected, avoided loss $32.15M,
forgone interest $23.12M, net $9.03M. False-negative cost is ~11.9x false-positive cost.

**Disaggregated by subgroup** (test): AUC declines monotonically by grade, 0.648 (A) to
0.585 (G), and is lowest for income quartile Q1 (0.648) vs Q4 (0.697) — least reliable in
the highest-risk, lowest-income segment.

**Calibration**: the model systematically **underestimates** default (observed exceeds
predicted in every decile; e.g. decile 10: 31.80% vs 30.86% predicted). Used as an
absolute probability the score is optimistic, which matters for any threshold decision.

**Explainability (SHAP, 50k stratified test sample)**: top features by mean |SHAP| are
`fico_range_low`, `installment_to_income`, `annual_inc`, `acc_open_past_24mths`, `dti`.
`verification_status` shows a split effect: `source verified` preserves the univariate
default-rate inversion, while plain `verified` reverses it under multivariate control (a
Simpson's-paradox confound). `era_pre_2012` has zero SHAP on the test set (the flag is
always 0 outside the training population).

## 9. Bias, risks & limitations

- **Selection bias**: estimates P(default | approved), never having seen rejected
  applicants; valid only as a second layer over an already-approved book, not a first-pass
  filter (`docs/DATA_CARD.md`).
- **Subgroup reliability**: weakest where risk is highest (§8), and the cost asymmetry is
  large — a bad loan costs 2.67x what a good loan returns at the median ($5,398.84 vs
  $2,023.62) — so subgroup weakness concentrates where errors are most expensive.
- **Temporal drift**: hyperparameters and the threshold were fit on 2007-2014 data. Drift
  of 2015+ vintages away from that distribution is now observable via the PSI monitor
  (§10), which is how the temporal-drift risk is surfaced rather than left unmonitored.
- **Term non-transferability**: not valid for 60-month loans without a separate scorecard
  (§4); the API enforces `term=36`.

## 10. Production: serving, drift monitoring, and retraining

The model is served and monitored, not an offline artifact (this replaces the earlier
"next step" framing, §12). Scope stays a demonstration: §4 governs (no real decisioning),
and "served" means a reproducible inference stack, not a hosted service under SLA.

**Serving.** `src/api.py` exposes the model with FastAPI: `POST /score` takes a raw loan
application and returns `probability_default` plus an `approve`/`reject` decision at the
0.31 threshold; `GET /health` reports readiness. Inference reuses the exact training-time
encoding (`src/scoring.py` → `src/cleaning.py` → `src/features.py`), never a
reimplementation, normalizing raw inputs into the same sentinel/flag treatment as training
(`docs/DATA_CARD.md` §4). Invalid or out-of-range input returns HTTP 422, not a silent
error; `term` is restricted to 36 (§4). A container (`Dockerfile`, lean
`requirements-api.txt`) serves the API, and a GitHub Actions workflow builds and
smoke-tests `/health`, a real `/score`, and the 422 path on every push (`docs/SETUP.md` §9).

**Drift monitoring.** `src/monitor.py` reuses the Phase-1 PSI engine (`src/psi.py`) to
compare an incoming batch against the "clean" training baseline (`era_pre_2012 == 0`) —
every new record is post-2012 by construction, so a raw baseline would inflate PSI on the
~26 rollout columns as a permanent artifact. It reports PSI per feature and for the output
score, banded <0.10 / 0.10–0.25 / >0.25, with a measured floor (`MIN_BATCH_FOR_PSI = 200`)
below which it returns `insufficient_sample` rather than sampling noise. It separates
genuine drift from artifact: `initial_list_status` (real policy shift, fractional → whole,
PSI ≈ 0.48) is annotated as a known cause, and `issue_d`/`earliest_cr_line` as definitional
artifacts of the split cut; only drift with no known cause is reported as
`drift_unexplained`.

**Retraining trigger.** Policy: retrain when the monitor reports `drift_unexplained`, or
critical-band PSI (>0.25) on features or the output score without a known benign cause,
confirmed on a batch at or above the sample floor — and gate any retrained model behind the
same profit-versus-baseline test (§8): it ships only if it beats the approve-all and
logistic baselines on held-out profit, not AUC. Recalibration was explored and rejected
(`docs/FACTS.md` §5), so the trigger targets refits. [TODO: automated retraining
*execution* is not implemented — the trigger is a defined policy, evaluated by running
`src/monitor.py` and reading the verdict; no scheduler or auto-retrain is wired up. No
production batch has been monitored, so the PSI figures here are from test vintages.]

## 11. Remaining next steps and footprint

Deliberately deferred (deployment and drift monitoring are done — §10): a separate
60-month scorecard; automated retraining execution with scheduling/alerting (§10 TODO);
and monitoring a real production batch to replace the test-vintage PSI figures.
Recalibration was explored and rejected (`docs/FACTS.md` §5).

**Compute footprint**: Trained on CPU in ~55s; no GPU required; carbon footprint
negligible (single-machine, minutes of total compute). Full pipeline reproduction
(cleaning, then split, features, train, and verify) measured at ~3.1 minutes. The
original hyperparameter search (walk-forward, ~90 fits) is not re-run by the
reproduction entry point (`docs/SETUP.md`).

## 12. Version history

- **2.0.0 — 2026-08-06** — Serving and drift monitoring implemented (§10); removed the
  "no API / no monitoring / no drift detection" claim and the "next step" framing of
  deployment. MAJOR bump: contradicts a prior claim readers relied on (model not deployed,
  unmonitored). Full changelog in `CHANGELOG.md`.
- **1.0.0** — Initial card: offline artifact, profit-based evaluation, temporal
  validation, documented bias; no serving or monitoring.

**Citation**: no formal paper or DOI exists for this project. If referencing this work,
cite the repository directly:

```bibtex
@misc{pavan2026creditdefault,
  author = {Pavan, Mateus Fardin},
  title  = {credit-default-prediction-lendingclub},
  year   = {2026},
  url    = {https://github.com/MateusFPavan/credit-default-prediction-lendingclub}
}
```
