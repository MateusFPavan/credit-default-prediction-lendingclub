---
model_name: credit-default-prediction-lendingclub / XGB_walkforward
task: binary-classification (credit default prediction)
library: scikit-learn, xgboost
language: en
license: MIT (see LICENSE at repository root)
data_license: CC0-1.0
---

# Model Card: XGB_walkforward (Lending Club Credit Default)

Related docs: data card at [`docs/DATA_CARD.md`](DATA_CARD.md); setup and reproduction at
[`docs/SETUP.md`](SETUP.md); full verified facts sheet at [`docs/FACTS.md`](FACTS.md).

## 1. Model summary

A gradient-boosted tree classifier (XGBoost, wrapped for interface consistency alongside
a scikit-learn `Pipeline` baseline) that scores probability of default on Lending Club
peer-to-peer personal loans, selected and evaluated to maximize **expected portfolio
profit** rather than accuracy or AUC.

## 2. Model details

- **Architecture**: `XGBClassifier` (gradient-boosted trees), used directly as the final
  estimator; a `LogisticRegression` inside a scikit-learn `Pipeline` (`StandardScaler` +
  `LogisticRegression`) serves as the baseline it is compared against.
- **Hyperparameters** (frozen): `max_depth=8`, `learning_rate=0.03`, `n_estimators=600`,
  `min_child_weight=10`, `subsample=0.8`, `colsample_bytree=0.6`, `random_state=42`,
  `n_jobs=1`, `eval_metric=logloss`. Operational decision threshold: **0.31**.
- **Features**: 79 named features, expanding to 90 columns after one-hot encoding of 5
  categorical fields. The training dataset (`train.parquet`) has 89 columns; 79 enter the
  model as features (5 `EVAL_ONLY`, 3 excluded by design, 1 target, 1 redundant column
  dropped).
- **Versions**: scikit-learn 1.9.0, xgboost 3.3.0, Python 3.12.10.
- **Trained**: 2026-07-11. Serialized at `models/xgb_final.joblib`. Hyperparameters,
  feature list, and a SHA256 of the exact `train.parquet` used are recorded in
  `models/model_meta.json`.
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
  validation, profit-based evaluation, documented bias). See `docs/DATA_CARD.md` §7 for
  the same framing on the data side.

## 4. Out-of-scope and misuse

- **Must not be used to score the rejected-applicant population.** The model estimates
  P(default | approved), never having seen a rejected application. See §9.
- **Not for live decisioning as-is.** It is packaged as a serializable, reproducible
  artifact, not a deployed service. There is no API, no monitoring, no drift detection
  (see §10, "next steps").
- **Degrades in the highest-risk segment.** AUC falls monotonically from grade A (0.648)
  to grade G (0.585), and is lowest for the lowest-income quartile (Q1, 0.648) versus Q4
  (0.697). It is least reliable exactly where a lender most needs it to be reliable (§8).
- **Not transferable to 60-month loans.** Applied without refitting to 60-month loans,
  performance degrades severely (AUC falls from 0.6846 to 0.6433, and the logistic
  baseline's profit gain turns negative). 36- and 60-month loans are structurally
  distinct risk pools, not the same population at a different maturity.
- **Probabilities are optimistic, not conservative.** The model systematically
  underestimates default (§8). Do not treat the raw score as a conservative lower bound.

## 5. How to use

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
validation 2014 (N=162,570), test 2015 (N=282,787), all 36-month loans. Hyperparameters
were selected by walk-forward validation across three expanding windows: train through
2011 and validate on 2012, train through 2012 and validate on 2013, and train through
2013 and validate on 2014. Each window optimized expected profit, not AUC, and no single
validation-year fit decided the choice. Reproducibility is bit-exact only under three
conditions: `random_state=42`, `n_jobs=1`, and training rows kept in `train.parquet`'s
on-disk order (XGBoost's histogram algorithm is not row-order invariant). Full
reproduction steps: `docs/SETUP.md`.

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
The model rejects 9.2% of eventual defaulters (equivalently, it approves the rest).

**Disaggregated by subgroup** (test): AUC declines monotonically by grade, from 0.648
(A) to 0.585 (G), and by income quartile is lowest for Q1 (0.648) and highest for Q4
(0.697). The model is least reliable exactly in the highest-risk, lowest-income segment.

**Calibration**: the model systematically **underestimates** default: observed default
exceeds predicted in every decile (e.g., decile 10: 31.80% observed vs. 30.86% predicted).
This is a concrete limitation, not a footnote: used as an absolute probability, the score
is optimistic, which matters for any threshold-based decision.

**Explainability (SHAP, 50k stratified test sample)**: top features by mean |SHAP| are
`fico_range_low`, `installment_to_income`, `annual_inc`, `acc_open_past_24mths`, `dti`.
`verification_status` shows a split effect: `source verified` preserves the univariate
default-rate inversion, while plain `verified` reverses it under multivariate control (a
Simpson's-paradox confound). `era_pre_2012` has zero SHAP on the test set (the flag is
always 0 outside the training population).

## 9. Bias, risks & limitations

- **Selection bias**: the model estimates P(default | approved), never having seen
  rejected applicants. It cannot say how it would perform as a first-pass underwriting
  filter, only as a second layer over an already-approved book (`docs/DATA_CARD.md`).
- **Subgroup reliability**: weakest exactly where risk is highest (grade and income, per
  §8), and the false-negative/false-positive cost asymmetry is large: a bad loan costs
  2.67x what a good loan returns at the median ($5,398.84 vs. $2,023.62), so subgroup
  weakness concentrates where errors are also most expensive.
- **Temporal drift risk**: hyperparameters and the threshold were fit on 2007-2014 data.
  No mechanism monitors whether 2015+ vintages continue to resemble that distribution.
- **Term non-transferability**: not valid for 60-month loans without a separate scorecard
  (§4).

## 10. Next steps (not gaps) and footprint

Named, deliberately deferred: deployment as a served API with drift monitoring; a
separate 60-month scorecard (the transfer analysis showed 36- and 60-month loans are
structurally distinct); recalibration was explored and rejected (it cost 42% of the
training data for no net profit gain, see `docs/FACTS.md` §5).

**Compute footprint**: Trained on CPU in ~55s; no GPU required; carbon footprint
negligible (single-machine, minutes of total compute). Full pipeline reproduction
(cleaning, then split, features, train, and verify) measured at ~3.1 minutes. The
original hyperparameter search (walk-forward, ~90 fits) is not re-run by the
reproduction entry point (`docs/SETUP.md`).

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
