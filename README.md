# Credit Default Prediction: Lending Club

[![CI](https://github.com/MateusFPavan/credit-default-prediction-lendingclub/actions/workflows/docker.yml/badge.svg)](https://github.com/MateusFPavan/credit-default-prediction-lendingclub/actions/workflows/docker.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

Turning loan approvals into a profit decision: a credit-default model for real
peer-to-peer consumer loans, selected and evaluated by expected portfolio profit, not
accuracy.

![Portfolio profit vs. approval threshold, 2015 test set](reports/figures/hero.png)

## Headline Result

**On the 2015 held-out test set, the model rejects 3.8% of loan applications, avoiding
$32.2M in losses at a cost of $23.1M in forgone interest.** The result is a net gain of
$9.0M over an approve-everyone policy (95% CI $7.6M-$10.7M). Over a logistic-regression
baseline, the gain is +$6.3M. The total portfolio profit under the model's policy is
$242.23M. That figure already includes the $233.2M an approve-all policy would produce
with no modeling effort. The model's own attributable contribution is the $9.0M delta,
not the gross total.

**Key insight**: the model was chosen by expected profit, not accuracy, and that choice
mattered. The winning model ties a simpler logistic-regression baseline on AUC (0.68) but
wins decisively on profit. A standard accuracy/AUC contest would have called these two
models a coin flip and missed $6.3M of real, measurable value.

Rejecting just the riskiest 10% of applicants avoids ~21% of all defaults, twice as
effective as a random cut of the same size. Trained and evaluated on ~673K matured
36-month Lending Club loans (2007-2015, 14.8% default rate).

**Honest caveat, stated up front, not buried**: the model is least reliable in the
highest-risk, lowest-income segment. It was trained only on approved loans, so it cannot
score rejected applicants (a selection-bias limit). It is also not built for live lending
decisions as-is.

## Interactive Dashboard

**[▶ Explore the live dashboard](https://mateusfpavan.github.io/credit-default-prediction-lendingclub/)** — a scrollable walkthrough from the $9M headline through model discrimination, the profit decomposition, an interactive threshold slider (recomputes the confusion matrix and portfolio profit live on the 2015 test set), PSI stability with a raw-vs-clean sentinel-artifact contrast, and honest subgroup limitations.

[![Credit risk dashboard](dashboard/screenshots/01_performance.png)](https://mateusfpavan.github.io/credit-default-prediction-lendingclub/)

A Power BI version of the same analysis (`.pbix` + screenshots) lives in [`dashboard/`](dashboard/).

## Key Results

| Model | Test profit (2015) | 95% CI |
|---|---|---|
| Approve-all baseline | $233.2M | — |
| Logistic-regression baseline | $235.9M | — |
| **XGBoost, walk-forward-tuned (this model)** | **$242.2M** | **[$237.9M, $246.7M]** |

Rejecting the riskiest applicants captures defaults far faster than a random cut, and the
$9.0M net gain decomposes cleanly into avoided losses minus forgone interest. The number
is not inflated by an aggressive rejection policy (96.2% of applications are still
approved):

![Rejecting the riskiest applicants captures defaults faster than random](reports/figures/lift_curve.png)

![Where the $9.0M net gain comes from](reports/figures/profit_decomposition.png)

## Methodology, in Brief

- **Decision metric is expected portfolio profit**, not accuracy: `profit = interest on
  approved good loans − lost principal on approved bad loans`. A bad loan costs 2.67x
  what a good loan returns at the median. Treating every error equally, as accuracy does,
  misrepresents the actual economics of the decision.
- **Validation is temporal / walk-forward, never random**: train ≤2013, validate 2014,
  test 2015. Hyperparameters were selected across three expanding time windows: train
  through 2011 and validate on 2012, train through 2012 and validate on 2013, and train
  through 2013 and validate on 2014. Each window optimized profit, not a single
  validation year, and never AUC.
- **Missing data was resolved by mechanism**, not blanket imputation: informative
  absence (MNAR), staged bureau-data rollouts, and sparse-but-informative nulls each got
  a different, evidence-based treatment.
- **Leakage was screened on three fronts**: temporal (never shuffled), target
  (post-origination columns dropped, confirmed by univariate AUC), and identity
  (borrower ID is 100% null, so group-level splitting is not possible; this is stated as
  a limitation).
- **Two engineered features were tried and dropped** (redundant FICO average; a
  bankcard-utilization ratio undefined for 30% of borrowers). This is reported as a
  strength of the process, not something omitted.
- **Calibration was checked, and a recalibration attempt was rejected**: it improved a
  reliability metric but cost 42% of training data for no net profit gain.

Full methodology and every underlying number: [`docs/technical_report.md`](docs/technical_report.md).

## Serving & Monitoring

The model is served and monitored, not left as a notebook artifact. This is a reproducible,
CI-tested inference stack, not a hosted service under SLA — the selection-bias limit below
still governs what the scores may be used for.

- **Inference API** ([`src/api.py`](src/api.py)): a FastAPI service. `POST /score` takes a
  raw loan application, applies the *exact* training-time cleaning and encoding (reused via
  `src/scoring.py` → `src/cleaning.py` → `src/features.py`, never reimplemented), and returns
  a default probability plus an approve/reject decision at the 0.31 profit threshold.
  `GET /health` reports readiness. Out-of-range or missing input returns HTTP 422 rather
  than a silent wrong score, and `term` is restricted to 36 (the model is not valid for
  60-month loans).
- **Drift monitoring** ([`src/monitor.py`](src/monitor.py)): reuses the project's PSI engine
  (`src/psi.py`) to compare an incoming batch against the training baseline. It separates
  *genuine* drift (a real platform policy shift in `initial_list_status`, PSI ≈ 0.48) from
  *artifacts* (the definitional `issue_d` split cut), and refuses to score batches below a
  measured sample floor where PSI is just noise.
- **Containerized + CI**: a [`Dockerfile`](Dockerfile) serves the API from a lean,
  serving-only dependency set; a [GitHub Actions workflow](.github/workflows/docker.yml)
  builds the image and smoke-tests `/health`, a real `/score`, and the 422 path on every
  push.

Serving, monitoring, and the retraining trigger are documented in
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) §10.

## Limitations

The model is not uniformly reliable, and that is reported directly rather than
smoothed over in an aggregate metric:

![Model discriminates worst in the highest-risk segments](reports/figures/subgroup_auc.png)

- **Weakest exactly where risk is highest**: AUC falls from 0.648 (grade A) to 0.585
  (grade G), and from 0.697 (highest income quartile) to 0.648 (lowest). That is the
  reverse of where a lender would most want precision.
- **Selection bias**: the model estimates P(default | approved), having never seen a
  rejected application. It cannot say how it would perform as a first-pass underwriting
  filter, only as a second layer over an already-approved loan book.
- **Not transferable to 60-month loans** without a dedicated scorecard. Applied without
  refitting, performance degrades severely, which is evidence that 36- and 60-month
  loans are structurally distinct risk populations.
- **Optimistic calibration**: observed default exceeds predicted in every decile, so the
  raw score should not be treated as a conservative probability. A recalibration attempt
  was tested and rejected (it cost 42% of training data for no net profit gain).

Remaining next steps (deliberate, not gaps): a dedicated 60-month scorecard, automated
retraining execution with alerting, and monitoring against a real production batch.

Full disaggregated results, calibration analysis, and SHAP explainability:
[`docs/technical_report.md`](docs/technical_report.md) §8.

## The Model

XGBoost wrapped for interface consistency alongside a scikit-learn `Pipeline` logistic
baseline. 79 named features, expanding to 90 columns after one-hot encoding. Serialized
with `joblib` at `models/xgb_final.joblib`. Hyperparameters, feature list, and a SHA256 of
the exact training data are recorded in `models/model_meta.json`. Reproducible end to end
via `python run_all.py` (~3 minutes, CPU only). Bit-exact determinism requires three
conditions: `random_state=42`, `n_jobs=1`, and training rows kept in their original
on-disk order (XGBoost's histogram algorithm is not row-order invariant).

Full specification: [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

## Reproducing This Project

```bash
git clone https://github.com/MateusFPavan/credit-default-prediction-lendingclub.git
cd credit-default-prediction-lendingclub
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# register the Jupyter kernel used by the pipeline notebooks — see docs/SETUP.md
```

Download the raw dataset manually (Kaggle `wordsforthewise/lending-club`, CC0 license,
requires a free Kaggle account and is not auto-downloaded) and place it at
`data/raw/accepted_2007_to_2018Q4.csv`. Then:

```bash
python run_all.py
```

This reproduces the final model and its exact test result in about 3 minutes on CPU: raw
CSV becomes a cleaned dataset, then a temporal split, then features, then a trained
model, then a verified profit figure. **It deliberately does not re-run the
model-*selection* experiments** (baseline comparison, walk-forward hyperparameter tuning,
bootstrap validation, in `notebooks/06` through `11`): those take hours and are not
needed to reproduce the delivered model. They are fully documented, not hidden. See
`docs/FACTS.md` and the notebooks themselves.

To serve the model behind the API (optional):

```bash
pip install -r requirements-api.txt
uvicorn src.api:app --reload          # API at http://localhost:8000, contract at /docs
# or, containerized:
docker build -t credit-default-api . && docker run -p 8000:8000 credit-default-api
```

Full setup and troubleshooting: [`docs/SETUP.md`](docs/SETUP.md).

## Repository Structure

```
data/            raw CSV (gitignored) and processed parquets (gitignored, sample versioned)
notebooks/       01-15 working notebooks (full process) + 1.0-7.0 narrated notebooks (presentation)
src/             data · features · economics · models · psi · scoring · cleaning · api · monitor · run/verify scripts
models/          xgb_final.joblib (versioned) + model_meta.json; logistic_baseline.joblib (gitignored)
reports/figures/ business-impact figures
docs/            technical report, data card, model card, setup guide, facts sheet
dashboard/       Power BI (.pbix) + theme + screenshots; interactive web dashboard in docs/index.html
references/      one-page recruiter case studies (EN and pt-BR)
Dockerfile       containerized inference API
.github/         GitHub Actions CI (build + smoke-test the container)
```

## Documentation Index

| Document | Purpose |
|---|---|
| [`docs/technical_report.md`](docs/technical_report.md) | Full methodology and results |
| [`docs/FACTS.md`](docs/FACTS.md) | Canonical, verified facts sheet, the single source of truth for every number |
| [`docs/DATA_CARD.md`](docs/DATA_CARD.md) | Dataset datasheet (provenance, license, missing-data mechanisms) |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | Model specification, training procedure, evaluation, serving |
| [`docs/SETUP.md`](docs/SETUP.md) | Environment setup and reproduction, step by step |
| [`CHANGELOG.md`](CHANGELOG.md) | Versioned change history (Keep a Changelog / SemVer) |
| [`references/one_pager.md`](references/one_pager.md) | One-page recruiter case study (EN) |
| [`references/one_pager.pt-br.md`](references/one_pager.pt-br.md) | One-page recruiter case study (pt-BR) |
| `notebooks/` | Full working process (`01`-`15`) and a narrated walkthrough (`1.0`-`7.0`) |

## Stack

Python · pandas · scikit-learn · XGBoost · SHAP · matplotlib · FastAPI · Docker · GitHub Actions

## License & Contact

MIT. See [`LICENSE`](LICENSE). **Author**: Mateus Fardin Pavan. **Repository**:
<https://github.com/MateusFPavan/credit-default-prediction-lendingclub>. **Contact**:
<https://www.linkedin.com/in/mateus-fardin-pavan/>.
