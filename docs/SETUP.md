# Setup & Reproducibility Guide

Follows the spirit of the NeurIPS ML Code Completeness Checklist: dependencies,
exact commands, expected results, and honest scope boundaries.

## 1. Prerequisites

- **OS**: developed on Windows; commands below are given for both bash (Git Bash/WSL/Linux/macOS)
  and Windows PowerShell where they differ.
- **Python 3.12.10** in a virtual environment at `.venv/`.
- **CPU only** — no GPU is used or required anywhere in this project.
- **A free Kaggle account** — required once, to download the raw dataset manually (§3).
  No other account or credential is needed.

## 2. Dependencies

`requirements.txt` at the repo root is a full pinned `pip freeze` (124 packages, produced
in this environment). Top-level libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`,
`imbalanced-learn`, `matplotlib`, `seaborn`, `jupyter`, `pyarrow`, `statsmodels`, `shap`.

**bash**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**PowerShell**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The notebook steps in `run_all.py` execute via `nbconvert` against a named Jupyter
kernel. Register it once, after installing dependencies (works identically in bash and
PowerShell):
```bash
python -m ipykernel install --user --name=credit-default-prediction-lendingclub
```

## 3. Data access

The raw file, `accepted_2007_to_2018Q4.csv` (~1.67 GB), is **not** in this repository and
is **not** downloaded automatically. Get it manually:

1. Create/sign in to a free Kaggle account.
2. Download the dataset **wordsforthewise/lending-club** (CC0 license).
3. Place the CSV at `data/raw/accepted_2007_to_2018Q4.csv` (relative to the repo root).

This step requires your own Kaggle credentials, which is why it is manual — no credential
is ever read, stored, or committed by this project.

A 1,000-row sample of the *processed* (cleaned) dataset is versioned at
`data/processed/loans_clean_sample.csv`, so you can inspect the schema and a few rows
without running anything. The full processed parquet files (`train.parquet`,
`validation.parquet`, `test.parquet`, `transfer_60m.parquet`, `loans_clean.parquet`) are
gitignored — `run_all.py` regenerates them from the raw CSV.

## 4. Environment & configuration

No environment variables, config files, or secrets are needed anywhere in this project.
Activate `.venv` (§2) before running anything; that is the only environment state
required. Nothing sensitive is ever committed — see §8 for what to do if you don't have
the raw CSV.

## 5. How to reproduce

With `.venv` active and the raw CSV in place (§3), run the single entry point from the
repo root:

```bash
python run_all.py
```

This runs the **essential path only**, in this fixed order:

1. Checks `data/raw/accepted_2007_to_2018Q4.csv` exists; aborts with a clear message
   (pointing back to §3) if it doesn't.
2. Notebook `03_build_processed.ipynb` — cleans the raw CSV into `loans_clean.parquet`.
3. Notebook `04_temporal_split.ipynb` — splits by `issue_d` into train/validation/test/transfer.
4. Feature engineering — calls `src/features.py`'s `build_features` directly (not via
   notebook 05, for speed) and re-saves the four splits.
5. Notebook `14_train_final_model.ipynb` — trains the frozen final XGBoost model on
   `train.parquet` and serializes it to `models/`.
6. `src/verify_pipeline.py` — an independent, `src/`-only check that retrains the model
   and asserts the test-set profit reproduces **exactly**; exits with an error if it
   doesn't.

**Reproducibility boundary — read before running anything else.** `run_all.py`
reproduces the *final, already-selected* model and its test result. It deliberately does
**not** re-run the model-*selection* experiments — baseline comparison, walk-forward
hyperparameter tuning, and bootstrap validation (notebooks 06–11). Those take hours via
`nbconvert` on this hardware and are not needed to reproduce the delivered model; they
are documented for reading in the notebooks themselves and in `docs/FACTS.md`, not
wired into automated re-execution. This is a scope decision, not a missing feature.

**The temporal ordering is load-bearing.** Splits are strictly by `issue_d` — train
≤2013, validation 2014, test 2015 (plus a 60-month transfer set) — never a random split.
Re-splitting randomly would leak future information into training and would not
reproduce any number in this document or in `docs/FACTS.md`.

## 6. Expected results, seed, and determinism

The business metric is **expected portfolio profit** on the held-out 2015 test split, not
accuracy. Final model (XGBoost, walk-forward-tuned, operating threshold 0.31):

| Model | Test profit |
|---|---|
| Approve-all baseline | $233,202,813.06 |
| **Final model** | **$242,230,710.89** |

`src/verify_pipeline.py` asserts this exact figure (to the cent) — a single run, no
repeated sampling required to reproduce it.

`random_state=42` is used throughout. The final XGBoost model is bit-for-bit
deterministic **only** under three conditions, all already enforced in `src/models.py`
and `run_all.py`:
1. `random_state=42` on the estimator;
2. `n_jobs=1` (multi-threaded histogram building is not provably reproducible run-to-run
   on this hardware);
3. training rows kept in `train.parquet`'s on-disk order — XGBoost's histogram splits are
   measurably sensitive to row order, even with the same seed.

`models/model_meta.json` records the SHA256 of the exact `train.parquet` used to fit the
shipped model, for traceability.

## 7. Runtime & resources

Measured on the development machine (CPU only, single run):

| Step | Time |
|---|---|
| Build processed dataset | 75.5s |
| Temporal split | 13.8s |
| Feature engineering | 3.7s |
| Train final model | 55.1s |
| Verify | 39.4s |
| **Total** | **187.6s (~3.1 min)** |

No GPU, no distributed setup, no special hardware.

## 8. Troubleshooting

- **"arquivo de dados bruto nao encontrado"** — the raw CSV isn't at
  `data/raw/accepted_2007_to_2018Q4.csv`; see §3.
- **Test profit doesn't match exactly** — check that XGBoost is running with `n_jobs=1`
  and that no step reordered `train.parquet` (e.g., sorted by `issue_d`) before fitting;
  either changes the result (§6).
- **nbconvert can't find the kernel** — register it (§2):
  `python -m ipykernel install --user --name=credit-default-prediction-lendingclub`.
- **Wanting to reproduce notebooks 06–11 (tuning/bootstrap) too** — expect hours, not
  minutes; run each via `jupyter nbconvert --execute --inplace` individually rather than
  through `run_all.py`, which does not include them by design (§5).
- **No Makefile, Docker, or CI** — `make` was unavailable on the development machine, so
  none was created; there is no containerized or CI-driven reproduction path at this
  time. [TODO: add if a reviewer requires one.]

---

**Self-check**: pinned deps + install ✓ (§2), data access ✓ (§3), exact reproduce
commands ✓ (§5), expected results/seeds ✓ (§6), relative paths only ✓ (throughout — no
absolute path appears anywhere in this guide or in `run_all.py`), no committed secrets ✓
(§3, §4).
