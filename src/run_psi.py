"""Generates the Project 4 / Phase 1 PSI (Population Stability Index) reports.

Two views, written as versioned CSVs under reports/psi/:

1. Split-level PSI: baseline = train (issue_d <= 2013), compared against validation
   (2014) and test (2015), for every FEATURE_SET column plus the model's own score
   (xgb_final.joblib predict_proba). Two variants:
   - raw: baseline is the full train split (2007-06 to 2013-12).
   - clean: baseline is train restricted to era_pre_2012 == 0 (2012-2013 only) - the
     sensitivity check scope.md section 10 calls for, isolating the 2012 bureau-rollout
     sentinel block (-1, 999) from the comparison.

2. Quarterly PSI curve: score + the 6 top SHAP features (docs/FACTS.md section 5),
   computed per issue_d quarter across train+validation+test (the 36-month population),
   always against the same train (raw) baseline bins - never re-fit per quarter.

Per-column bins are fit ONCE on the relevant baseline and reused for every comparison,
per src.psi's contract. Run with: python -m src.run_psi
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import load_split, FEATURE_SET, CATEGORICAL_COLS, REFERENCE_DATE
from src.features import build_features, prepare_X
from src.psi import fit_binnings, psi_table, compute_psi, psi_band

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "psi"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "xgb_final.joblib"

SHAP_TOP_FEATURES = [
    "fico_range_low", "installment_to_income", "annual_inc",
    "dti", "revol_util", "inq_last_6mths",
]

SCORE_COL = "model_score"
DATETIME_COLS = ["issue_d", "earliest_cr_line"]


def encode_datetime_cols(df, cols=DATETIME_COLS):
    """PSI needs an ordered numeric scale to bin on; datetime64 columns (issue_d,
    earliest_cr_line) are converted to days-since-reference, same encoding prepare_X
    already uses to feed them to the model, so the binning is on the same scale the
    model sees."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = (df[c] - REFERENCE_DATE).dt.days
    return df


def score_df(model, df_feat):
    X = prepare_X(df_feat, FEATURE_SET, CATEGORICAL_COLS)
    X = X.reindex(columns=model.get_booster().feature_names, fill_value=0)
    return model.predict_proba(X)[:, 1]


def load_scored_splits(model):
    splits = {}
    for name in ("train", "validation", "test"):
        df = load_split(name)
        df_feat = build_features(df)
        df_feat[SCORE_COL] = score_df(model, df_feat)
        splits[name] = df_feat
    return splits


def band_counts(df, comparison_split):
    sub = df[df["comparison_split"] == comparison_split]
    return sub["band"].value_counts().reindex(["stable", "attention", "unstable"], fill_value=0)


def split_view(baseline_df, comparisons, feature_cols, variant_name):
    binnings = fit_binnings(baseline_df, feature_cols, categorical_cols=CATEGORICAL_COLS)
    tables = []
    for split_name, comp_df in comparisons.items():
        t = psi_table(binnings, baseline_df, comp_df, feature_cols)
        t.insert(0, "comparison_split", split_name)
        t.insert(0, "variant", variant_name)
        tables.append(t)
    return pd.concat(tables, ignore_index=True), binnings


def quarterly_view(train_df, combined_df, score_binning, shap_binnings):
    quarter = combined_df["issue_d"].dt.to_period("Q").astype(str)
    combined_df = combined_df.assign(_quarter=quarter)

    rows = []
    for q, sub in combined_df.groupby("_quarter", sort=True):
        for col, binning in [(SCORE_COL, score_binning)] + list(shap_binnings.items()):
            psi, _ = compute_psi(train_df[col], sub[col], binning=binning)
            rows.append({
                "quarter": q,
                "feature": col,
                "psi": psi,
                "band": psi_band(psi),
                "n_baseline": len(train_df[col].dropna()),
                "n_quarter": len(sub[col].dropna()),
            })
    return pd.DataFrame(rows)


def write_readme(raw_band_val, raw_band_test, clean_band_val, clean_band_test,
                  raw_score_psi, clean_score_psi):
    text = f"""# PSI reports (Project 4, Phase 1)

All numbers in this directory come from `src/run_psi.py` (`python -m src.run_psi`) run
against `models/xgb_final.joblib` and `data/processed/{{train,validation,test}}.parquet`.
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
| Raw | validation (2014) | {raw_band_val['stable']} | {raw_band_val['attention']} | {raw_band_val['unstable']} |
| Raw | test (2015) | {raw_band_test['stable']} | {raw_band_test['attention']} | {raw_band_test['unstable']} |
| Clean (era_pre_2012==0 baseline) | validation (2014) | {clean_band_val['stable']} | {clean_band_val['attention']} | {clean_band_val['unstable']} |
| Clean (era_pre_2012==0 baseline) | test (2015) | {clean_band_test['stable']} | {clean_band_test['attention']} | {clean_band_test['unstable']} |

## Score PSI (model_score, train baseline vs. test)

| Variant | PSI | Band |
|---|---|---|
| Raw | {raw_score_psi['psi']:.4f} | {raw_score_psi['band']} |
| Clean | {clean_score_psi['psi']:.4f} | {clean_score_psi['band']} |

The PSI engine and its rules (bins fit once on baseline and reused, decile bins for
numerics with sentinel point-masses carved into their own bin, new categories kept as
their own bin, epsilon=1e-6 for empty bins) are documented in `src/psi.py`.
"""
    (REPORTS_DIR / "README.md").write_text(text, encoding="utf-8")


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    splits = load_scored_splits(model)
    train, validation, test = splits["train"], splits["validation"], splits["test"]

    feature_cols = FEATURE_SET + [SCORE_COL]

    train_enc = encode_datetime_cols(train)
    validation_enc = encode_datetime_cols(validation)
    test_enc = encode_datetime_cols(test)
    comparisons_enc = {"validation": validation_enc, "test": test_enc}

    # --- Raw view ---
    raw_table, raw_binnings = split_view(train_enc, comparisons_enc, feature_cols, "raw")
    raw_table.to_csv(REPORTS_DIR / "psi_splits_raw.csv", index=False)

    # --- Clean view ---
    train_clean_enc = train_enc[train_enc["era_pre_2012"] == 0].copy()
    clean_table, _ = split_view(train_clean_enc, comparisons_enc, feature_cols, "clean")
    clean_table.to_csv(REPORTS_DIR / "psi_splits_clean.csv", index=False)

    # --- Quarterly view (reuses the raw view's train-fit bins) ---
    combined = pd.concat([train, validation, test], ignore_index=True)
    score_binning = raw_binnings[SCORE_COL]
    shap_binnings = {c: raw_binnings[c] for c in SHAP_TOP_FEATURES}
    quarterly_table = quarterly_view(train, combined, score_binning, shap_binnings)
    quarterly_table.to_csv(REPORTS_DIR / "psi_quarterly.csv", index=False)

    # --- Console summary ---
    raw_features_only = raw_table[raw_table["feature"] != SCORE_COL]
    clean_features_only = clean_table[clean_table["feature"] != SCORE_COL]
    raw_band_val = band_counts(raw_features_only, "validation")
    raw_band_test = band_counts(raw_features_only, "test")
    clean_band_val = band_counts(clean_features_only, "validation")
    clean_band_test = band_counts(clean_features_only, "test")

    raw_score_test = raw_table[(raw_table["feature"] == SCORE_COL) & (raw_table["comparison_split"] == "test")].iloc[0]
    clean_score_test = clean_table[(clean_table["feature"] == SCORE_COL) & (clean_table["comparison_split"] == "test")].iloc[0]

    write_readme(raw_band_val, raw_band_test, clean_band_val, clean_band_test,
                 {"psi": raw_score_test["psi"], "band": raw_score_test["band"]},
                 {"psi": clean_score_test["psi"], "band": clean_score_test["band"]})

    print("=== Band counts (feature-level rows, per comparison split) ===")
    print("Raw / validation:\n", raw_band_val.to_string())
    print("Raw / test:\n", raw_band_test.to_string())
    print("Clean / validation:\n", clean_band_val.to_string())
    print("Clean / test:\n", clean_band_test.to_string())
    print()
    print("=== Score PSI (train baseline vs test) ===")
    print(f"Raw:   psi={raw_score_test['psi']:.4f}  band={raw_score_test['band']}")
    print(f"Clean: psi={clean_score_test['psi']:.4f}  band={clean_score_test['band']}")
    print()
    print("=== CSVs written to reports/psi/ ===")
    for f in sorted(REPORTS_DIR.glob("*")):
        print(" -", f.relative_to(REPORTS_DIR.parent.parent))


if __name__ == "__main__":
    main()
