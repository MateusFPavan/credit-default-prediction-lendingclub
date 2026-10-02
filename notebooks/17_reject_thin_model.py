# ============================================================================
# Phase 2a - Reject inference: thin model on shared features + parcelling sweep
# ============================================================================
# Purpose
#   Fit a "thin" model on the only features that approved and rejected applications
#   share, and use it to infer labels for the rejected applications by parcelling.
#   Reject labels are unknown by construction, so the inferred bad rate is reported as a
#   two-way sensitivity sweep over the parcelling assumptions (n_bands x base_mult), not
#   as a single estimate.
#
# Inputs
#   approved : src.data.load_split("train") -> loan_amnt, dti, emp_length_anos
#              (emp_length_anos already uses the -1 sentinel for missing, notebook 03).
#   rejected : data/processed/reject/rejected.parquet/app_year=*/*.parquet (Phase 1),
#              read with DuckDB (hive_partitioning=true) -> amount_requested, dti,
#              emp_length_raw (text such as "10+ years", mapped here with EMP_MAP to the
#              same convention notebook 03 uses for the approved loans).
#   src.economics (profit function) is not used here: this script fits the thin model
#   and runs the parcelling sweep. Profit-based evaluation is in
#   notebooks/18_profit_evaluation.py; CI-EX and Kickout are outside this script.
#
# Method decisions (see docs/reject_inference.md)
#   - Thin model features: amount, dti, emp_length. Rejected applications have only 4
#     usable fields, so the shared features are the only possible basis. The 4th field,
#     state, is deferred to a second experiment on geography (diagnostic in
#     notebooks/scratch/scratch_diag_state.py: 50 states + DC, 22 nulls in 27.6M rows,
#     no territories or invalid codes).
#   - Thin model = Logistic Regression: interpretable, and XGBoost has little to gain
#     with 3 features.
#   - dti of rejected applications is treated by mechanism, not by blanket exclusion
#     (diagnostic in notebooks/scratch/scratch_diag_dti_reject.py and
#     notebooks/scratch/scratch_diag_dti_parte2.py):
#       * dti == -1 ('-1%'): MNAR sentinel (same convention as -1 in emp_length_anos)
#         -> flag dti_missing, value imputed with the median of the valid values.
#       * dti == 100 ('100%'): right-censoring (>=100%; the spike is ~170x the nearest
#         neighbouring value) -> flag dti_censored, value kept at 100.
#       * dti in {9999, 99999, 199998}: redundant sentinel (0.3% of rows, already
#         covered by -1) -> rows dropped.
#       * real tail 100-1000 (excluding the spikes): genuine extreme values -> kept.
#     dti of approved loans needs no treatment (no sentinels, p99 ~33).
#   - The flags dti_missing, dti_censored and emp_length_missing are not model features.
#     They are always 0 among approved loans, so they have zero variance in the training
#     set and Logistic Regression cannot learn a weight for them (the coefficient was
#     measured at exactly 0.0). Information that exists only among rejects cannot be
#     learned by a model trained only on approvals. The flags stay in the rejected
#     dataframe as metadata (rej_flags) that records the mechanism; the signal itself
#     already enters the model through the treated value.
#   - emp_length of rejected applications follows the convention already used for
#     emp_length_anos on approved loans (train.parquet: scale 0-10, sentinel -1,
#     companion flag emp_length_missing, 0 mismatches between the two), via EMP_MAP:
#       * '< 1 year' -> 0, kept as a real signal, not treated as missing. The
#         Philadelphia Fed study (Jagtiani & Lam) found employment length to be the most
#         important factor in Lending Club rejections (~88% relative importance vs ~6%
#         amount, ~5% dti), so rejects are expected to be dominated by short employment:
#         it is the causal mechanism of rejection, not a form anomaly. Unlike dti, where
#         '-1%' and '9999%' were direct evidence of a sentinel, there is no distinct
#         sentinel marker for this value.
#       * '10+ years' -> 10: same coding as the approved loans, with no separate
#         censoring flag. The convention already exists in the project, and a new flag
#         would break feature symmetry with the approved loans.
#       * None -> -1 + flag emp_length_missing (project MNAR convention).
#   - amount has no exception mechanism (diagnostic in
#     notebooks/scratch/scratch_diag_amount_emplen.py: no sentinel, no censoring spike,
#     a normal distribution of human round numbers). Only a simple filter is applied:
#     amount <= 0 is logically impossible (1,288 rows, 0.005%).
#   - Parcelling uses a single bad-rate multiplier, base_mult in {1.0, 1.5, 2.0, 2.5,
#     3.0}, calibrated from the ratio between observed bad rates instead of a blind
#     sweep over the generic 1-5 range (the literature suggests 2-5x). SAS Enterprise
#     Miner calibrates the factor from the data (approved bad rate 8% -> rejected 13%,
#     factor ~1.5). Here the approved bad rate is 12.43%, so the plausible multiplier
#     range is ~1.5-2.5 (rejected bad rate 18-31%). Without real reject labels the
#     "right" factor cannot be calibrated (unlike Hugo Lopes 2018, who had labels and
#     chose by AUC), so the sweep shows sensitivity around the plausible range. Values
#     4.0 and 5.0 saturate: the censored group's bad rate reaches 1.0 and the overall
#     inferred bad rate reaches ~70% of the whole rejected population, which is
#     implausible (the multiplier is applied uniformly per band, so at high values it
#     hits the 1.0 cap even in the good bands). They are recorded in
#     docs/reject_inference.md as tested and saturating, and are not re-run here.
#   - An additional multiplier for dti_censored rows (censored_extra) was tested and
#     removed. It passed the bad-rate test (varying it changes the censored group's bad
#     rate), but in the profit comparison at a fixed acceptance rate
#     (notebooks/18_profit_evaluation.py) the difference with vs without censored_extra
#     was 0.00 in all 24 combinations tested (2 base_mult x 3 LGD x 4 acceptance rates).
#     Cause: a fixed acceptance rate accepts the lowest-PD loans, and censored rows
#     (dti>=100%) are already pushed to the worst risk by the value dti=100 plus
#     base_mult, so they never enter the accepted group. The extra multiplier only acts
#     on loans that are rejected anyway; the censoring signal is already in the value.
#     The flags therefore remain metadata only and none of them is a parcelling
#     multiplier.
#   - 4-fold stratified CV on the approved loans (canonical setup in Kozodoi/Lessmann,
#     arXiv 1909.06108 and 2407.13009) gives the out-of-sample AUC of the thin model:
#     each fold is fitted on the training folds and scored on the validation fold. The
#     final model that scores the rejects is fitted on all approved loans (production
#     model; the rejects have no folds), and the CV AUC is reported separately.
#   - Baseline 'ignore rejects' (baseline_ignore_rejects()): the thin model trained only
#     on approved loans, with no reject information. It is the first benchmark in
#     reject-inference papers and is compared by profit in
#     notebooks/18_profit_evaluation.py.
#
# Known limitation
#   The parcelling bands (parcelling_labels, below) are still built from
#   model.predict_proba(X_appr) with the final model, fitted on all approved loans. This
#   is the same in-sample bias that CV removes from the AUC, but not from the bands. The
#   out-of-sample measure of the thin model's predictive power is the CV AUC; the
#   parcelling bands remain in-sample until CV is integrated into band construction.

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_split

REJECT_GLOB = "data/processed/reject/rejected.parquet/app_year=*/*.parquet"

SHARED = ["amount", "dti", "emp_length"]  # thin model features (flags are metadata only)


EMP_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}


def treat_emp_length_rejected(df, raw_col="emp_length_raw", out_col="emp_length"):
    """
    Convert emp_length_raw (text) to a numeric 0-10 scale, following the same convention
    already used for emp_length_anos on the approved loans:
      - '< 1 year' -> 0 (real signal, kept; see the rationale in the file header).
      - '1'..'9 years' -> 1..9.
      - '10+ years' -> 10 (no censoring flag; same coding as the approved loans).
      - None/missing -> -1 + flag emp_length_missing (project MNAR convention).
    """
    s = df[raw_col]
    mapped = s.map(EMP_MAP)
    is_missing = mapped.isna()  # None or any value outside the map
    counts = {
        "lt1_year_0": int((mapped == 0).sum()),
        "10plus_10": int((mapped == 10).sum()),
        "missing_-1": int(is_missing.sum()),
        "total": len(df),
    }
    df[out_col] = mapped.fillna(-1).astype(int)
    df["emp_length_missing"] = is_missing.astype(int)
    print("[TREATMENT emp_length - rejected]")
    for k, v in counts.items():
        print(f"  {k:<14}: {v:,}")
    unmapped = df.loc[is_missing & s.notna(), raw_col].value_counts().head()
    if len(unmapped):
        print("  [WARNING] unmapped text values (check):")
        print(unmapped)
    return df, counts


def treat_dti_rejected(df, col="dti"):
    """
    Treat dti of the rejected applications by mechanism (diagnostic in
    notebooks/scratch/scratch_diag_dti_reject.py and
    notebooks/scratch/scratch_diag_dti_parte2.py):
      - dti == -1  -> missing sentinel ('-1%', same convention as emp_length_anos):
                      flag dti_missing=1, value imputed (median of the valid values).
      - dti == 9999/99999/199998 -> redundant sentinel (0.3%, already covered by -1
                      via dti_missing): rows dropped.
      - dti == 100 -> right-censoring ('100%', spike of ~170x its neighbourhood):
                      flag dti_censored=1, value kept at 100 (lower bound).
      - everything else (incl. the real 100-1000 tail) -> kept as is.
    Returns the treated df (with dti_missing/dti_censored) + a dict of counts.
    """
    n0 = len(df)
    counts = {}
    redund = df[col].isin([9999.0, 99999.0, 199998.0])
    counts["descartados_9999"] = int(redund.sum())
    df = df[~redund].copy()
    is_missing = (df[col] == -1)
    counts["missing_flag_-1"] = int(is_missing.sum())
    df["dti_missing"] = is_missing.astype(int)
    is_censored = (df[col] == 100)
    counts["censored_flag_100"] = int(is_censored.sum())
    df["dti_censored"] = is_censored.astype(int)
    valid = df.loc[(df[col] > 0) & (df[col] < 100), col]
    median_valid = float(valid.median())
    counts["mediana_imputada"] = median_valid
    df.loc[is_missing, col] = median_valid
    counts["restaram"] = len(df)
    counts["removidas_total"] = n0 - len(df)
    print("[TREATMENT dti - rejected]")
    for k, v in counts.items():
        vv = f"{v:,}" if isinstance(v, int) else f"{v:.2f}"
        print(f"  {k:<20}: {vv}")
    return df, counts


def treat_amount_rejected(df, col="amount"):
    """
    amount of the rejected applications has no exception mechanism (diagnostic in
    notebooks/scratch/scratch_diag_amount_emplen.py: no sentinel, no censoring, a normal
    distribution of human round numbers). Only a simple filter: amount<=0 is logically
    impossible.
    """
    n0 = len(df)
    df = df[df[col] > 0].copy()
    print(f"[TREATMENT amount] removed {n0 - len(df):,} rows (amount<=0); "
          f"remaining {len(df):,}")
    return df


def load_approved_shared():
    """Approved loans (split 'train', 172,988 rows, through 2013) -> X[SHARED], y, issue_d.
    dti of the approved loans is already clean (no sentinel, p99~33) and is not treated."""
    df_raw = load_split("train")
    issue_d_raw = df_raw["issue_d"]
    df = pd.DataFrame({
        "amount": df_raw["loan_amnt"].astype(float),
        "dti": df_raw["dti"].astype(float),
        "emp_length": df_raw["emp_length_anos"].astype(float),  # -1 sentinel set in notebook 03
        "target": df_raw["target"].astype(int),
    })
    X = df[SHARED]
    y = df["target"].to_numpy()
    issue_d = issue_d_raw
    return X, y, issue_d


def load_rejected_shared():
    """Rejected applications (partitioned Parquet from Phase 1, 27,648,741 rows) ->
    X[SHARED], rej_flags. amount filtered (amount<=0); dti treated by mechanism;
    emp_length mapped with the approved-loan convention. dti_missing/dti_censored/
    emp_length_missing are not part of X: they are returned as rej_flags, metadata that
    is neither a model feature nor a parcelling multiplier."""
    con = duckdb.connect()
    rel = f"read_parquet('{REJECT_GLOB}', hive_partitioning=true)"
    df_raw = con.execute(
        f"SELECT amount_requested, dti, emp_length_raw FROM {rel}"
    ).fetchdf()
    df = pd.DataFrame({
        "amount": df_raw["amount_requested"].astype(float),
        "dti": df_raw["dti"].astype(float),
        "emp_length_raw": df_raw["emp_length_raw"],
    })
    df = treat_amount_rejected(df)
    df, dti_counts = treat_dti_rejected(df)
    df, emp_counts = treat_emp_length_rejected(df)
    X = df[SHARED]
    rej_flags = df[["dti_missing", "dti_censored", "emp_length_missing"]].reset_index(drop=True)
    X = X.reset_index(drop=True)
    return X, rej_flags, {"dti": dti_counts, "emp_length": emp_counts}


# --- Thin model: logistic regression on the shared features -----------------
def train_thin_model(X, y):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(X, y)
    return model


def thin_model_cv(X_appr, y_appr, n_splits=4, seed=42):
    """
    4-fold stratified CV on the approved loans (canonical setup, Kozodoi). In each fold
    the model is fitted on the training folds and scored (AUC) on the validation fold,
    which removes the in-sample optimism of fitting and evaluating on the same X_appr.
    Returns the final model fitted on all approved loans (used to score the rejects,
    which have no folds) and the array of out-of-sample fold AUCs (mean/std printed).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr_idx, va_idx in skf.split(X_appr, y_appr):
        m = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=1000, random_state=seed))])
        m.fit(X_appr.iloc[tr_idx], y_appr[tr_idx])
        p_va = m.predict_proba(X_appr.iloc[va_idx])[:, 1]
        aucs.append(roc_auc_score(y_appr[va_idx], p_va))
    aucs = np.array(aucs)
    print(f"[THIN MODEL CV] AUC out-of-sample: {aucs.mean():.4f} +/- {aucs.std():.4f} "
          f"(folds: {np.round(aucs, 4)})")
    final = train_thin_model(X_appr, y_appr)  # production model: fitted on all approved loans
    return final, aucs


def baseline_ignore_rejects(X_appr, y_appr):
    """
    Formal 'ignore rejects' baseline: the thin model trained only on approved loans,
    with no inference on rejected applications. It is the reference against which
    reject-inference strategies such as parcelling and CI-EX are compared by profit at a
    fixed acceptance rate (notebooks/18_profit_evaluation.py), and the first benchmark
    in reject-inference papers (Kozodoi).
    """
    return train_thin_model(X_appr, y_appr)


# --- Parcelling (single multiplier; censored_extra removed, no effect on profit) ---
def parcelling_labels(model, X_appr, y_appr, X_rej, n_bands, base_mult):
    """
    Parcelling with a single bad-rate multiplier, applied uniformly per band.
    - base_mult: swept over 1.0-3.0, the plausible range for this dataset, calibrated
      from the ratio between rejected and approved bad rates rather than a blind guess
      over the generic 1-5 range from the literature. See docs/reject_inference.md,
      Phase 2a, multiplier calibration. 4.0/5.0 were tested: they saturate and produce
      an overall bad rate of ~70% (implausible), and are kept only as a record.
    No censored_extra multiplier (removed: no effect on profit; see the file header).
    """
    p_appr = model.predict_proba(X_appr)[:, 1]
    p_rej = model.predict_proba(X_rej)[:, 1]
    edges = np.quantile(p_appr, np.linspace(0, 1, n_bands + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    appr_band = np.digitize(p_appr, edges[1:-1])
    rej_band = np.digitize(p_rej, edges[1:-1])

    band_bad = {}
    for b in range(n_bands):
        m = appr_band == b
        band_bad[b] = y_appr[m].mean() if m.sum() > 0 else y_appr.mean()

    rng = np.random.default_rng(42)
    inferred = np.array([
        int(rng.random() < min(band_bad[b] * base_mult, 1.0)) for b in rej_band
    ])
    return inferred, band_bad


# --- Two-way sweep: bands x base_mult --------------------------------------------
def sweep(model, X_appr, y_appr, X_rej,
          bands_list=(5, 10, 20), base_list=(1.0, 1.5, 2.0, 2.5, 3.0)):
    print(f"{'bands':>7}{'base':>6}{'inferred_bad_rate':>20}")
    results = {}
    for nb in bands_list:
        for base in base_list:
            inferred, _ = parcelling_labels(model, X_appr, y_appr, X_rej, nb, base)
            rate = float(inferred.mean())
            results[(nb, base)] = rate
            print(f"{nb:>7}{base:>6}{rate:>20.4f}")
    vals = list(results.values())
    print(f"\n[STABILITY] inferred bad rate ranges from {min(vals):.4f} to {max(vals):.4f} "
          f"(spread {max(vals)-min(vals):.4f})")
    return results


if __name__ == "__main__":
    print("[Phase 2a] thin model with 4-fold CV + 'ignore rejects' baseline")

    print("\nLoading approved loans (split 'train')...")
    X_appr, y_appr, issue_d_appr = load_approved_shared()
    print(f"  {len(X_appr):,} rows | bad rate {y_appr.mean()*100:.2f}%")
    print(X_appr.describe())

    print("\nLoading rejected applications (partitioned Parquet, 27,648,741 rows expected)...")
    X_rej, rej_flags, rej_counts = load_rejected_shared()
    print(f"  {len(X_rej):,} rows | censored (dti>=100%): {int(rej_flags['dti_censored'].sum()):,}"
          f" | emp_length_missing: {int(rej_flags['emp_length_missing'].sum()):,}"
          f" (flags kept as metadata, not used in parcelling)")
    print(X_rej.describe())

    print("\nTraining thin model with 4-fold stratified CV "
          "(out-of-sample AUC)...")
    thin, aucs_cv = thin_model_cv(X_appr, y_appr)
    coefs = dict(zip(SHARED, thin.named_steps["clf"].coef_[0]))
    print("  Final model coefficients (standardized, fitted on all approved loans):", coefs)

    print("\nBuilding the formal 'ignore rejects' baseline...")
    baseline = baseline_ignore_rejects(X_appr, y_appr)
    print(f"  baseline_ignore_rejects() OK -- thin model trained only on approved loans, "
          f"no reject information. Coefficients identical to the final CV model "
          f"(same data/seed): {dict(zip(SHARED, baseline.named_steps['clf'].coef_[0]))}")

    print("\nRunning parcelling sweep (bands x base_mult, no censored_extra)...")
    res = sweep(thin, X_appr, y_appr, X_rej)

    print("\n[Phase 2a] done.")
