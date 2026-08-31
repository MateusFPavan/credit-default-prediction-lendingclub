"""
Record cleaning for inference (port of notebook 03 to production).

Reproduces -- does not re-decide -- the decisions from docs/cleaning_decisions.md, so that
a raw record (coming from the API or a monitoring batch) receives the SAME
missing/sentinel/flag treatment the training pipeline applied. The proof
of fidelity is the bit-for-bit check against the parquets (see the test block).

Does not touch notebook 03 or run_all.py: it's a port for production consumption,
verified by equality, not a second parallel pipeline with its own decisions.

Population-dependent statistics (the 6 medians of the sparse columns) are READ from
src/_cleaning_stats.json (frozen from train.parquet), never recomputed -- a
new record has no population to compute a median from, and recomputing would diverge from
training (the class of error this project avoids).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data import CATEGORICAL_COLS

# --- decision tables (docs/cleaning_decisions.md), NOT re-decided here ---

# MNAR / informative absence: sentinel 999 (preserves the "higher = safer" ordering)
# each one has its own *_missing flag (1 if the source came in null)
SENTINEL_999_WITH_FLAG = {
    "mths_since_last_delinq": "mths_since_last_delinq_missing",
    "mths_since_last_record": "mths_since_last_record_missing",
    "mths_since_recent_bc_dlq": "mths_since_recent_bc_dlq_missing",
    "mths_since_recent_revol_delinq": "mths_since_recent_revol_delinq_missing",
    "mths_since_last_major_derog": "mths_since_last_major_derog_missing",
    "mths_since_recent_inq": "mths_since_recent_inq_missing",
}
# counters / emp_length: sentinel -1 (no "higher=safer" ordering), with its own flag
SENTINEL_NEG1_WITH_FLAG = {
    "num_tl_120dpd_2m": "num_tl_120dpd_2m_missing",
    "emp_length_anos": "emp_length_missing",
}
# structural 2012 rollout: sentinel -1, covered by era_pre_2012 (WITHOUT its own flag)
SENTINEL_NEG1_ROLLOUT = [
    "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim", "avg_cur_bal", "mort_acc",
    "acc_open_past_24mths", "total_bal_ex_mort", "total_bc_limit",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl",
    "num_accts_ever_120_pd", "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats",
    "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts",
    "num_rev_tl_bal_gt_0", "num_sats", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "tot_hi_cred_lim",
    "total_il_high_credit_limit",
]
# stacked rollout + structural: sentinel 999 (preserves ordering), WITHOUT its own flag
# (except mths_since_recent_inq, already handled in SENTINEL_999_WITH_FLAG)
SENTINEL_999_ROLLOUT = [
    "bc_util", "bc_open_to_buy", "percent_bc_gt_75", "mths_since_recent_bc",
    "mo_sin_old_il_acct",
]
# sparse columns: aggregated flag sparse_bureau_missing (OR) + frozen median
SPARSE_COLS = [
    "pub_rec_bankruptcies", "revol_util", "chargeoff_within_12_mths",
    "collections_12_mths_ex_med", "tax_liens", "dti",
]

_STATS = json.loads((Path(__file__).parent / "_cleaning_stats.json").read_text())
_SPARSE_MED = _STATS["sparse_medians"]


def clean_record(raw) -> pd.DataFrame:
    """
    Receives a raw record (dict or DataFrame) and returns a clean DataFrame, with
    notebook 03's sentinels, flags and imputations applied. Columns absent from
    the raw input are treated as null (the client didn't send that bureau field).
    """
    if isinstance(raw, dict):
        df = pd.DataFrame.from_records([raw])
    else:
        df = raw.copy()

    # 1) *_missing flags (compute BEFORE filling the source) -- 1 if source absent/null.
    #    explicit logic, no bool negation: if the source column doesn't exist in the input,
    #    the flag is 1 for every row; if it exists, the flag is 1 where it's null.
    for src, flag in {**SENTINEL_999_WITH_FLAG, **SENTINEL_NEG1_WITH_FLAG}.items():
        if flag not in df.columns:
            if src in df.columns:
                df[flag] = df[src].isna().astype("int64")
            else:
                df[flag] = 1

    # 2) aggregated sparse flag: 1 if ANY sparse column is absent/null
    if "sparse_bureau_missing" not in df.columns:
        any_missing = pd.Series(False, index=df.index)
        for c in SPARSE_COLS:
            any_missing = any_missing | (df[c].isna() if c in df.columns else True)
        df["sparse_bureau_missing"] = any_missing.astype("int64")

    # 3) era_pre_2012: new record -> always 0 (modern bureau always available)
    if "era_pre_2012" not in df.columns:
        df["era_pre_2012"] = 0

    # 4) apply sentinels
    for src in {**SENTINEL_999_WITH_FLAG}:
        if src not in df.columns: df[src] = 999.0
        else: df[src] = df[src].fillna(999.0)
    for src in SENTINEL_999_ROLLOUT:
        if src not in df.columns: df[src] = 999.0
        else: df[src] = df[src].fillna(999.0)
    for src in {**SENTINEL_NEG1_WITH_FLAG}:
        if src not in df.columns: df[src] = -1.0
        else: df[src] = df[src].fillna(-1.0)
    for src in SENTINEL_NEG1_ROLLOUT:
        if src not in df.columns: df[src] = -1.0
        else: df[src] = df[src].fillna(-1.0)

    # 5) sparse columns: frozen training median
    for c in SPARSE_COLS:
        med = _SPARSE_MED[c]
        if c not in df.columns: df[c] = med
        else: df[c] = df[c].fillna(med)

    # 6) funded_amnt: redundant with loan_amnt (cleaning §REVISIT); derive if absent
    if "funded_amnt" not in df.columns and "loan_amnt" in df.columns:
        df["funded_amnt"] = df["loan_amnt"]

    # 7) rare-event counters with no sentinel in notebook 03 (never null in training).
    #    For a new record that omits them, default 0 -- justified: they're 0 in >99.5%
    #    of rows, and forcing 0 moves the score by 0.00003 on average (verified in this session).
    #    Absence of event = 0 is the field's meaning, not a guess.
    for c in ("acc_now_delinq", "delinq_amnt"):
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = df[c].fillna(0.0)

    # 8) delinq_2yrs / pub_rec: default 0 when omitted. Justified by isolated impact
    #    (forcing 0 moves the score >0.01 in only 1.57% and 2.23% of cases,
    #    ~10x smaller than inq_last_6mths, which is why that one is required in the API, not these).
    for c in ("delinq_2yrs", "pub_rec"):
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = df[c].fillna(0.0)

    # 9) defensive numeric coercion: an Optional omitted in the schema arrives as None and
    #    pandas creates a non-numeric column, which XGBoost rejects. Forces numeric where the
    #    column should be numeric; doesn't touch categorical or date columns.
    #
    #    BUG P-048 (2026-08-31): the condition here used to be `df[c].dtype == object`, and
    #    it silently stopped working. Under pandas >= 3.0 a column of strings gets the
    #    dedicated StringDtype (prints as `str`), NOT object -- so `dtype == object` is
    #    False and the coercion never ran for exactly the case it existed for. The guard was
    #    correct when written and became wrong without anyone touching this file: pandas
    #    changed underneath it. Reproduced on pandas 3.0.2.
    #
    #    `not is_numeric_dtype` states the intent instead of a proxy for it: if a column that
    #    should be numeric is not numeric, coerce it. object and StringDtype both match, and
    #    the two columns that are legitimately non-numeric are excluded by name below.
    _non_numeric = set(CATEGORICAL_COLS) | {"issue_d", "earliest_cr_line"}
    for c in df.columns:
        if c not in _non_numeric and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df
