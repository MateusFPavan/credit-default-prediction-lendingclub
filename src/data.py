"""Data loading and feature-set definitions for the credit-default-prediction-lendingclub project.

These constants and the loader are copied from the values established and validated across
notebooks 04-13 - the exact FEATURE_SET (79 columns) used to train and evaluate every model
in this project, and a single entry point to load any of the four processed splits.
"""
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

_SPLIT_FILES = {
    "train": "train.parquet",
    "validation": "validation.parquet",
    "test": "test.parquet",
    "transfer_60m": "transfer_60m.parquet",
}

# Columns used only for financial calculations (interest/loss) or as the raw target source,
# never fed to a model as a predictive feature.
EVAL_ONLY = ["loan_status", "loan_amnt", "installment", "term", "total_rec_prncp"]

# Columns intentionally withheld from FEATURE_SET: int_rate/grade/sub_grade encode LendingClub's
# own risk assessment (partly derived from information not available at underwriting time in a
# comparable form), so training on them would leak the incumbent model's decision into ours.
EXCLUDED = ["int_rate", "grade", "sub_grade"]

_family_C_features = [
    "funded_amnt", "home_ownership", "annual_inc", "verification_status", "issue_d",
    "purpose", "dti", "delinq_2yrs", "earliest_cr_line", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "initial_list_status", "collections_12_mths_ex_med",
    "mths_since_last_major_derog", "application_type", "acc_now_delinq", "tot_coll_amt",
    "tot_cur_bal", "total_rev_hi_lim", "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy",
    "bc_util", "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
    "mths_since_recent_bc", "mths_since_recent_bc_dlq", "mths_since_recent_inq",
    "mths_since_recent_revol_delinq", "num_accts_ever_120_pd", "num_actv_bc_tl", "num_actv_rev_tl",
    "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts",
    "num_rev_tl_bal_gt_0", "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies",
    "tax_liens", "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
    "total_il_high_credit_limit", "emp_length_anos",
]
assert len(_family_C_features) == 65

_engineered_flags = [
    "era_pre_2012",
    "mths_since_last_delinq_missing", "mths_since_last_record_missing",
    "mths_since_recent_bc_dlq_missing", "mths_since_recent_revol_delinq_missing",
    "mths_since_last_major_derog_missing", "emp_length_missing",
    "mths_since_recent_inq_missing", "num_tl_120dpd_2m_missing", "sparse_bureau_missing",
]
assert len(_engineered_flags) == 10

_new_features = [
    "installment_to_income", "loan_to_income", "credit_history_months",
    "revol_bal_to_income", "open_acc_ratio",
]
assert len(_new_features) == 5

# fico_range_high dropped: correlation of 1.0 with fico_range_low (constant spread across
# the dataset), so it carries no information fico_range_low doesn't already provide.
_REDUNDANT_COLS = {"fico_range_high": "correlacao 1.0 com fico_range_low"}

FEATURE_SET = [c for c in _family_C_features if c not in _REDUNDANT_COLS] + _engineered_flags + _new_features
assert len(FEATURE_SET) == 79

CATEGORICAL_COLS = ["home_ownership", "purpose", "verification_status", "initial_list_status", "application_type"]
REFERENCE_DATE = pd.Timestamp("2000-01-01")


def load_split(name):
    """Load one of the processed dataset splits from data/processed/.

    Parameters
    ----------
    name : str
        One of "train", "validation", "test", "transfer_60m".

    Returns
    -------
    pandas.DataFrame
        The split as written to disk, in on-disk row order (never reordered here).
    """
    if name not in _SPLIT_FILES:
        raise ValueError(f"Unknown split '{name}'. Expected one of {sorted(_SPLIT_FILES)}.")
    return pd.read_parquet(PROCESSED_DIR / _SPLIT_FILES[name])
