"""Feature engineering for the credit-default-prediction-lendingclub project.

build_features is copied verbatim from notebook 05 - it does not read any statistic
external to the row itself (no fit/transform state, no leakage risk from other rows).

prepare_X is the encoding step reused, identically, across notebooks 06-13: it converts
the two datetime columns to days-since-reference and one-hot encodes the categorical
columns. It is not part of notebook 05's validated function - it is added here because
training or scoring any model requires it, and every downstream notebook already depends
on this exact implementation.
"""
from src.data import CATEGORICAL_COLS, REFERENCE_DATE

import numpy as np
import pandas as pd


def build_features(df):
    """Receives ONE DataFrame, returns the same one with new columns. Only row-wise transformations;
    does not read, receive, or reference any other dataset.

    fico_mean and bankcard_to_total_limit were removed: fico_mean has a 1.0000 correlation
    with fico_range_high (the low/high spread is constant, the mean is just a linear translation -
    it adds no information). bankcard_to_total_limit was dropped for having ~30% of training
    without a defined value (era_pre_2012 sentinel plus organic 0/0) in exchange for a
    univariate AUC of only 0.5393.
    """
    df = df.copy()

    df["installment_to_income"] = df["installment"] / (df["annual_inc"] / 12)
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"]
    df["credit_history_months"] = ((df["issue_d"].dt.year - df["earliest_cr_line"].dt.year) * 12
                                    + (df["issue_d"].dt.month - df["earliest_cr_line"].dt.month))
    df["revol_bal_to_income"] = df["revol_bal"] / df["annual_inc"]
    df["open_acc_ratio"] = df["open_acc"] / df["total_acc"]

    return df


def prepare_X(df, feature_cols, categorical_cols=CATEGORICAL_COLS, drop_first=True):
    """Select feature_cols and encode them into a model-ready numeric matrix.

    Datetime columns (issue_d, earliest_cr_line) are converted to days since
    REFERENCE_DATE; categorical_cols are one-hot encoded with drop_first=True. Identical
    to the prepare_X used across notebooks 06-13.

    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe, must contain all of feature_cols.
    feature_cols : list of str
        Columns to select (typically FEATURE_SET from src.data).
    categorical_cols : list of str
        Subset of feature_cols to one-hot encode.

    drop_first : bool, default True
        True reproduces the training-time encoding (notebooks 06-13): the
        alphabetically-first category of each column becomes the implicit base and gets
        no column. KEEP True for training.

        False is for INFERENCE, and it is not an alternative encoding - it is how you get
        the SAME encoding on a batch that does not contain every category. See the
        warning below.

    Returns
    -------
    pandas.DataFrame
        Numeric feature matrix.

    Warning
    -------
    With drop_first=True the produced column set depends on WHICH CATEGORIES ARE PRESENT
    IN THIS CALL, not on the training vocabulary. On a single-row batch every categorical
    has exactly one category, drop_first removes it, and ZERO one-hot columns are
    produced; a caller that then reindexes with fill_value=0 silently gets an all-base
    row. That was bug P-043: the API scored every applicant as the base category, and the
    same record got a different encoding depending on who else was in the batch.

    For inference, call with drop_first=False and reindex onto the trained column list.
    That is provably equivalent to the training encoding: a non-base category keeps its
    column (value 1), and a base category produces a column absent from the trained list,
    which the reindex drops - leaving the whole group at zero, which is exactly how the
    base is represented at training time.
    """
    X = df[feature_cols].copy()
    for c in ["issue_d", "earliest_cr_line"]:
        if c in X.columns:
            X[c] = (X[c] - REFERENCE_DATE).dt.days
    cat_present = [c for c in categorical_cols if c in X.columns]
    X = pd.get_dummies(X, columns=cat_present, drop_first=drop_first)
    return X
def assert_matriz_finita(X, contexto=""):
    """Raise if the feature matrix contains NaN or +/-Inf. Silent otherwise.

    Bug P-011 (2026-08-31). build_features divides by annual_inc (three ratios) and by
    total_acc (one), so a zero in either produces Inf or NaN. Measured on the frozen
    training split: 0 of 90 columns carry NaN or Inf, over 172,988 rows, and no row has
    annual_inc == 0 or total_acc == 0. So on the training path this check is silent
    today, by measurement and not by hope.

    Which means: on the training path it cannot fire because of the data -- the parquet is
    frozen and was measured clean. It fires when SOMEONE CHANGES THE FEATURE CODE. That is
    what it is for, and it is why it lives here rather than in the notebooks.

    It is deliberately NOT called inside prepare_X. prepare_X is shared by training and
    serving, and raising on the serving path would kill the drift monitor on a single
    dirty row. The serving side is guarded differently: src.api narrows the contract so
    the zero cannot arrive (P-047), and score_frame checks the OUTPUT probability, which
    cannot be non-finite under any circumstance.
    """
    num = X.select_dtypes(include=[np.number])
    com_nan = [c for c in X.columns if X[c].isna().any()]
    com_inf = [c for c in num.columns if np.isinf(num[c]).any()]
    if com_nan or com_inf:
        raise ValueError(
            f"Non-finite feature matrix{' (' + contexto + ')' if contexto else ''}: "
            f"NaN in {com_nan}; Inf in {com_inf}. "
            "build_features divides by annual_inc and total_acc -- a zero in either is the "
            "usual cause. See P-011."
        )
