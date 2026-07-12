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

import pandas as pd


def build_features(df):
    """Recebe UM DataFrame, devolve o mesmo com colunas novas. So transformacoes row-wise;
    nao le, nao recebe e nao referencia nenhum outro conjunto.

    fico_mean e bankcard_to_total_limit foram removidas: fico_mean tem correlacao 1.0000
    com fico_range_high (spread low/high e constante, media e apenas translacao linear -
    nao adiciona informacao). bankcard_to_total_limit foi descartada por ter ~30% do treino
    sem valor definido (sentinela em era_pre_2012 mais 0/0 organico) em troca de AUC
    univariada de apenas 0.5393.
    """
    df = df.copy()

    df["installment_to_income"] = df["installment"] / (df["annual_inc"] / 12)
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"]
    df["credit_history_months"] = ((df["issue_d"].dt.year - df["earliest_cr_line"].dt.year) * 12
                                    + (df["issue_d"].dt.month - df["earliest_cr_line"].dt.month))
    df["revol_bal_to_income"] = df["revol_bal"] / df["annual_inc"]
    df["open_acc_ratio"] = df["open_acc"] / df["total_acc"]

    return df


def prepare_X(df, feature_cols, categorical_cols=CATEGORICAL_COLS):
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

    Returns
    -------
    pandas.DataFrame
        Numeric feature matrix. Column set/order can differ between two different calls
        if the underlying categorical columns don't share the same categories - callers
        must reindex a scored split's columns to the training matrix's columns before
        predicting (see notebooks 06-13 for the established pattern).
    """
    X = df[feature_cols].copy()
    for c in ["issue_d", "earliest_cr_line"]:
        if c in X.columns:
            X[c] = (X[c] - REFERENCE_DATE).dt.days
    cat_present = [c for c in categorical_cols if c in X.columns]
    X = pd.get_dummies(X, columns=cat_present, drop_first=True)
    return X
