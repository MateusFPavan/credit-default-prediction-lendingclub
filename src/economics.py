"""Profit-based evaluation used throughout the credit-default-prediction-lendingclub project.

These three functions are copied, unchanged in behavior, from the versions used across
notebooks 06-13: interest/loss reconstruction from raw loan fields, profit at a fixed
threshold, and the profit-maximizing threshold over a standard grid.
"""
import numpy as np


def compute_interest_loss(df):
    """Reconstruct realized interest and loss from raw loan fields.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain installment, term, loan_amnt, total_rec_prncp.

    Returns
    -------
    (pandas.Series, pandas.Series)
        interest: realized interest income if the loan is fully paid.
        loss: principal lost if the loan charges off (clipped at zero - a loan that
        recovered more principal than was lent contributes no negative loss).
    """
    interest = (df["installment"] * df["term"]) - df["loan_amnt"]
    loss_raw = df["loan_amnt"] - df["total_rec_prncp"]
    return interest, loss_raw.clip(lower=0)


def profit_at_threshold(y_true, y_prob, threshold, interest, loss):
    """Portfolio profit if every loan with y_prob < threshold is approved.

    Parameters
    ----------
    y_true : array-like of {0, 1}
        1 = defaulted (charged off), 0 = fully paid.
    y_prob : array-like of float
        Predicted probability of default.
    threshold : float
        Approve when y_prob < threshold.
    interest : array-like of float
        Interest earned per loan if fully paid (see compute_interest_loss).
    loss : array-like of float
        Principal lost per loan if charged off (see compute_interest_loss).

    Returns
    -------
    float
        Sum of interest for approved good loans minus sum of loss for approved bad loans.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    interest = np.asarray(interest)
    loss = np.asarray(loss)
    aprovados = y_prob < threshold
    return interest[aprovados & (y_true == 0)].sum() - loss[aprovados & (y_true == 1)].sum()


def optimal_threshold(y_true, y_prob, interest, loss, thresholds=None):
    """Sweep candidate thresholds (default 0.01-0.99, step 0.01) and return the profit
    maximizer.

    Uses the same O(N log N) sort + cumulative-sum approach as fast_profit_curve in
    notebooks 09-13, so it is fast enough for repeated use inside a bootstrap or a
    hyperparameter search, and produces results identical to sweeping profit_at_threshold
    over the same grid.

    Parameters
    ----------
    y_true, y_prob, interest, loss : array-like
        Same meaning as in profit_at_threshold.
    thresholds : array-like of float, optional
        Candidate thresholds to evaluate. Defaults to np.arange(0.01, 1.0, 0.01) rounded
        to 2 decimals (0.01 through 0.99).

    Returns
    -------
    (float, float)
        (best_threshold, best_profit) - the threshold with the highest profit among
        thresholds, and that profit. Ties resolve to the first (lowest) threshold
        achieving the maximum, matching numpy.argmax's behavior used throughout the
        project's notebooks.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    interest = np.asarray(interest)
    loss = np.asarray(loss)
    thresholds = np.asarray(thresholds)

    order = np.argsort(y_prob)
    y_sorted = y_true[order]
    interest_sorted = interest[order]
    loss_sorted = loss[order]
    prob_sorted = y_prob[order]

    contrib = np.where(y_sorted == 0, interest_sorted, -loss_sorted)
    cumsum = np.cumsum(contrib)
    idx_cut = np.searchsorted(prob_sorted, thresholds, side="left")
    profits = np.where(idx_cut > 0, cumsum[np.clip(idx_cut - 1, 0, None)], 0.0)

    best_i = int(np.argmax(profits))
    return float(thresholds[best_i]), float(profits[best_i])
