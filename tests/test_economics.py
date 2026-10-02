"""
Unit tests for src/economics.py -- the profit functions behind every dollar figure in
the report.

Covers: interest/loss reconstruction (including loss clipped at zero),
profit_at_threshold on small hand-computed cases, and optimal_threshold
against a brute-force sweep + the tie-break rule (lowest threshold among
the tied ones, per the function's docstring).
"""
import numpy as np
import pandas as pd
import pytest

from src.economics import compute_interest_loss, profit_at_threshold, optimal_threshold


# --- compute_interest_loss ---

def test_compute_interest_loss_basic():
    df = pd.DataFrame({
        "installment": [300.0], "term": [36], "loan_amnt": [10000.0],
        "total_rec_prncp": [10000.0],
    })
    interest, loss = compute_interest_loss(df)
    assert interest.iloc[0] == pytest.approx(300.0 * 36 - 10000.0)
    assert loss.iloc[0] == pytest.approx(0.0)


def test_compute_interest_loss_clips_negative_loss_at_zero():
    """A loan that recovered more principal than was lent does not produce a negative loss."""
    df = pd.DataFrame({
        "installment": [300.0], "term": [36], "loan_amnt": [10000.0],
        "total_rec_prncp": [10500.0],
    })
    _, loss = compute_interest_loss(df)
    assert loss.iloc[0] == 0.0


def test_compute_interest_loss_partial_loss():
    df = pd.DataFrame({
        "installment": [300.0], "term": [36], "loan_amnt": [10000.0],
        "total_rec_prncp": [4000.0],
    })
    _, loss = compute_interest_loss(df)
    assert loss.iloc[0] == pytest.approx(6000.0)


# --- profit_at_threshold ---

def test_profit_at_threshold_hand_computed():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.3, 0.5, 0.9])
    interest = np.array([100.0, 200.0, 300.0, 400.0])
    loss = np.array([50.0, 60.0, 70.0, 80.0])
    # threshold=0.4 -> approved (prob<0.4): indices 0,1, both good (y_true=0)
    profit = profit_at_threshold(y_true, y_prob, 0.4, interest, loss)
    assert profit == pytest.approx(300.0)  # 100 + 200, no loss


def test_profit_at_threshold_approves_a_bad_loan():
    y_true = np.array([0, 1])
    y_prob = np.array([0.2, 0.3])
    interest = np.array([100.0, 150.0])
    loss = np.array([40.0, 90.0])
    # threshold=0.5 -> both approved
    profit = profit_at_threshold(y_true, y_prob, 0.5, interest, loss)
    assert profit == pytest.approx(100.0 - 90.0)


def test_profit_at_threshold_no_approvals_is_zero():
    y_true = np.array([0, 1])
    y_prob = np.array([0.2, 0.3])
    interest = np.array([100.0, 150.0])
    loss = np.array([40.0, 90.0])
    profit = profit_at_threshold(y_true, y_prob, 0.0, interest, loss)
    assert profit == pytest.approx(0.0)


# --- optimal_threshold ---

def test_optimal_threshold_matches_brute_force_sweep():
    y_true = np.array([0, 1, 0, 1, 0])
    y_prob = np.array([0.15, 0.25, 0.40, 0.55, 0.70])
    interest = np.array([100.0, 120.0, 90.0, 150.0, 200.0])
    loss = np.array([30.0, 400.0, 25.0, 600.0, 40.0])
    thresholds = np.round(np.arange(0.05, 0.95, 0.05), 2)

    best_t, best_profit = optimal_threshold(y_true, y_prob, interest, loss, thresholds=thresholds)

    brute = [(t, profit_at_threshold(y_true, y_prob, t, interest, loss)) for t in thresholds]
    expected_t, expected_profit = max(brute, key=lambda tp: tp[1])

    assert best_t == pytest.approx(expected_t)
    assert best_profit == pytest.approx(expected_profit)


def test_optimal_threshold_ties_resolve_to_lowest():
    """Built to tie on purpose: approving the 2nd loan does not change the profit, so
    0.2 and 0.6 give the same result -- the lower one (0.2) must win."""
    y_true = np.array([0, 0, 1])
    y_prob = np.array([0.1, 0.5, 0.9])
    interest = np.array([100.0, 0.0, 0.0])
    loss = np.array([0.0, 0.0, 500.0])
    thresholds = np.array([0.2, 0.6])

    best_t, best_profit = optimal_threshold(y_true, y_prob, interest, loss, thresholds=thresholds)
    assert best_t == pytest.approx(0.2)
    assert best_profit == pytest.approx(100.0)
