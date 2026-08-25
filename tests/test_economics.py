"""
Testes unitarios de src/economics.py -- funcoes de profit que geram todo numero em
dolar do relatorio (cobertura C1.2, auditoria Boletim 12/13).

Cobre: reconstrucao de interest/loss (incluindo loss clipada em zero),
profit_at_threshold em casos pequenos calculados a mao, e optimal_threshold
contra um sweep de forca bruta + o criterio de desempate (menor threshold
entre os empatados, per docstring da funcao).
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
    """Emprestimo que recuperou mais principal do que foi emprestado nao gera perda negativa."""
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
    # threshold=0.4 -> aprovados (prob<0.4): indices 0,1, ambos bons (y_true=0)
    profit = profit_at_threshold(y_true, y_prob, 0.4, interest, loss)
    assert profit == pytest.approx(300.0)  # 100 + 200, sem perda


def test_profit_at_threshold_approves_a_bad_loan():
    y_true = np.array([0, 1])
    y_prob = np.array([0.2, 0.3])
    interest = np.array([100.0, 150.0])
    loss = np.array([40.0, 90.0])
    # threshold=0.5 -> ambos aprovados
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
    """Construido para empatar de proposito: aprovar o 2o emprestimo nao muda o profit,
    entao 0.2 e 0.6 dao o mesmo resultado -- o menor (0.2) deve vencer."""
    y_true = np.array([0, 0, 1])
    y_prob = np.array([0.1, 0.5, 0.9])
    interest = np.array([100.0, 0.0, 0.0])
    loss = np.array([0.0, 0.0, 500.0])
    thresholds = np.array([0.2, 0.6])

    best_t, best_profit = optimal_threshold(y_true, y_prob, interest, loss, thresholds=thresholds)
    assert best_t == pytest.approx(0.2)
    assert best_profit == pytest.approx(100.0)
