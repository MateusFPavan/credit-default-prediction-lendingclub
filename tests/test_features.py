"""
Unit tests for src/features.py -- build_features and prepare_X.

Covers ML Test Score item Data 7 ("all input feature code is tested"): the feature
creation code had no tests of its own, only economics.py and psi.py did.

The most important test in this file is test_build_features_e_puro_por_linha: it
turns the claim in build_features' docstring ("does not read, receive, or reference
any other dataset") into an executable test. That property is why the pipeline has no
state leaking between training and inference; without a test it is only a promise in
prose.

The two test_REGRESSAO_* tests guard against batch-dependent one-hot encoding at
inference, a bug found while this file was being written. With drop_first=True and a
1-row batch, get_dummies produces ZERO categorical columns, and score_frame's
reindex(fill_value=0) silently filled everything with 0 -- since the API is
single-record, EVERY request was scored as if the applicant were the base category.

Verified before the fix: both tests ran marked as expected failures and pytest
reported "2 xfailed" -- i.e. they genuinely failed, reproducing the bug. Against the
real model the same record scored p=0.1341794431 alone and p=0.1289446801 in a batch.
The fix (drop_first=False at inference + reindex) made them pass; the marker was then
removed and they became ordinary regression tests.
"""
import numpy as np
import pandas as pd
import pytest

from src.data import CATEGORICAL_COLS, REFERENCE_DATE
from src.features import build_features, prepare_X


def _frame_minimo(n=3):
    """Frame with the columns build_features consumes, distinct values per row."""
    return pd.DataFrame({
        "installment": [300.0, 500.0, 150.0][:n],
        "annual_inc": [60000.0, 120000.0, 30000.0][:n],
        "loan_amnt": [10000.0, 25000.0, 5000.0][:n],
        "revol_bal": [8500.0, 20000.0, 1000.0][:n],
        "open_acc": [8.0, 12.0, 4.0][:n],
        "total_acc": [20.0, 30.0, 10.0][:n],
        "issue_d": pd.to_datetime(["2015-06-01", "2016-01-01", "2014-11-15"][:n]),
        "earliest_cr_line": pd.to_datetime(["2001-08-01", "1998-03-01", "2010-03-15"][:n]),
    })


# --- build_features: row-wise purity (the property the pipeline rests on) ---

def test_build_features_e_puro_por_linha():
    """Processing one subset at a time must give exactly the same result as processing
    the whole frame and slicing afterwards. If build_features read any statistic from
    other rows (mean, median, min/max), this test would break."""
    df = _frame_minimo(3)
    inteiro = build_features(df)

    derivadas = ["installment_to_income", "loan_to_income", "credit_history_months",
                 "revol_bal_to_income", "open_acc_ratio"]

    for i in range(len(df)):
        uma_linha = build_features(df.iloc[[i]])
        pd.testing.assert_frame_equal(
            uma_linha[derivadas].reset_index(drop=True),
            inteiro[derivadas].iloc[[i]].reset_index(drop=True),
        )


def test_build_features_nao_muta_a_entrada():
    df = _frame_minimo()
    colunas_antes = list(df.columns)
    build_features(df)
    assert list(df.columns) == colunas_antes


# --- build_features: each derived feature, computed by hand ---

def test_installment_to_income_e_parcela_sobre_renda_mensal():
    df = build_features(_frame_minimo(1))
    assert df["installment_to_income"].iloc[0] == pytest.approx(300.0 / (60000.0 / 12))


def test_loan_to_income_usa_renda_anual_nao_mensal():
    df = build_features(_frame_minimo(1))
    assert df["loan_to_income"].iloc[0] == pytest.approx(10000.0 / 60000.0)


def test_revol_bal_to_income_usa_renda_anual():
    df = build_features(_frame_minimo(1))
    assert df["revol_bal_to_income"].iloc[0] == pytest.approx(8500.0 / 60000.0)


def test_open_acc_ratio():
    df = build_features(_frame_minimo(1))
    assert df["open_acc_ratio"].iloc[0] == pytest.approx(8.0 / 20.0)


def test_credit_history_months_conta_meses_atravessando_o_ano():
    """2001-08 -> 2015-06: 14 full years minus 2 months = 166 months."""
    df = build_features(_frame_minimo(1))
    assert df["credit_history_months"].iloc[0] == (2015 - 2001) * 12 + (6 - 8)
    assert df["credit_history_months"].iloc[0] == 166


def test_credit_history_months_pode_ser_negativo_se_as_datas_estiverem_invertidas():
    """Documents current behaviour: the function does not validate date order.
    A record with earliest_cr_line after issue_d produces a negative value instead
    of an error. Date order is validated by the API schema, not by build_features."""
    df = _frame_minimo(1)
    df["earliest_cr_line"] = pd.to_datetime(["2020-01-01"])
    out = build_features(df)
    assert out["credit_history_months"].iloc[0] < 0


# --- build_features: the edge that produces infinity (see the finiteness guards) ---

def test_renda_zero_produz_infinito_nas_tres_razoes_de_renda():
    """A FINDING, not a bug fixed here: annual_inc == 0 yields +inf in
    installment_to_income, loan_to_income and revol_bal_to_income (division by zero
    in float64 does not raise, it returns inf).

    build_features deliberately does NOT handle this -- it is a pure row-wise
    transformation with no policy. Stopping the case is the job of the API contract
    (annual_inc > 0) and the finiteness guards (features.assert_matriz_finita, and the
    output check in score_frame). This test exists so the edge stays on record and
    breaks if someone changes the behaviour by accident."""
    df = _frame_minimo(1)
    df["annual_inc"] = [0.0]
    out = build_features(df)
    assert np.isinf(out["installment_to_income"].iloc[0])
    assert np.isinf(out["loan_to_income"].iloc[0])
    assert np.isinf(out["revol_bal_to_income"].iloc[0])


def test_total_acc_zero_produz_nan_em_open_acc_ratio():
    """0/0 in float64 gives NaN (not inf). Unlike inf, NaN is a legitimate XGBoost
    input: the model routes it along a learned default direction."""
    df = _frame_minimo(1)
    df["open_acc"] = [0.0]
    df["total_acc"] = [0.0]
    out = build_features(df)
    assert np.isnan(out["open_acc_ratio"].iloc[0])


# --- prepare_X: determinism and no state ---

def test_prepare_X_e_deterministico():
    df = build_features(_frame_minimo())
    cols = ["installment", "annual_inc", "issue_d", "earliest_cr_line", "home_ownership"]
    df["home_ownership"] = ["rent", "own", "mortgage"]
    a = prepare_X(df, cols, ["home_ownership"])
    b = prepare_X(df, cols, ["home_ownership"])
    pd.testing.assert_frame_equal(a, b)


def test_prepare_X_nao_guarda_estado_entre_chamadas():
    """A call with one frame must not influence the result for the next frame.
    If prepare_X remembered categories it had seen (like a fitted encoder), this breaks."""
    cols = ["annual_inc", "home_ownership"]
    df1 = pd.DataFrame({"annual_inc": [1.0, 2.0], "home_ownership": ["rent", "own"]})
    df2 = pd.DataFrame({"annual_inc": [3.0], "home_ownership": ["rent"]})

    sozinho = prepare_X(df2, cols, ["home_ownership"])
    prepare_X(df1, cols, ["home_ownership"])
    depois = prepare_X(df2, cols, ["home_ownership"])
    pd.testing.assert_frame_equal(sozinho, depois)


def test_prepare_X_converte_datas_para_dias_desde_a_referencia():
    df = pd.DataFrame({
        "issue_d": pd.to_datetime(["2015-06-01"]),
        "earliest_cr_line": pd.to_datetime(["2001-08-01"]),
    })
    X = prepare_X(df, ["issue_d", "earliest_cr_line"], [])
    assert REFERENCE_DATE == pd.Timestamp("2000-01-01")
    assert X["issue_d"].iloc[0] == 5630
    assert X["earliest_cr_line"].iloc[0] == 578


def test_prepare_X_faz_one_hot_com_drop_first():
    """3 categories -> 2 columns (the alphabetically first one is the base)."""
    df = pd.DataFrame({"home_ownership": ["rent", "own", "mortgage"]})
    X = prepare_X(df, ["home_ownership"], ["home_ownership"])
    assert "home_ownership_mortgage" not in X.columns
    assert set(X.columns) == {"home_ownership_own", "home_ownership_rent"}


def test_prepare_X_ignora_categorica_que_nao_esta_em_feature_cols():
    """categorical_cols may list absent columns without breaking (cat_present)."""
    df = pd.DataFrame({"annual_inc": [1.0]})
    X = prepare_X(df, ["annual_inc"], CATEGORICAL_COLS)
    assert list(X.columns) == ["annual_inc"]


# --- prepare_X: the risk its docstring names, and the bug it hides ---

def test_prepare_X_muda_colunas_com_categorias_diferentes():
    """prepare_X's docstring warns that, with drop_first=True, the produced column set
    depends on which categories are present in the call, not on the training
    vocabulary.

    This test proves the warning is real."""
    cols = ["home_ownership"]
    treino = pd.DataFrame({"home_ownership": ["rent", "own", "mortgage"]})
    lote = pd.DataFrame({"home_ownership": ["rent", "own"]})

    X_treino = prepare_X(treino, cols, cols)
    X_lote = prepare_X(lote, cols, cols)

    assert list(X_treino.columns) == ["home_ownership_own", "home_ownership_rent"]
    assert list(X_lote.columns) == ["home_ownership_rent"]  # drop_first removed 'own'
    assert list(X_treino.columns) != list(X_lote.columns)


def test_REGRESSAO_um_registro_sozinho_mantem_as_categoricas():
    """Regression guard: a single record on its own must be encoded correctly.

    Before the fix: drop_first=True removed the only category present, get_dummies
    produced ZERO columns, and the reindex filled everything with 0 -- the applicant
    became the base category. Since the API is single-record (src/api.py: `def score(req:
    ScoreRequest)`), that applied to EVERY request."""
    cols = ["home_ownership"]
    colunas_de_treino = ["home_ownership_own", "home_ownership_rent"]

    um = pd.DataFrame({"home_ownership": ["rent"]})
    X = prepare_X(um, cols, cols, drop_first=False)
    X_alinhado = X.reindex(columns=colunas_de_treino, fill_value=0)

    assert X_alinhado["home_ownership_rent"].iloc[0] == 1
    assert X_alinhado["home_ownership_own"].iloc[0] == 0


def test_categoria_base_sozinha_fica_com_todas_as_dummies_em_zero():
    """The other half of the equivalence, and the one almost nobody tests.

    The base category ('mortgage', alphabetically first) has no training column. With
    drop_first=False it GETS a column, which the reindex drops because it is not in the
    trained list -- leaving the whole group at zero, which is exactly how the base is
    represented at training time. Without this test, the fix could be right for 3 of
    the 4 categories and wrong precisely for the base."""
    cols = ["home_ownership"]
    colunas_de_treino = ["home_ownership_own", "home_ownership_rent"]

    um = pd.DataFrame({"home_ownership": ["mortgage"]})
    X = prepare_X(um, cols, cols, drop_first=False)
    assert "home_ownership_mortgage" in X.columns

    X_alinhado = X.reindex(columns=colunas_de_treino, fill_value=0)
    assert list(X_alinhado.columns) == colunas_de_treino
    assert X_alinhado.sum(axis=1).iloc[0] == 0


def test_REGRESSAO_mesmo_registro_independe_do_lote():
    """Regression guard, second face of the same bug and the more serious one.

    Before the fix the SAME record got a different encoding depending on who else was
    in the batch, because drop_first picks the base from the categories PRESENT in that
    batch. Row independence is a non-negotiable property of a scorer."""
    cols = ["home_ownership"]
    colunas_de_treino = ["home_ownership_own", "home_ownership_rent"]
    reg = {"home_ownership": "rent"}
    outro = {"home_ownership": "own"}

    sozinho = prepare_X(pd.DataFrame([reg]), cols, cols, drop_first=False).reindex(
        columns=colunas_de_treino, fill_value=False)
    em_lote = prepare_X(pd.DataFrame([reg, outro]), cols, cols, drop_first=False).reindex(
        columns=colunas_de_treino, fill_value=False)

    # compares VALUE, and dtype too -- with fill_value=False (not 0) both matrices stay
    # bool in both situations. With fill_value=0 the values matched (0 == False) but the
    # dtype did not (int64 vs bool), which is how this test failed on the first attempt
    # at the fix.
    assert sozinho.iloc[0].tolist() == em_lote.iloc[0].tolist()
    assert list(sozinho.dtypes) == list(em_lote.dtypes), (
        f"dtype depende do lote: {dict(sozinho.dtypes)} vs {dict(em_lote.dtypes)}"
    )


def test_drop_first_True_na_inferencia_ainda_falha_em_silencio():
    """Why the fix had to be in the CALLER and not in the reindex.

    This test keeps the old behaviour (drop_first=True on a single row) alive to
    document WHY it is dangerous: reindex(fill_value=0) fills a MISSING column with 0,
    which is indistinguishable from 'the base category was observed'. It does not raise,
    does not log, passes CI -- the response stays well-formed.

    Transferable lesson: a fill value that coincides with a legitimate value turns an
    error into silence. Checking that the safeguard exists is not the same as checking
    what it receives."""
    cols = ["home_ownership"]
    colunas_de_treino = ["home_ownership_own", "home_ownership_rent"]

    X = prepare_X(pd.DataFrame({"home_ownership": ["rent"]}), cols, cols, drop_first=True)
    alinhado = X.reindex(columns=colunas_de_treino, fill_value=0)

    assert list(alinhado.columns) == colunas_de_treino   # correct shape
    assert alinhado.sum(axis=1).iloc[0] == 0             # silently wrong content
