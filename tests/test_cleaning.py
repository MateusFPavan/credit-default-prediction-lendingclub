"""
Tests for src/cleaning.py -- the last module on the network-free serving path.

cleaning.py runs on EVERY request: score_frame calls clean_record before build_features,
applying the frozen sentinels, flags and medians. A bug here reaches every score.

What each group guards, in order of severity:

  1. BATCH INDEPENDENCE. The property that once broke in prepare_X, the neighbouring
     function. Here it should hold by construction -- everything is row-wise or comes
     from the frozen JSON -- but "should hold by construction" is exactly what was once
     said of prepare_X, shortly before it turned out to be batch-dependent. So it is
     asserted here.

  2. MEDIAN READ, NEVER RECOMPUTED. The module docstring calls this "the class of error
     this project avoids": recomputing the median from the batch makes serving-time
     imputation diverge from training. The test feeds batches with different
     distributions and requires the SAME imputed value.

  3. FLAG BEFORE FILL. The order matters and is easy to invert in a refactor: if someone
     fills the source before computing the flag, the flag becomes 0 for everyone and the
     missingness signal -- MNAR, treated as informative by this project -- disappears
     silently, without breaking anything.

Needs no parquet and no model: cleaning.py depends only on src/_cleaning_stats.json,
which is versioned.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.cleaning as cleaning
from src.cleaning import (
    SENTINEL_999_WITH_FLAG,
    SENTINEL_NEG1_WITH_FLAG,
    SENTINEL_999_ROLLOUT,
    SENTINEL_NEG1_ROLLOUT,
    SPARSE_COLS,
    clean_record,
)


MINIMO = {
    "loan_amnt": 10000.0, "installment": 325.5, "term": 36,
    "annual_inc": 60000.0, "fico_range_low": 710.0,
    "earliest_cr_line": "2001-08-01", "issue_d": "2015-06-01",
    "home_ownership": "rent", "purpose": "debt_consolidation",
    "verification_status": "verified", "initial_list_status": "w",
    "application_type": "individual",
    "open_acc": 8.0, "total_acc": 20.0, "revol_bal": 8500.0,
    "inq_last_6mths": 1.0,
}


# ------------------------------------------------ 1. batch independence (the serious one)

def test_registro_sozinho_e_em_lote_produzem_a_mesma_linha():
    """The property that once broke in the neighbouring prepare_X.

    If someone ever swaps a frozen median for `df[c].median()`, or a flag for something
    that looks at the batch, this test breaks. Without it, the failure would show up as a
    slightly different score depending on the company a record keeps -- the same failure
    mode as the batch-dependent encoding bug, and nobody notices it from the response."""
    a = dict(MINIMO)
    b = dict(MINIMO, annual_inc=20000.0, revol_util=90.0, dti=44.0, open_acc=2.0)

    sozinho = clean_record(pd.DataFrame([a]))
    em_lote = clean_record(pd.DataFrame([a, b]))

    colunas = sorted(set(sozinho.columns) & set(em_lote.columns))
    assert sorted(sozinho.columns) == sorted(em_lote.columns)
    pd.testing.assert_series_equal(
        sozinho[colunas].iloc[0], em_lote[colunas].iloc[0], check_names=False
    )


def test_a_ordem_das_linhas_no_lote_nao_muda_nenhuma_delas():
    """Complements the previous test: being equal alone is not enough, a row must be
    equal in any position."""
    a = dict(MINIMO)
    b = dict(MINIMO, annual_inc=20000.0, dti=44.0)

    ab = clean_record(pd.DataFrame([a, b])).reset_index(drop=True)
    ba = clean_record(pd.DataFrame([b, a])).reset_index(drop=True)

    colunas = sorted(ab.columns)
    pd.testing.assert_series_equal(
        ab[colunas].iloc[0], ba[colunas].iloc[1], check_names=False
    )


# ------------------------------------------------ 2. frozen median, never recomputed

def test_mediana_imputada_nao_depende_do_lote():
    """"The class of error this project avoids", in the module docstring's own words.

    Two batches with deliberately opposite distributions in the same column. If the
    median came from the batch, the two imputed values would differ."""
    baixo = [dict(MINIMO, revol_util=v) for v in (1.0, 2.0, 3.0)]
    alto = [dict(MINIMO, revol_util=v) for v in (95.0, 96.0, 97.0)]
    faltante = dict(MINIMO, revol_util=np.nan)

    imputado_baixo = clean_record(pd.DataFrame(baixo + [faltante]))["revol_util"].iloc[-1]
    imputado_alto = clean_record(pd.DataFrame(alto + [faltante]))["revol_util"].iloc[-1]

    assert imputado_baixo == imputado_alto


def test_o_valor_imputado_e_exatamente_o_do_json_congelado():
    """Being stable is not enough -- it has to be the training number."""
    stats = json.loads(
        (Path(cleaning.__file__).parent / "_cleaning_stats.json").read_text()
    )["sparse_medians"]

    df = clean_record(pd.DataFrame([dict(MINIMO)]))  # no sparse column provided
    for c in SPARSE_COLS:
        assert df[c].iloc[0] == stats[c], f"{c}: {df[c].iloc[0]} != {stats[c]}"


# ------------------------------------------------- 3. flag computed BEFORE the fill

@pytest.mark.parametrize("origem,flag", sorted(
    {**SENTINEL_999_WITH_FLAG, **SENTINEL_NEG1_WITH_FLAG}.items()
))
def test_coluna_ausente_marca_a_flag_e_recebe_a_sentinela(origem, flag):
    """The order: the flag is computed before the source is filled.

    If a refactor inverts it, the flag becomes 0 for everyone and the missingness --
    MNAR and informative here -- disappears without breaking anything."""
    df = clean_record(pd.DataFrame([dict(MINIMO)]))
    assert df[flag].iloc[0] == 1, f"{flag} deveria marcar ausencia"
    assert df[origem].notna().iloc[0], f"{origem} deveria ter recebido sentinela"


@pytest.mark.parametrize("origem,flag", sorted(
    {**SENTINEL_999_WITH_FLAG, **SENTINEL_NEG1_WITH_FLAG}.items()
))
def test_coluna_nula_tambem_marca_a_flag(origem, flag):
    """Present-but-null must be treated as absent. Two doors, one result."""
    df = clean_record(pd.DataFrame([dict(MINIMO, **{origem: np.nan})]))
    assert df[flag].iloc[0] == 1


@pytest.mark.parametrize("origem,flag", sorted(
    {**SENTINEL_999_WITH_FLAG, **SENTINEL_NEG1_WITH_FLAG}.items()
))
def test_valor_presente_nao_marca_a_flag_nem_e_sobrescrito(origem, flag):
    """A flag, like a guard, must stay quiet on the normal path.

    And the provided value must not be replaced by the sentinel -- that would lose real
    data."""
    df = clean_record(pd.DataFrame([dict(MINIMO, **{origem: 7.0})]))
    assert df[flag].iloc[0] == 0
    assert df[origem].iloc[0] == 7.0


def test_sentinela_999_e_neg1_vao_para_as_colunas_certas():
    """The two decision tables must not swap places: 999 preserves the 'higher = safer'
    ordering; -1 is for counters, where that ordering does not exist."""
    df = clean_record(pd.DataFrame([dict(MINIMO)]))
    for c in list(SENTINEL_999_WITH_FLAG) + SENTINEL_999_ROLLOUT:
        assert df[c].iloc[0] == 999.0, f"{c} deveria ser 999"
    for c in list(SENTINEL_NEG1_WITH_FLAG) + SENTINEL_NEG1_ROLLOUT:
        assert df[c].iloc[0] == -1.0, f"{c} deveria ser -1"


# --------------------------------------------------------- aggregated flag and derived

def test_sparse_bureau_missing_e_um_OU_sobre_as_seis():
    todas = {c: 1.0 for c in SPARSE_COLS}
    completo = clean_record(pd.DataFrame([dict(MINIMO, **todas)]))
    assert completo["sparse_bureau_missing"].iloc[0] == 0

    for c in SPARSE_COLS:
        faltando_uma = dict(MINIMO, **todas)
        faltando_uma[c] = np.nan
        df = clean_record(pd.DataFrame([faltando_uma]))
        assert df["sparse_bureau_missing"].iloc[0] == 1, f"{c} sozinha deveria ligar a flag"


def test_era_pre_2012_e_sempre_zero_para_registro_novo():
    df = clean_record(pd.DataFrame([dict(MINIMO)]))
    assert df["era_pre_2012"].iloc[0] == 0


def test_funded_amnt_e_derivado_de_loan_amnt_quando_ausente():
    df = clean_record(pd.DataFrame([dict(MINIMO)]))
    assert df["funded_amnt"].iloc[0] == MINIMO["loan_amnt"]


def test_funded_amnt_informado_nao_e_sobrescrito():
    df = clean_record(pd.DataFrame([dict(MINIMO, funded_amnt=9000.0)]))
    assert df["funded_amnt"].iloc[0] == 9000.0


@pytest.mark.parametrize("coluna", ["acc_now_delinq", "delinq_amnt", "delinq_2yrs", "pub_rec"])
def test_contadores_de_evento_raro_caem_para_zero(coluna):
    """Absence of event = 0 is the field's meaning, not a guess -- and the justification
    is written in cleaning.py itself."""
    df = clean_record(pd.DataFrame([dict(MINIMO)]))
    assert df[coluna].iloc[0] == 0.0


# ------------------------------------------------------------------ function contract

def test_nao_muta_o_dataframe_de_entrada():
    entrada = pd.DataFrame([dict(MINIMO)])
    antes = entrada.copy(deep=True)
    clean_record(entrada)
    pd.testing.assert_frame_equal(entrada, antes)


def test_aceita_dict_e_dataframe_com_o_mesmo_resultado():
    de_dict = clean_record(dict(MINIMO))
    de_frame = clean_record(pd.DataFrame([dict(MINIMO)]))
    colunas = sorted(de_dict.columns)
    assert sorted(de_frame.columns) == colunas
    pd.testing.assert_series_equal(
        de_dict[colunas].iloc[0], de_frame[colunas].iloc[0], check_names=False
    )


def test_e_idempotente():
    """score_frame calls clean_record on every request, and the drift monitor may receive
    an already-clean batch. Running it twice must change nothing."""
    uma = clean_record(pd.DataFrame([dict(MINIMO)]))
    duas = clean_record(uma)
    colunas = sorted(uma.columns)
    pd.testing.assert_frame_equal(uma[colunas], duas[colunas])


def test_coercao_numerica_nao_toca_categorica_nem_data():
    """Step 9's defensive coercion exists because an omitted Optional arrives as None and
    pandas creates an 'object' column, which XGBoost rejects. It must not run over the
    categorical or date columns -- build_features and prepare_X use .dt on the dates."""
    df = clean_record(pd.DataFrame([dict(MINIMO)]))
    assert df["home_ownership"].iloc[0] == "rent"
    assert df["purpose"].iloc[0] == "debt_consolidation"
    assert df["earliest_cr_line"].iloc[0] == "2001-08-01"
    assert df["issue_d"].iloc[0] == "2015-06-01"


def test_a_condicao_do_passo_9_nao_pode_ser_dtype_igual_object():
    """Regression guard for the pandas >= 3.0 string dtype, and why it is a separate test.

    The original condition was `df[c].dtype == object`. It was CORRECT when written and
    became wrong on its own: under pandas >= 3.0 a column of strings gets the dedicated
    StringDtype (prints as `str`), not object -- so the comparison was False and the
    coercion never ran for exactly the case it existed for. Reproduced on pandas 3.0.2.

    This test does not look at the code; it looks at the BEHAVIOUR under the new dtype. If
    someone reverts the condition to `== object`, it breaks.

    Transferable lesson, and a different one from the usual contract bug, where the
    contract promised something the code never did. Here the code did it, and stopped
    because a DEPENDENCY changed underneath it. Guarding against that takes a behaviour
    test, not code review -- no review catches it, because on the day it was written it
    was right."""
    entrada = pd.DataFrame([dict(MINIMO, revol_bal="8500")])
    assert entrada["revol_bal"].dtype != object, (
        "pandas antigo: este teste perde o sentido, mas nao fica errado"
    )
    df = clean_record(entrada)
    assert pd.api.types.is_numeric_dtype(df["revol_bal"])


def test_coluna_numerica_que_chega_como_texto_vira_numero():
    df = clean_record(pd.DataFrame([dict(MINIMO, revol_bal="8500")]))
    assert pd.api.types.is_numeric_dtype(df["revol_bal"])
    assert df["revol_bal"].iloc[0] == 8500.0


def test_valor_numerico_impossivel_vira_NaN_em_vez_de_explodir():
    """errors='coerce': dirty input becomes NaN (a clean error later) instead of a 500."""
    df = clean_record(pd.DataFrame([dict(MINIMO, revol_bal="oito mil")]))
    assert pd.isna(df["revol_bal"].iloc[0])


# ------------------------------------------------------------- 4. emp_length -> years

BRUTOS_DO_TREINO = [
    ("< 1 year", 0.0), ("1 year", 1.0), ("2 years", 2.0), ("3 years", 3.0),
    ("4 years", 4.0), ("5 years", 5.0), ("6 years", 6.0), ("7 years", 7.0),
    ("8 years", 8.0), ("9 years", 9.0), ("10+ years", 10.0),
]


@pytest.mark.parametrize("bruto,esperado", BRUTOS_DO_TREINO)
def test_parse_emp_length_reproduz_a_convencao_do_notebook_02(bruto, esperado):
    """The 11 raw values that exist in the data, with the number training produced.

    List taken from the printed output of notebook 02 itself. Any divergence here is
    train/serve skew -- the model was trained on this scale and no other."""
    assert cleaning.parse_emp_length(bruto) == esperado


def test_menos_de_um_ano_vira_zero_e_nao_um():
    """The branch that carries the whole function.

    Digit extraction on '< 1 year' would give 1.0, which is the WRONG answer and looks
    right: an applicant with less than a year of employment would be scored as having
    one. That is why the explicit branch exists and must stay ABOVE the digit path."""
    assert cleaning.parse_emp_length("< 1 year") == 0.0


@pytest.mark.parametrize("lixo", [None, float("nan"), "", "sei la", "years"])
def test_valor_que_nao_da_pra_ler_vira_ausencia_e_nao_chute(lixo):
    assert pd.isna(cleaning.parse_emp_length(lixo))


def test_REGRESSAO_informar_emp_length_desliga_a_flag_de_ausente(): 
    """Regression guard for the unconverted emp_length, and the test that describes the bug.

    Before: the API sent emp_length='10+ years', nothing converted it, emp_length_anos never
    reached the frame, step 1 flagged it missing and step 4 sentinelled it to -1. EVERY
    request was scored as 'employment length unknown' -- on a feature ranked 26th of 88
    by weight. In training only 4.355% of rows are missing; in serving it was 100%."""
    df = clean_record(pd.DataFrame([dict(MINIMO, emp_length="10+ years")]))
    assert df["emp_length_anos"].iloc[0] == 10.0
    assert df["emp_length_missing"].iloc[0] == 0


def test_sem_emp_length_continua_ausente_com_sentinela():
    """The other side: a request that does NOT send it must still fall into -1 + flag,
    which is exactly what training does with 4.355% of rows."""
    df = clean_record(pd.DataFrame([dict(MINIMO)]))
    assert df["emp_length_anos"].iloc[0] == -1.0
    assert df["emp_length_missing"].iloc[0] == 1


def test_emp_length_anos_ja_presente_nao_e_sobrescrito():
    """Parquet path: the split already carries emp_length_anos and not emp_length."""
    df = clean_record(pd.DataFrame([dict(MINIMO, emp_length_anos=7.0, emp_length="2 years")]))
    assert df["emp_length_anos"].iloc[0] == 7.0


def test_a_derivacao_acontece_ANTES_da_flag():
    """Order is the point. Deriving after step 1 would flag as missing a value that had
    just been computed -- the same bug, one line later. This test breaks if someone moves
    step 0 further down."""
    df = clean_record(pd.DataFrame([dict(MINIMO, emp_length="3 years")]))
    assert (df["emp_length_anos"].iloc[0], df["emp_length_missing"].iloc[0]) == (3.0, 0)
