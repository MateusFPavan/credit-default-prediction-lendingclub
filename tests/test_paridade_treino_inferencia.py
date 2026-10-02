"""
TRAINING <-> INFERENCE parity against the real artifact (ML Test Score, Monitor 3).

Why this file is separate from tests/test_features.py:

  test_features.py proves row independence WITHIN inference -- the same record alone
  vs in a batch -- on synthetic frames with ONE categorical column and a hand-written
  list of trained columns. That is necessary and not sufficient.

  Monitor 3 asks something else: are the matrix TRAINING built and the matrix
  INFERENCE builds, for the SAME rows, equal element by element? Answering that takes
  the artifact's real 90 columns, because the pathological case only shows up at full
  scale -- no two-column synthetic frame reaches it.

  The case that motivated this file was application_type, with ZERO trained columns
  because it was constant in training. It was REMOVED from FEATURE_SET after an
  ablation with a difference of exactly $0.00. The reason for the file did not change:
  the BASE category of every categorical column still has no trained column, which is
  the same pathology in a form that cannot be removed.

  This comparison was first run once, in a throwaway scratch script outside the repo.
  It passed -- but the only full-scale proof lived in a file meant to be deleted. A
  test that only exists once is not a safety net.

Needs no parquet: depends only on models/xgb_final.joblib and src/_cleaning_stats.json,
both versioned. The category vocabulary is DERIVED from the artifact, never typed in --
if a retrain changes the categories, the tests adjust on their own, except the two that
guard named decisions (application_type's removal, and an unknown category being
indistinguishable from the base), which are meant to fail on purpose.
"""
import pandas as pd
import pytest

from src.cleaning import clean_record
from src.data import FEATURE_SET, CATEGORICAL_COLS
from src.features import build_features, prepare_X
from src.scoring import load_model, score_frame, _normalize_dates, _trained_columns


# Any valid record. The numeric values do not matter for encoding parity -- what
# matters is that they are the SAME in both matrices being compared.
BASE = {
    "loan_amnt": 10000.0, "installment": 325.5, "term": 36,
    "annual_inc": 60000.0, "fico_range_low": 710.0, "dti": 15.2,
    "earliest_cr_line": "2001-08-01", "issue_d": "2015-06-01",
    "home_ownership": "rent", "purpose": "debt_consolidation",
    "verification_status": "verified", "initial_list_status": "w",
    "application_type": "individual",
    "acc_open_past_24mths": 3.0, "open_acc": 8.0, "total_acc": 20.0,
    "revol_bal": 8500.0, "revol_util": 42.3, "inq_last_6mths": 1.0,
}

# Stands in for the BASE category of any column: it is not a trained category, and it
# sorts before all of them ('_' = 0x5F < 'a' = 0x61), so pd.get_dummies(drop_first=
# True) drops exactly this one -- which is what training did with the real base.
# The real base name is NOT recoverable from the artifact (the same reason an unknown
# category looks like the base), and this trick avoids needing it: it reproduces the
# STRUCTURE of the training encoding without typing any category. The ordering is
# checked in test_o_placeholder_ordena_antes_de_tudo.
PLACEHOLDER = "__base__"


# --------------------------------------------------------------------------- helpers

@pytest.fixture(scope="module")
def treinadas():
    """The 90 columns in exact training order, read from the .joblib itself."""
    return _trained_columns(load_model())


def _onehot(treinadas):
    return [c for c in treinadas if any(c.startswith(p + "_") for p in CATEGORICAL_COLS)]


def _categorias_por_coluna(treinadas):
    """{categorical column: [categories with a trained column]}, derived from the artifact.

    Each column's base category does NOT appear here: drop_first removed its column at
    training time. A column with an empty list was constant in training (as
    application_type was, before it was removed)."""
    return {
        p: sorted(c[len(p) + 1:] for c in treinadas if c.startswith(p + "_"))
        for p in CATEGORICAL_COLS
    }


def _frame_cobrindo_o_vocabulario(treinadas):
    """Frame containing ALL trained categories, plus the base.

    This is the condition for reproducing the training encoding: get_dummies picks the
    base among the categories PRESENT in the call, so only a frame covering the whole
    vocabulary picks the same base training picked. Row 0 carries the PLACEHOLDER in
    every column -- it is the base-category row. The other rows cycle through the named
    categories.
    """
    cats = _categorias_por_coluna(treinadas)
    n_linhas = max([len(v) for v in cats.values()] + [0]) + 1
    linhas = []
    for i in range(n_linhas):
        rec = dict(BASE)
        for coluna, valores in cats.items():
            if i == 0 or not valores:
                rec[coluna] = PLACEHOLDER
            else:
                rec[coluna] = valores[(i - 1) % len(valores)]
        linhas.append(rec)
    return pd.DataFrame(linhas)


def _matriz(df, drop_first, treinadas):
    """The matrix that reaches the model, via the same steps as score_frame."""
    d = clean_record(_normalize_dates(df.copy()))
    X = prepare_X(build_features(d), FEATURE_SET, CATEGORICAL_COLS, drop_first=drop_first)
    return X.reindex(columns=treinadas, fill_value=False)


# ------------------------------------------------ preconditions of the scaffolding itself

def test_nenhuma_categorica_e_prefixo_de_outra():
    """If one were, _categorias_por_coluna would silently assign columns to the wrong
    one."""
    for a in CATEGORICAL_COLS:
        for b in CATEGORICAL_COLS:
            assert a == b or not b.startswith(a + "_"), f"{a} e prefixo de {b}"


def test_o_placeholder_ordena_antes_de_tudo(treinadas):
    """The PLACEHOLDER trick only works if it is the first category in sort order.

    If a retrain introduces a category that sorts before '__base__', this test fails and
    signals that the scaffolding -- not the code -- needs to change."""
    for coluna, valores in _categorias_por_coluna(treinadas).items():
        for v in valores:
            assert PLACEHOLDER < v, f"{coluna}: '{v}' ordena antes do placeholder"


def test_o_frame_cobre_exatamente_o_vocabulario_de_treino(treinadas):
    """Proves the frame reproduces the training encoding, not a similar one.

    With drop_first=True (the default, which IS the training path) on this frame,
    get_dummies must produce EXACTLY the set of trained one-hot columns: none missing
    (otherwise the reindex would fill them in, and the comparison would be against an
    invented matrix) and none extra (otherwise the frame has a category training never
    saw)."""
    df = _frame_cobrindo_o_vocabulario(treinadas)
    d = clean_record(_normalize_dates(df.copy()))
    X = prepare_X(build_features(d), FEATURE_SET, CATEGORICAL_COLS)  # drop_first=True
    produzidas = {c for c in X.columns if any(c.startswith(p + "_") for p in CATEGORICAL_COLS)}
    assert produzidas == set(_onehot(treinadas)), (
        f"faltando: {sorted(set(_onehot(treinadas)) - produzidas)} / "
        f"sobrando: {sorted(produzidas - set(_onehot(treinadas)))}"
    )


# ------------------------------------------------------------- the train/serve parity test

def test_matriz_de_treino_e_de_inferencia_batem_linha_a_linha(treinadas):
    """The assert Monitor 3 asks for, against the artifact's 90 columns.

    Left: the WHOLE frame via drop_first=True -- how notebooks 06-13 built the training
    matrix. Right: each row ALONE via drop_first=False + reindex -- how the API builds
    the matrix for a request.

    Before the drop_first=False fix, this test would have failed on every row whose
    category was not the base."""
    df = _frame_cobrindo_o_vocabulario(treinadas)
    X_treino = _matriz(df, True, treinadas)

    for i in range(len(df)):
        X_inf = _matriz(df.iloc[[i]], False, treinadas)
        assert X_inf.iloc[0].tolist() == X_treino.iloc[i].tolist(), (
            f"linha {i} difere entre treino e inferencia. "
            f"categoricas da linha: "
            f"{ {c: df.iloc[i][c] for c in CATEGORICAL_COLS} }"
        )


def test_o_bloco_one_hot_tem_o_mesmo_dtype_nas_duas_matrizes(treinadas):
    """The fill_value=False lesson from the batch-dependent encoding fix.

    With fill_value=0 the VALUES matched (0 == False) and the dtype did not (int64 vs
    bool). Equal in value and different in type is a different matrix -- and that is how
    the first attempt at the fix passed by eye and failed the test."""
    df = _frame_cobrindo_o_vocabulario(treinadas)
    onehot = _onehot(treinadas)
    X_treino = _matriz(df, True, treinadas)

    for i in range(len(df)):
        X_inf = _matriz(df.iloc[[i]], False, treinadas)
        assert list(X_inf[onehot].dtypes) == list(X_treino[onehot].dtypes), f"linha {i}"


def test_score_de_cada_linha_sozinha_bate_com_o_score_do_lote_inteiro(treinadas):
    """The same parity, one level up: on the score, not the matrix.

    Goes through score_frame, which is the path the API and the drift monitor actually
    use. Covers the case where the matrix matches and something else on the path does
    not."""
    df = _frame_cobrindo_o_vocabulario(treinadas)
    em_lote = score_frame(df)["probability_default"].tolist()

    for i in range(len(df)):
        sozinho = score_frame(df.iloc[[i]])["probability_default"].iloc[0]
        assert sozinho == em_lote[i], f"linha {i}: {sozinho} sozinha vs {em_lote[i]} em lote"


# ------------------------------------------- named findings, kept alive as asserts

def test_application_type_continua_fora_do_contrato(treinadas):
    """Guards the DECISION to remove application_type. It used to guard the DEFECT.

    The previous version asserted that application_type had ZERO trained columns -- it
    documented an inert feature that the API contract presented as live. It anticipated
    ONE way of going stale ("if a retrain includes more than one category"), i.e. the
    finding being INVALIDATED. It did not anticipate what actually happened: the finding
    being RESOLVED. The feature was removed from FEATURE_SET, CATEGORICAL_COLS and
    ScoreRequest, and `cats["application_type"]` started raising KeyError -- the test
    became an obstacle to the very fix it existed to motivate.

    Lesson: a test that documents a finding must say what to do when the finding is
    FIXED, not only when it is disproven. The fix is exactly what is being pursued.

    The removal was not by argument: the ablation gave a difference of exactly $0.00
    with CI [$0, $0] -- an identity, not an estimate, because the feature never produced
    a column. Now the test guards the exit: re-adding it without reading why breaks here."""
    from src.data import FEATURE_SET

    assert "application_type" not in FEATURE_SET, (
        "application_type is back in FEATURE_SET. It was constant in train (zero one-hot "
        "columns) and was removed after an ablation with a $0.00 delta. "
        "See MODEL_CARD section 9 and CHANGELOG 3.1.0 before reverting."
    )
    assert "application_type" not in CATEGORICAL_COLS
    assert not [c for c in treinadas if c.startswith("application_type_")], (
        "o artefato treinado tem coluna de application_type -- o modelo foi retreinado "
        "com a feature de volta, e a remocao precisa ser reavaliada"
    )


def test_categoria_desconhecida_e_indistinguivel_da_base_no_artefato(treinadas):
    """Documents the unknown-category gap at full scale, not on a two-column frame.

    A category training never saw produces a column outside the trained list, which the
    reindex drops -- leaving the group at zero. The BASE category produces exactly the
    same zero. The two cannot be told apart from the .joblib, which is why the first
    unseen-category warning fired on every request.

    The frozen training vocabulary (scoring._training_vocabulary) is what tells them
    apart at inference; the artifact never will, so this test must keep passing."""
    onehot = _onehot(treinadas)
    coluna = next(c for c, v in _categorias_por_coluna(treinadas).items() if v)

    base = dict(BASE); base[coluna] = PLACEHOLDER
    desconhecida = dict(BASE); desconhecida[coluna] = "categoria_que_nunca_existiu"

    X_base = _matriz(pd.DataFrame([base]), False, treinadas)
    X_desc = _matriz(pd.DataFrame([desconhecida]), False, treinadas)

    grupo = [c for c in onehot if c.startswith(coluna + "_")]
    assert X_base[grupo].sum(axis=1).iloc[0] == 0
    assert X_desc[grupo].sum(axis=1).iloc[0] == 0
    assert X_base.iloc[0].tolist() == X_desc.iloc[0].tolist()
