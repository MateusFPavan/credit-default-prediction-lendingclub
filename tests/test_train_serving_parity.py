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
# checked in test_placeholder_sorts_before_everything.
PLACEHOLDER = "__base__"


# --------------------------------------------------------------------------- helpers

@pytest.fixture(scope="module")
def trained_cols():
    """The 90 columns in exact training order, read from the .joblib itself."""
    return _trained_columns(load_model())


def _onehot(trained_cols):
    return [c for c in trained_cols if any(c.startswith(p + "_") for p in CATEGORICAL_COLS)]


def _categories_per_column(trained_cols):
    """{categorical column: [categories with a trained column]}, derived from the artifact.

    Each column's base category does NOT appear here: drop_first removed its column at
    training time. A column with an empty list was constant in training (as
    application_type was, before it was removed)."""
    return {
        p: sorted(c[len(p) + 1:] for c in trained_cols if c.startswith(p + "_"))
        for p in CATEGORICAL_COLS
    }


def _frame_covering_the_vocabulary(trained_cols):
    """Frame containing ALL trained categories, plus the base.

    This is the condition for reproducing the training encoding: get_dummies picks the
    base among the categories PRESENT in the call, so only a frame covering the whole
    vocabulary picks the same base training picked. Row 0 carries the PLACEHOLDER in
    every column -- it is the base-category row. The other rows cycle through the named
    categories.
    """
    cats = _categories_per_column(trained_cols)
    n_rows = max([len(v) for v in cats.values()] + [0]) + 1
    rows = []
    for i in range(n_rows):
        rec = dict(BASE)
        for column, categories in cats.items():
            if i == 0 or not categories:
                rec[column] = PLACEHOLDER
            else:
                rec[column] = categories[(i - 1) % len(categories)]
        rows.append(rec)
    return pd.DataFrame(rows)


def _matrix(df, drop_first, trained_cols):
    """The matrix that reaches the model, via the same steps as score_frame."""
    d = clean_record(_normalize_dates(df.copy()))
    X = prepare_X(build_features(d), FEATURE_SET, CATEGORICAL_COLS, drop_first=drop_first)
    return X.reindex(columns=trained_cols, fill_value=False)


# ------------------------------------------------ preconditions of the scaffolding itself

def test_no_categorical_is_a_prefix_of_another():
    """If one were, _categories_per_column would silently assign columns to the wrong
    one."""
    for a in CATEGORICAL_COLS:
        for b in CATEGORICAL_COLS:
            assert a == b or not b.startswith(a + "_"), f"{a} is a prefix of {b}"


def test_placeholder_sorts_before_everything(trained_cols):
    """The PLACEHOLDER trick only works if it is the first category in sort order.

    If a retrain introduces a category that sorts before '__base__', this test fails and
    signals that the scaffolding -- not the code -- needs to change."""
    for column, categories in _categories_per_column(trained_cols).items():
        for v in categories:
            assert PLACEHOLDER < v, f"{column}: '{v}' sorts before the placeholder"


def test_frame_covers_exactly_the_training_vocabulary(trained_cols):
    """Proves the frame reproduces the training encoding, not a similar one.

    With drop_first=True (the default, which IS the training path) on this frame,
    get_dummies must produce EXACTLY the set of trained one-hot columns: none missing
    (otherwise the reindex would fill them in, and the comparison would be against an
    invented matrix) and none extra (otherwise the frame has a category training never
    saw)."""
    df = _frame_covering_the_vocabulary(trained_cols)
    d = clean_record(_normalize_dates(df.copy()))
    X = prepare_X(build_features(d), FEATURE_SET, CATEGORICAL_COLS)  # drop_first=True
    produced = {c for c in X.columns if any(c.startswith(p + "_") for p in CATEGORICAL_COLS)}
    assert produced == set(_onehot(trained_cols)), (
        f"missing: {sorted(set(_onehot(trained_cols)) - produced)} / "
        f"extra: {sorted(produced - set(_onehot(trained_cols)))}"
    )


# ------------------------------------------------------------- the train/serve parity test

def test_training_and_inference_matrices_match_row_by_row(trained_cols):
    """The assert Monitor 3 asks for, against the artifact's 90 columns.

    Left: the WHOLE frame via drop_first=True -- how notebooks 06-13 built the training
    matrix. Right: each row ALONE via drop_first=False + reindex -- how the API builds
    the matrix for a request.

    Before the drop_first=False fix, this test would have failed on every row whose
    category was not the base."""
    df = _frame_covering_the_vocabulary(trained_cols)
    X_train = _matrix(df, True, trained_cols)

    for i in range(len(df)):
        X_inf = _matrix(df.iloc[[i]], False, trained_cols)
        assert X_inf.iloc[0].tolist() == X_train.iloc[i].tolist(), (
            f"row {i} differs between training and inference. "
            f"categoricals of the row: "
            f"{ {c: df.iloc[i][c] for c in CATEGORICAL_COLS} }"
        )


def test_one_hot_block_has_the_same_dtype_in_both_matrices(trained_cols):
    """The fill_value=False lesson from the batch-dependent encoding fix.

    With fill_value=0 the VALUES matched (0 == False) and the dtype did not (int64 vs
    bool). Equal in value and different in type is a different matrix -- and that is how
    the first attempt at the fix passed by eye and failed the test."""
    df = _frame_covering_the_vocabulary(trained_cols)
    onehot = _onehot(trained_cols)
    X_train = _matrix(df, True, trained_cols)

    for i in range(len(df)):
        X_inf = _matrix(df.iloc[[i]], False, trained_cols)
        assert list(X_inf[onehot].dtypes) == list(X_train[onehot].dtypes), f"row {i}"


def test_score_of_each_row_alone_matches_the_whole_batch_score(trained_cols):
    """The same parity, one level up: on the score, not the matrix.

    Goes through score_frame, which is the path the API and the drift monitor actually
    use. Covers the case where the matrix matches and something else on the path does
    not."""
    df = _frame_covering_the_vocabulary(trained_cols)
    in_batch = score_frame(df)["probability_default"].tolist()

    for i in range(len(df)):
        alone = score_frame(df.iloc[[i]])["probability_default"].iloc[0]
        assert alone == in_batch[i], f"row {i}: {alone} alone vs {in_batch[i]} in batch"


# ------------------------------------------- named findings, kept alive as asserts

def test_application_type_stays_out_of_the_contract(trained_cols):
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
    assert not [c for c in trained_cols if c.startswith("application_type_")], (
        "the trained artifact has an application_type column -- the model was retrained "
        "with the feature added back, and the removal needs to be re-evaluated"
    )


def test_unknown_category_is_indistinguishable_from_base_in_artifact(trained_cols):
    """Documents the unknown-category gap at full scale, not on a two-column frame.

    A category training never saw produces a column outside the trained list, which the
    reindex drops -- leaving the group at zero. The BASE category produces exactly the
    same zero. The two cannot be told apart from the .joblib, which is why the first
    unseen-category warning fired on every request.

    The frozen training vocabulary (scoring._training_vocabulary) is what tells them
    apart at inference; the artifact never will, so this test must keep passing."""
    onehot = _onehot(trained_cols)
    column = next(c for c, v in _categories_per_column(trained_cols).items() if v)

    base = dict(BASE); base[column] = PLACEHOLDER
    unknown = dict(BASE); unknown[column] = "category_that_never_existed"

    X_base = _matrix(pd.DataFrame([base]), False, trained_cols)
    X_unknown = _matrix(pd.DataFrame([unknown]), False, trained_cols)

    group = [c for c in onehot if c.startswith(column + "_")]
    assert X_base[group].sum(axis=1).iloc[0] == 0
    assert X_unknown[group].sum(axis=1).iloc[0] == 0
    assert X_base.iloc[0].tolist() == X_unknown.iloc[0].tolist()
