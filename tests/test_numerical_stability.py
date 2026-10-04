"""
Numerical stability: NaN and Inf in the feature matrix, and the train/test encoding
divergence that verify_pipeline checks before its reindex.

Why it exists, and why there is more than one guard:

  build_features divides by annual_inc (three ratios) and by total_acc (one). A zero in
  either produces Inf or NaN. Measured on the frozen training split: 0 of 90 columns
  with NaN or Inf, 172,988 rows, and ZERO rows with annual_inc == 0 or
  total_acc == 0.

  So on the training path the assert cannot fire because of the data -- the parquet is
  frozen and was measured clean. It fires when SOMEONE CHANGES THE FEATURE CODE. That is
  what it is for.

  The path that needs a real guard is SERVING, where the input is neither frozen nor
  measured. There the guard is on the OUTPUT (a non-finite probability is never
  legitimate and cannot be handled later), and the likely source -- annual_inc or
  total_acc at zero, a region with zero training rows -- is rejected by the API contract
  (both must be > 0) but not necessarily by a batch.

  assert_finite_matrix is deliberately NOT called inside prepare_X: prepare_X is shared
  by both paths, and raising on serving would kill the drift monitor on a single dirty
  row. Same reasoning that removed the first unseen-category warning.
"""
import numpy as np
import pandas as pd
import pytest

from src.cleaning import clean_record
from src.data import FEATURE_SET, CATEGORICAL_COLS
from src.features import assert_finite_matrix, build_features, prepare_X
from src.scoring import load_model, score_frame, _normalize_dates


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


def _matrix(rec):
    d = clean_record(_normalize_dates(pd.DataFrame([rec])))
    return prepare_X(build_features(d), FEATURE_SET, CATEGORICAL_COLS, drop_first=False)


# ------------------------------------------------- the guard stays quiet on normal input

def test_matrix_of_a_valid_record_passes_silently():
    """A guard that fires on the happy path is worse than no guard at all.

    If this test fails, the finiteness assert is crying wolf and must not be committed --
    that is exactly how the first unseen-category warning died: it fired on every
    request."""
    assert_finite_matrix(_matrix(BASE), "valid record")


def test_guard_is_not_inside_prepare_X():
    """An explicit decision, kept alive as a test.

    prepare_X is shared by training and serving. If someone moves the check inside it, a
    single dirty row starts taking down the whole drift monitor. This test breaks when
    that happens."""
    rec = dict(BASE, annual_inc=0.0)
    X = _matrix(rec)                      # must not raise
    assert np.isinf(X["loan_to_income"].iloc[0])


# -------------------------------------------------------- the guard catches what it should

def test_guard_catches_NaN_and_names_the_column():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, np.nan]})
    with pytest.raises(ValueError, match=r"NaN in \['b'\]"):
        assert_finite_matrix(X)


def test_guard_catches_Inf_and_names_the_column():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, np.inf]})
    with pytest.raises(ValueError, match=r"Inf in \['b'\]"):
        assert_finite_matrix(X)


def test_guard_ignores_bool_column():
    """The 16 one-hot columns are bool. isinf on bool raises TypeError if the numeric
    filter is wrong -- this test pins the filter."""
    X = pd.DataFrame({"num": [1.0], "dummy": [True]})
    assert_finite_matrix(X)


@pytest.mark.parametrize("field,value,label", [
    ("annual_inc", 0.0, "zero income -> Inf in installment_to_income, loan_to_income and revol_bal_to_income"),
    ("total_acc", 0.0, "zero total_acc -> Inf or NaN in open_acc_ratio"),
])
def test_guard_catches_the_real_mechanism_not_just_a_synthetic_NaN(field, value, label):
    """Ties the guard to the real cause.

    Training has ZERO rows in either case (measured: 0 of 172,988). The API contract
    rejects both (annual_inc and total_acc must be > 0), but a batch scored directly --
    e.g. by the drift monitor -- does not go through it, and that is the path by which
    Inf would reach the model in production."""
    X = _matrix(dict(BASE, **{field: value}))
    with pytest.raises(ValueError):
        assert_finite_matrix(X, label)


# ------------------------------------------------------------- the serving-side guard

def test_score_of_a_valid_record_returns_a_finite_probability():
    out = score_frame(pd.DataFrame([BASE]))
    p = out["probability_default"].iloc[0]
    assert np.isfinite(p) and 0.0 <= p <= 1.0


def test_score_frame_raises_if_the_probability_comes_out_non_finite():
    """The only honest way to test a guard whose real trigger is unreachable today.

    No known input makes the model return NaN -- which is why the guard is cheap and
    silent. But 'I don't know how to trigger it' is no proof that it works, and a guard
    that is never exercised is decoration. The fake below replaces only predict_proba's
    output."""
    class _ModelThatReturnsNaN:
        def __init__(self, real):
            self._real = real

        def get_booster(self):
            return self._real.get_booster()

        def predict_proba(self, X):
            p = self._real.predict_proba(X)
            p[:, 1] = np.nan
            return p

    with pytest.raises(ValueError, match="non-finite"):
        score_frame(pd.DataFrame([BASE]), model=_ModelThatReturnsNaN(load_model()))


# ------------------------------------------ why verify_pipeline checks before reindexing

def test_drop_first_on_a_subset_produces_a_different_column_set():
    """Documents WHY verify_pipeline checks that the test encoding matches train.

    Its reindex is the same construction that once broke single-record scoring on the
    serving path. Today it is safe by a property of the DATA -- each full split contains
    every category -- not of the code. This test shows that property failing as soon as
    the frame stops covering the vocabulary: exactly the scenario the check refuses
    loudly, instead of producing a plausible but wrong profit."""
    cols = ["home_ownership"]
    complete = pd.DataFrame({"home_ownership": ["mortgage", "own", "rent"]})
    partial = pd.DataFrame({"home_ownership": ["own", "rent"]})

    X_complete = prepare_X(complete, cols, cols)          # drop_first=True: base 'mortgage'
    X_partial = prepare_X(partial, cols, cols)            # drop_first=True: base 'own'

    assert list(X_complete.columns) == ["home_ownership_own", "home_ownership_rent"]
    assert list(X_partial.columns) == ["home_ownership_rent"]
    assert set(X_complete.columns) != set(X_partial.columns), (
        "if these sets ever become equal, the reindex guard in verify_pipeline has lost its reason"
    )
