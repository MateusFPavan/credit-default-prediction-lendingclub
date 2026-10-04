"""
Tests for src/scoring.py -- the inference path, against the real model.

They exist because of a regression in which the API accepted, validated and IGNORED
home_ownership, purpose, verification_status and initial_list_status, and the same
applicant got a different score depending on who else was in the batch (one-hot
encoding with drop_first=True at inference).

Why no existing safety net caught it, and why these tests are the right one:
  - verify_pipeline.py runs on the WHOLE train/test splits -- every category present,
    correct encoding, profit reproduced to the cent. It does not see the bug.
  - the CI smoke test checks that the response has probability_default and decision.
    Both were there. The response was syntactically perfect and semantically wrong.
  - test_api.py::test_score_valid_is_coherent checks the range 0<=p<=1 and that the
    decision agrees with the threshold. Both still held.

None of them tested SENSITIVITY: that changing a feature changes the score. That is the
only test that catches this class of bug, because the failure mode produces a legal value.

Needs no parquet: score_frame depends only on models/xgb_final.joblib and
src/_cleaning_stats.json, both versioned.

An unknown category is still scored as the base category. That is NOT solved here, and
the reason is in test_base_category_is_indistinguishable_from_unknown_in_artifact:
the two cases are indistinguishable from the .joblib alone.
Telling them apart takes the frozen training vocabulary (scoring._training_vocabulary).
"""
import pandas as pd
import pytest

from src.scoring import score_frame


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


def _p(rec: dict) -> float:
    return float(score_frame(pd.DataFrame([rec]))["probability_default"].iloc[0])


# --- sensitivity: the missing test ---

@pytest.mark.parametrize("field,a,b", [
    ("home_ownership", "rent", "own"),
    ("purpose", "debt_consolidation", "small_business"),
    ("verification_status", "verified", "not verified"),
    ("initial_list_status", "w", "f"),
])
def test_categorical_feature_affects_the_score(field, a, b):
    """Changing the categorical, with everything else equal, MUST change the probability.

    Before the encoding fix all four fields gave spread=0.0000000000 -- p=0.1341794431 for
    any value. If this test fails again, the inference encoding has regressed."""
    ra = dict(BASE); ra[field] = a
    rb = dict(BASE); rb[field] = b
    assert _p(ra) != _p(rb), (
        f"{field} does not affect the score: '{a}' and '{b}' give the same probability. "
        "That is the symptom of the batch-dependent encoding bug."
    )


# --- row independence ---

def test_score_of_a_record_is_independent_of_batch():
    """A row's score must not depend on the other rows in the frame.

    Before the encoding fix: 0.1341794431 alone vs 0.1289446801 in a batch (a difference
    of 0.0052347630). An applicant got a different decision depending on the company."""
    other = dict(BASE)
    other.update(home_ownership="own", purpose="medical",
                 initial_list_status="f", verification_status="not verified")

    alone = _p(BASE)
    in_batch = float(
        score_frame(pd.DataFrame([BASE, other]))["probability_default"].iloc[0]
    )
    assert alone == in_batch, (
        f"same record: {alone} alone vs {in_batch} in batch"
    )


def test_whole_batch_matches_row_by_row():
    """Offline analogue of ML Test Score item Monitor 3 (train/serving skew).

    Scoring N records at once must give exactly the same as scoring each one alone.
    It is the general form of the test above, and the guard against any future
    regression that makes the encoding depend on the batch's contents."""
    variants = []
    for ho in ("rent", "own", "mortgage", "other"):
        for pu in ("debt_consolidation", "medical", "car"):
            r = dict(BASE); r["home_ownership"] = ho; r["purpose"] = pu
            variants.append(r)

    in_batch = score_frame(pd.DataFrame(variants))["probability_default"].tolist()
    one_by_one = [_p(r) for r in variants]
    assert in_batch == one_by_one


# --- unknown category: known, documented gap, NOT solved ---

def test_unknown_category_scores_as_base_category():
    """KNOWN GAP, not desired behaviour.

    A category never seen in training encodes as all-dummies-zero, i.e. it is scored as
    the base category -- silently. That is scikit-learn's
    OneHotEncoder(handle_unknown='ignore') behaviour, whose default is 'error'
    precisely because silence is dangerous.

    Why the artifact cannot warn: 'unknown category' and 'base category' are
    INDISTINGUISHABLE from it. The base has no column either (drop_first removed it at
    training time). A first attempt at a warning was written and removed because it
    fired on EVERY request -- application_type had ZERO trained columns, so its only
    legitimate value ('individual') was flagged as unknown. Telling them apart requires
    the training vocabulary frozen to disk, the way _cleaning_stats.json already freezes
    the medians -- see scoring._training_vocabulary().

    This test exists so the gap is EXPLICIT, and to break if someone changes the
    behaviour without revisiting the decision."""
    unknown = dict(BASE); unknown["home_ownership"] = "nonexistent_category"
    base = dict(BASE); base["home_ownership"] = "mortgage"  # the real base category

    assert _p(unknown) == _p(base)


def test_base_category_is_indistinguishable_from_unknown_in_artifact():
    """The structural fact behind the unknown-category gap, as a test instead of a comment.

    The trained columns do not contain the base category of any categorical column.
    application_type is the extreme case: ZERO trained columns, because training had a
    single value and drop_first eliminated it. So 'individual' -- a legitimate value,
    present in every record -- does not appear in the trained list, exactly as an
    invented value would not."""
    from src.scoring import load_model, _trained_columns
    trained_cols = _trained_columns(load_model())

    assert not [c for c in trained_cols if c.startswith("application_type_")], (
        "application_type should have zero trained columns"
    )
    assert "home_ownership_mortgage" not in trained_cols, (
        "the base category of home_ownership has no column, by construction"
    )


# --- basic contract sanity ---

def test_score_is_deterministic():
    assert _p(BASE) == _p(BASE)


def test_decision_agrees_with_threshold():
    out = score_frame(pd.DataFrame([BASE]))
    p = out["probability_default"].iloc[0]
    d = out["decision"].iloc[0]
    assert (d == "reject") == (p >= 0.31)
