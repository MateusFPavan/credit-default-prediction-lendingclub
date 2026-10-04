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


MINIMAL = {
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

def test_record_alone_and_in_batch_produce_the_same_row():
    """The property that once broke in the neighbouring prepare_X.

    If someone ever swaps a frozen median for `df[c].median()`, or a flag for something
    that looks at the batch, this test breaks. Without it, the failure would show up as a
    slightly different score depending on the company a record keeps -- the same failure
    mode as the batch-dependent encoding bug, and nobody notices it from the response."""
    a = dict(MINIMAL)
    b = dict(MINIMAL, annual_inc=20000.0, revol_util=90.0, dti=44.0, open_acc=2.0)

    alone = clean_record(pd.DataFrame([a]))
    in_batch = clean_record(pd.DataFrame([a, b]))

    col_names = sorted(set(alone.columns) & set(in_batch.columns))
    assert sorted(alone.columns) == sorted(in_batch.columns)
    pd.testing.assert_series_equal(
        alone[col_names].iloc[0], in_batch[col_names].iloc[0], check_names=False
    )


def test_row_order_in_batch_changes_none_of_the_rows():
    """Complements the previous test: being equal alone is not enough, a row must be
    equal in any position."""
    a = dict(MINIMAL)
    b = dict(MINIMAL, annual_inc=20000.0, dti=44.0)

    ab = clean_record(pd.DataFrame([a, b])).reset_index(drop=True)
    ba = clean_record(pd.DataFrame([b, a])).reset_index(drop=True)

    col_names = sorted(ab.columns)
    pd.testing.assert_series_equal(
        ab[col_names].iloc[0], ba[col_names].iloc[1], check_names=False
    )


# ------------------------------------------------ 2. frozen median, never recomputed

def test_imputed_median_does_not_depend_on_batch():
    """"The class of error this project avoids", in the module docstring's own words.

    Two batches with deliberately opposite distributions in the same column. If the
    median came from the batch, the two imputed values would differ."""
    low_batch = [dict(MINIMAL, revol_util=v) for v in (1.0, 2.0, 3.0)]
    high_batch = [dict(MINIMAL, revol_util=v) for v in (95.0, 96.0, 97.0)]
    missing_row = dict(MINIMAL, revol_util=np.nan)

    imputed_low = clean_record(pd.DataFrame(low_batch + [missing_row]))["revol_util"].iloc[-1]
    imputed_high = clean_record(pd.DataFrame(high_batch + [missing_row]))["revol_util"].iloc[-1]

    assert imputed_low == imputed_high


def test_imputed_value_is_exactly_the_frozen_json_value():
    """Being stable is not enough -- it has to be the training number."""
    stats = json.loads(
        (Path(cleaning.__file__).parent / "_cleaning_stats.json").read_text()
    )["sparse_medians"]

    df = clean_record(pd.DataFrame([dict(MINIMAL)]))  # no sparse column provided
    for c in SPARSE_COLS:
        assert df[c].iloc[0] == stats[c], f"{c}: {df[c].iloc[0]} != {stats[c]}"


# ------------------------------------------------- 3. flag computed BEFORE the fill

@pytest.mark.parametrize("source,flag", sorted(
    {**SENTINEL_999_WITH_FLAG, **SENTINEL_NEG1_WITH_FLAG}.items()
))
def test_absent_column_sets_flag_and_gets_sentinel(source, flag):
    """The order: the flag is computed before the source is filled.

    If a refactor inverts it, the flag becomes 0 for everyone and the missingness --
    MNAR and informative here -- disappears without breaking anything."""
    df = clean_record(pd.DataFrame([dict(MINIMAL)]))
    assert df[flag].iloc[0] == 1, f"{flag} should flag the absence"
    assert df[source].notna().iloc[0], f"{source} should have received the sentinel"


@pytest.mark.parametrize("source,flag", sorted(
    {**SENTINEL_999_WITH_FLAG, **SENTINEL_NEG1_WITH_FLAG}.items()
))
def test_null_column_also_sets_flag(source, flag):
    """Present-but-null must be treated as absent. Two doors, one result."""
    df = clean_record(pd.DataFrame([dict(MINIMAL, **{source: np.nan})]))
    assert df[flag].iloc[0] == 1


@pytest.mark.parametrize("source,flag", sorted(
    {**SENTINEL_999_WITH_FLAG, **SENTINEL_NEG1_WITH_FLAG}.items()
))
def test_present_value_neither_sets_flag_nor_is_overwritten(source, flag):
    """A flag, like a guard, must stay quiet on the normal path.

    And the provided value must not be replaced by the sentinel -- that would lose real
    data."""
    df = clean_record(pd.DataFrame([dict(MINIMAL, **{source: 7.0})]))
    assert df[flag].iloc[0] == 0
    assert df[source].iloc[0] == 7.0


def test_sentinels_999_and_neg1_go_to_the_right_columns():
    """The two decision tables must not swap places: 999 preserves the 'higher = safer'
    ordering; -1 is for counters, where that ordering does not exist."""
    df = clean_record(pd.DataFrame([dict(MINIMAL)]))
    for c in list(SENTINEL_999_WITH_FLAG) + SENTINEL_999_ROLLOUT:
        assert df[c].iloc[0] == 999.0, f"{c} should be 999"
    for c in list(SENTINEL_NEG1_WITH_FLAG) + SENTINEL_NEG1_ROLLOUT:
        assert df[c].iloc[0] == -1.0, f"{c} should be -1"


# --------------------------------------------------------- aggregated flag and derived

def test_sparse_bureau_missing_is_an_OR_over_the_six():
    all_present = {c: 1.0 for c in SPARSE_COLS}
    complete = clean_record(pd.DataFrame([dict(MINIMAL, **all_present)]))
    assert complete["sparse_bureau_missing"].iloc[0] == 0

    for c in SPARSE_COLS:
        one_missing = dict(MINIMAL, **all_present)
        one_missing[c] = np.nan
        df = clean_record(pd.DataFrame([one_missing]))
        assert df["sparse_bureau_missing"].iloc[0] == 1, f"{c} alone should turn the flag on"


def test_era_pre_2012_is_always_zero_for_a_new_record():
    df = clean_record(pd.DataFrame([dict(MINIMAL)]))
    assert df["era_pre_2012"].iloc[0] == 0


def test_funded_amnt_is_derived_from_loan_amnt_when_absent():
    df = clean_record(pd.DataFrame([dict(MINIMAL)]))
    assert df["funded_amnt"].iloc[0] == MINIMAL["loan_amnt"]


def test_provided_funded_amnt_is_not_overwritten():
    df = clean_record(pd.DataFrame([dict(MINIMAL, funded_amnt=9000.0)]))
    assert df["funded_amnt"].iloc[0] == 9000.0


@pytest.mark.parametrize("column", ["acc_now_delinq", "delinq_amnt", "delinq_2yrs", "pub_rec"])
def test_rare_event_counters_default_to_zero(column):
    """Absence of event = 0 is the field's meaning, not a guess -- and the justification
    is written in cleaning.py itself."""
    df = clean_record(pd.DataFrame([dict(MINIMAL)]))
    assert df[column].iloc[0] == 0.0


# ------------------------------------------------------------------ function contract

def test_does_not_mutate_the_input_dataframe():
    input_df = pd.DataFrame([dict(MINIMAL)])
    before = input_df.copy(deep=True)
    clean_record(input_df)
    pd.testing.assert_frame_equal(input_df, before)


def test_accepts_dict_and_dataframe_with_the_same_result():
    via_dict = clean_record(dict(MINIMAL))
    via_frame = clean_record(pd.DataFrame([dict(MINIMAL)]))
    col_names = sorted(via_dict.columns)
    assert sorted(via_frame.columns) == col_names
    pd.testing.assert_series_equal(
        via_dict[col_names].iloc[0], via_frame[col_names].iloc[0], check_names=False
    )


def test_is_idempotent():
    """score_frame calls clean_record on every request, and the drift monitor may receive
    an already-clean batch. Running it twice must change nothing."""
    once = clean_record(pd.DataFrame([dict(MINIMAL)]))
    twice = clean_record(once)
    col_names = sorted(once.columns)
    pd.testing.assert_frame_equal(once[col_names], twice[col_names])


def test_numeric_coercion_leaves_categorical_and_date_alone():
    """Step 9's defensive coercion exists because an omitted Optional arrives as None and
    pandas creates an 'object' column, which XGBoost rejects. It must not run over the
    categorical or date columns -- build_features and prepare_X use .dt on the dates."""
    df = clean_record(pd.DataFrame([dict(MINIMAL)]))
    assert df["home_ownership"].iloc[0] == "rent"
    assert df["purpose"].iloc[0] == "debt_consolidation"
    assert df["earliest_cr_line"].iloc[0] == "2001-08-01"
    assert df["issue_d"].iloc[0] == "2015-06-01"


def test_step_9_condition_cannot_be_dtype_equals_object():
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
    input_df = pd.DataFrame([dict(MINIMAL, revol_bal="8500")])
    assert input_df["revol_bal"].dtype != object, (
        "old pandas: this test loses its point, but does not become wrong"
    )
    df = clean_record(input_df)
    assert pd.api.types.is_numeric_dtype(df["revol_bal"])


def test_numeric_column_arriving_as_text_becomes_a_number():
    df = clean_record(pd.DataFrame([dict(MINIMAL, revol_bal="8500")]))
    assert pd.api.types.is_numeric_dtype(df["revol_bal"])
    assert df["revol_bal"].iloc[0] == 8500.0


def test_impossible_numeric_value_becomes_NaN_instead_of_crashing():
    """errors='coerce': dirty input becomes NaN (a clean error later) instead of a 500."""
    df = clean_record(pd.DataFrame([dict(MINIMAL, revol_bal="eight thousand")]))
    assert pd.isna(df["revol_bal"].iloc[0])


# ------------------------------------------------------------- 4. emp_length -> years

TRAINING_RAW_VALUES = [
    ("< 1 year", 0.0), ("1 year", 1.0), ("2 years", 2.0), ("3 years", 3.0),
    ("4 years", 4.0), ("5 years", 5.0), ("6 years", 6.0), ("7 years", 7.0),
    ("8 years", 8.0), ("9 years", 9.0), ("10+ years", 10.0),
]


@pytest.mark.parametrize("raw,expected", TRAINING_RAW_VALUES)
def test_parse_emp_length_reproduces_the_notebook_02_convention(raw, expected):
    """The 11 raw values that exist in the data, with the number training produced.

    List taken from the printed output of notebook 02 itself. Any divergence here is
    train/serve skew -- the model was trained on this scale and no other."""
    assert cleaning.parse_emp_length(raw) == expected


def test_less_than_one_year_becomes_zero_not_one():
    """The branch that carries the whole function.

    Digit extraction on '< 1 year' would give 1.0, which is the WRONG answer and looks
    right: an applicant with less than a year of employment would be scored as having
    one. That is why the explicit branch exists and must stay ABOVE the digit path."""
    assert cleaning.parse_emp_length("< 1 year") == 0.0


@pytest.mark.parametrize("garbage", [None, float("nan"), "", "no idea", "years"])
def test_unreadable_value_becomes_missing_not_a_guess(garbage):
    assert pd.isna(cleaning.parse_emp_length(garbage))


def test_REGRESSION_providing_emp_length_clears_the_missing_flag(): 
    """Regression guard for the unconverted emp_length, and the test that describes the bug.

    Before: the API sent emp_length='10+ years', nothing converted it, emp_length_anos never
    reached the frame, step 1 flagged it missing and step 4 sentinelled it to -1. EVERY
    request was scored as 'employment length unknown' -- on a feature ranked 26th of 88
    by weight. In training only 4.355% of rows are missing; in serving it was 100%."""
    df = clean_record(pd.DataFrame([dict(MINIMAL, emp_length="10+ years")]))
    assert df["emp_length_anos"].iloc[0] == 10.0
    assert df["emp_length_missing"].iloc[0] == 0


def test_without_emp_length_stays_missing_with_sentinel():
    """The other side: a request that does NOT send it must still fall into -1 + flag,
    which is exactly what training does with 4.355% of rows."""
    df = clean_record(pd.DataFrame([dict(MINIMAL)]))
    assert df["emp_length_anos"].iloc[0] == -1.0
    assert df["emp_length_missing"].iloc[0] == 1


def test_existing_emp_length_anos_is_not_overwritten():
    """Parquet path: the split already carries emp_length_anos and not emp_length."""
    df = clean_record(pd.DataFrame([dict(MINIMAL, emp_length_anos=7.0, emp_length="2 years")]))
    assert df["emp_length_anos"].iloc[0] == 7.0


def test_derivation_happens_BEFORE_the_flag():
    """Order is the point. Deriving after step 1 would flag as missing a value that had
    just been computed -- the same bug, one line later. This test breaks if someone moves
    step 0 further down."""
    df = clean_record(pd.DataFrame([dict(MINIMAL, emp_length="3 years")]))
    assert (df["emp_length_anos"].iloc[0], df["emp_length_missing"].iloc[0]) == (3.0, 0)
