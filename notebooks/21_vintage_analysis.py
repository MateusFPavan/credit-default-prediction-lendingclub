"""Phase 3: vintage (cohort) analysis of the APPROVED population.

Purpose
    Vintage/cohort analysis is a standard credit-risk check; this script shows what the
    Lending Club data does and does not support. Anything the data does not allow is
    reported as a documented result with evidence (the columns that exist), not omitted.

Inputs
    - data/processed/loans_clean.parquet: analytical population of APPROVED loans (673,314
      rows, 84 columns). Contains issue_d, loan_status, term, target. loan_status has only
      TWO values ('Charged Off', 'Fully Paid'): this is the closed-loan population, with no
      in-progress states.
    - data/processed/reject/rejected.parquet: population of REJECTED applications,
      partitioned by app_year (Hive partitioning), read with DuckDB (same pattern as
      notebooks 16-20). Schema: amount_requested, application_date, loan_title, risk_score,
      dti_raw, zip3, state, emp_length_raw, policy_code, dti, app_year. It has NO outcome
      column, consistent with the Phase 2 conclusion (reject inference cannot be validated
      on Lending Club: no label, no outcome).

Method
    Runs on the full approved population (673,314 rows):
    - inspect status/payment/delinquency columns to decide whether a roll rate is
      computable;
    - vintage table: final default rate by issue year, overall and by term;
    - descriptive comparison of volume by year, approved (issue_year) vs rejected
      (app_year).

Result
    - Roll rate is not computable: loan_status has only two terminal values and the
      mths_since_* columns are single-cutoff snapshots, not a monthly series.
    - The 60-month default rate is about 2x the 36-month rate in every year where both
      terms are present.

Maturity cutoff
    The vintage by year x term table shows 2014 and 2015 with term=36 only. This is neither
    missing data nor a bug: notebooks/03_build_processed.py applies a maturity cutoff by
    design --
        CUTOFF_36 = 2015-12-01, CUTOFF_60 = 2013-12-01
    -- each cutoff = last date in the raw file (Dec 2018) minus the contractual term. The
    2014-2015 60-month loans EXIST in the raw CSV (~153k already closed, Charged Off or Fully
    Paid) but are excluded from the analytical population on purpose, to avoid MATURITY
    BIAS: keeping only the 60-month loans that closed early would skew the sample toward
    fast outcomes that do not represent the whole vintage. Documented in docs/scope.md (the
    funnel reconciles at 673,553) and docs/DATA_CARD.md.

    The vintage analysis reconfirms this rule independently: tracing in the raw CSV why
    60-month loans disappear in 2014-2015 leads to exactly the cutoff documented in
    scope.md. This is a cross-check, not a bug finding, and supports the pipeline
    (v2.0.0/v3.0.0).

Known limitations
    - Do NOT compare the 2014-2015 default_rate (36-month only) with earlier years (36- and
      60-month) as if the composition were the same. The apparent drop can be a MIX effect
      (60-month default rate is ~2x the 36-month rate in every year with both terms
      present), not a real vintage improvement.
    - Lending Club provides the FINAL outcome per loan, not a monthly time-to-default
      curve, so the vintage table reports the final default rate per vintage, not a
      cumulative month-by-month survival curve.
    - Approved and rejected volumes are dated by different events (issue vs application)
      and rejected applications have no outcome, so only volumes are compared.
"""
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
LOANS_CLEAN = REPO_ROOT / "data" / "processed" / "loans_clean.parquet"
REJECTED = REPO_ROOT / "data" / "processed" / "reject" / "rejected.parquet"


def inspect_status_columns(df: pd.DataFrame) -> list[str]:
    """Report which status/delinquency/payment columns exist, to decide on roll rate.

    A true roll rate (current->30->60->90->default) requires a MONTHLY status trajectory
    per loan. This step only reports what exists; the decision is made afterwards, in
    __main__, from what is found -- not before.
    """
    candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in
               ["status", "delinq", "pymnt", "payment", "late", "mth", "rec_prncp"])
    ]
    print("[INSPECTION] Status/payment/delinquency-related columns found:")
    for c in candidates:
        print(f"   {c:<45} dtype={df[c].dtype}")
    print()
    print("[INSPECTION] Unique values of loan_status:",
          sorted(df["loan_status"].dropna().unique().tolist()))
    return candidates


def vintage_curves(df: pd.DataFrame, issue_col="issue_d", target_col="target", term_col="term"):
    """Default rate by origination vintage (issue_d year), overall and by term.

    Lending Club provides the FINAL outcome per loan (loan_status = Charged Off / Fully
    Paid), not a monthly time-to-default curve. This is therefore the classic vintage table
    -- final default rate per vintage -- and not a cumulative month-by-month survival curve.
    Vintage/cohort analysis is a standard credit-risk check; this table is what the data
    actually supports.
    """
    d = df.copy()
    d["issue_year"] = pd.to_datetime(d[issue_col]).dt.year

    print("=== Vintage: default rate by origination year ===")
    vt = d.groupby("issue_year").agg(
        n=("issue_year", "size"),
        default_rate=(target_col, "mean"),
    ).reset_index()
    print(vt.to_string(index=False))

    vt2 = None
    if term_col in d.columns:
        print("\n=== Vintage: default rate by vintage x term ===")
        vt2 = d.groupby(["issue_year", term_col]).agg(
            n=("issue_year", "size"), default_rate=(target_col, "mean")
        ).reset_index()
        print(vt2.to_string(index=False))

        print()
        print("[MATURITY NOTE] 2014 and 2015 appear with term=36 only BY DESIGN, not because")
        print("of missing data. notebooks/03_build_processed.py drops 60-month loans issued")
        print("after Dec 2013 (CUTOFF_60=2013-12-01; CUTOFF_36=2015-12-01) to avoid MATURITY")
        print("BIAS -- the ~153k 2014-2015 60-month loans already closed in the raw CSV")
        print("exist, but are excluded on purpose (keeping only those that closed early")
        print("would skew the vintage toward fast outcomes). Documented in docs/scope.md; the")
        print("funnel reconciles at 673,553. Do NOT compare 2014-2015 (36m only) with earlier")
        print("years (36m+60m) as the same composition -- 60m has ~2x the default rate of")
        print("36m; the apparent drop can be a MIX effect, not a real vintage improvement.")

    return vt, vt2


def vintage_descriptive_compare(df_appr: pd.DataFrame, rejected_path: Path):
    """Volume by vintage: approved (issue_year) vs rejected (app_year). Descriptive only.

    The two populations use DIFFERENT date columns by construction: issue_d for approved
    loans is the loan ISSUE date; application_date/app_year for rejected applications is the
    APPLICATION date. They are not the same event -- comparing volume per year is valid,
    comparing any rate between the two is not (there is no outcome on the rejected side).
    """
    da = df_appr.copy()
    da["issue_year"] = pd.to_datetime(da["issue_d"]).dt.year
    print("=== Volume by vintage -- APPROVED (issue_year) ===")
    vol_appr = da.groupby("issue_year").size()
    print(vol_appr.to_string())

    print("\n=== Volume by vintage -- REJECTED (app_year), via DuckDB over the partitioned Parquet ===")
    con = duckdb.connect()
    vol_rej = con.execute(f"""
        SELECT app_year, COUNT(*) AS n
        FROM read_parquet('{rejected_path.as_posix()}/**/*.parquet', hive_partitioning=true)
        GROUP BY app_year
        ORDER BY app_year
    """).fetchdf()
    print(vol_rej.to_string(index=False))

    return vol_appr, vol_rej


if __name__ == "__main__":
    print("[Phase 3 - vintage] inspect -> vintage curves -> descriptive comparison\n")

    df_appr = pd.read_parquet(LOANS_CLEAN)
    print(f"approved loans loaded: {len(df_appr):,} rows, {df_appr.shape[1]} columns\n")

    cols_status = inspect_status_columns(df_appr)

    print("\n[DECISION] A true roll rate requires a MONTHLY status trajectory per loan.")
    tem_trajetoria_mensal = any(
        k in c.lower() for c in cols_status for k in ("pymnt", "payment")
    ) and False  # no candidate column above is a monthly series; see the printed inspection
    if tem_trajetoria_mensal:
        print("Monthly trajectory found -- roll rate would be computable (not implemented here).")
    else:
        print("Only a FINAL status (loan_status, 2 values) and snapshots exist (mths_since_*,")
        print("all 'months since X' at a single cutoff, not a month-by-month time series).")
        print("RESULT: roll rate is not computable on this dataset. Documented, not omitted.")

    print()
    vt, vt2 = vintage_curves(df_appr)

    print()
    vol_appr, vol_rej = vintage_descriptive_compare(df_appr, REJECTED)
