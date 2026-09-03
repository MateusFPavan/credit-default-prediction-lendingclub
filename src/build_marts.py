"""
Build the dimensional model (star schema) from the cleaned analytical population.

    python -m src.build_marts

Reads  : data/processed/loans_clean.parquet  (+ the three split parquets, for dim_date.split)
Writes : data/marts/dim_date.csv, dim_grade.csv, dim_purpose.csv, dim_loan_profile.csv
         data/marts/fct_loan.parquet

WHY THIS EXISTS. The Power BI file shipped nine flat tables -- pre-aggregated model
results with no relationships between them (`facts_metrics`, `facts_confusion`,
`psi_quarterly`, ...). Those answer questions that were decided in advance and cannot
answer anything else: there is no way to slice default rate or profit by grade AND
vintage, because there is no grade and no vintage, only rows of finished numbers.

This module publishes the underlying grain instead, so the questions can be asked in
the tool rather than in Python.

DESIGN DECISIONS, and the reasoning is the point (see docs/DIMENSIONAL_MODEL.md):

  * The split is an ATTRIBUTE OF dim_date, not a table. This project's train/validation/
    test split is temporal, so membership is a property of the month itself. Modelling it
    as its own dimension would imply a loan could have been in a different split without
    changing its issue date, which is false here.

  * dim_loan_profile is a JUNK DIMENSION. home_ownership (4), verification_status (3),
    initial_list_status (2) and term (2) are low-cardinality flags. Four separate
    two-to-four-row dimensions would be ceremony, not modelling; Kimball's answer is one
    dimension holding the OBSERVED combinations. 46 of the 48 possible combinations occur.

  * dim_grade is at SUB-GRADE grain with `grade` as an attribute -- a hierarchy inside one
    dimension, deliberately NOT snowflaked into dim_grade -> dim_sub_grade. Snowflaking a
    35-row dimension buys nothing and costs a join.

  * There is NO geographic dimension. `addr_state` does not survive into the cleaned
    population, and inventing one would be worse than not having it.

  * The fact carries NO model predictions yet. In-sample predictions for the training
    vintages sitting in the same column as out-of-sample predictions for 2015 would be a
    trap for whoever reads the dashboard. See docs/DIMENSIONAL_MODEL.md for the plan.

The build refuses to write anything unless every foreign key resolves, the fact row count
equals the source row count, and loan_amnt reconciles to the cent.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("data/processed/loans_clean.parquet")
OUT = Path("data/marts")
SPLITS = ("train", "validation", "test")

# The junk dimension's member columns, in the order they define the sort.
JUNK_COLS = ["home_ownership", "verification_status", "initial_list_status", "term"]

# Measures kept on the fact, in the order they appear. Anything absent is skipped rather
# than assumed -- the cleaned population's column set has changed before (P-045).
MEASURES = ["loan_amnt", "funded_amnt", "int_rate", "installment", "annual_inc",
            "dti", "fico_range_low", "total_rec_prncp", "emp_length_anos", "target"]


def build_dim_date(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({"month_start": sorted(df["issue_d"].unique())})
    d.insert(0, "date_key", np.arange(1, len(d) + 1))
    d["year"] = d.month_start.dt.year
    d["quarter"] = d.month_start.dt.quarter
    d["month"] = d.month_start.dt.month
    d["year_month"] = d.month_start.dt.strftime("%Y-%m")
    d["era_pre_2012"] = (d.year < 2012).astype(int)

    d["split"] = pd.NA
    for name in SPLITS:
        p = Path(f"data/processed/{name}.parquet")
        if not p.exists():
            print(f"  split {name:<11}: parquet MISSING -- split left null for those months")
            continue
        issued = pd.read_parquet(p, columns=["issue_d"])["issue_d"]
        inside = d.month_start.between(issued.min(), issued.max())
        d.loc[inside, "split"] = name
        print(f"  split {name:<11}: {issued.min():%Y-%m} to {issued.max():%Y-%m} "
              f"({int(inside.sum())} months)")
    return d


def build_dimensions(df: pd.DataFrame):
    dim_date = build_dim_date(df)

    dim_grade = (df[["sub_grade", "grade"]].drop_duplicates()
                 .sort_values("sub_grade").reset_index(drop=True))
    dim_grade.insert(0, "grade_key", np.arange(1, len(dim_grade) + 1))
    # sub_grade sorts A1..G5, so the surrogate key doubles as the risk ordering.
    dim_grade["risk_rank"] = dim_grade["grade_key"]

    dim_purpose = pd.DataFrame({"purpose": sorted(df["purpose"].dropna().unique())})
    dim_purpose.insert(0, "purpose_key", np.arange(1, len(dim_purpose) + 1))

    dim_profile = (df[JUNK_COLS].drop_duplicates()
                   .sort_values(JUNK_COLS).reset_index(drop=True))
    dim_profile.insert(0, "profile_key", np.arange(1, len(dim_profile) + 1))
    possible = int(np.prod([df[c].nunique() for c in JUNK_COLS]))
    print(f"\n  dim_loan_profile: {len(dim_profile)} observed combinations of {possible} possible")

    return dim_date, dim_grade, dim_purpose, dim_profile


def build_fact(df, dim_date, dim_grade, dim_purpose, dim_profile) -> pd.DataFrame:
    fact = (df
            .merge(dim_date[["month_start", "date_key"]],
                   left_on="issue_d", right_on="month_start", how="left")
            .merge(dim_grade[["sub_grade", "grade_key"]], on="sub_grade", how="left")
            .merge(dim_purpose, on="purpose", how="left")
            .merge(dim_profile, on=JUNK_COLS, how="left"))

    measures = [c for c in MEASURES if c in fact.columns]
    missing = [c for c in MEASURES if c not in fact.columns]
    if missing:
        print(f"  note: measures absent from the source, skipped: {missing}")

    fact = fact[["date_key", "grade_key", "purpose_key", "profile_key"] + measures].copy()
    fact.insert(0, "loan_key", np.arange(1, len(fact) + 1))
    return fact


def verify(fact, source, dims) -> bool:
    """Every check must pass before anything is written. Silence is not success."""
    print("\n=== referential integrity (every line must read 0) ===")
    ok = True
    for key, dim in dims:
        orphans = int((~fact[key].isin(dim[key])).sum()) + int(fact[key].isna().sum())
        print(f"  {key:>13}: {orphans} orphan(s)")
        ok &= orphans == 0

    if len(fact) != len(source):
        print(f"  !!! fact has {len(fact):,} rows, source has {len(source):,} "
              f"-- a merge fanned out")
        ok = False

    delta = source.loan_amnt.sum() - fact.loan_amnt.sum()
    print(f"\n=== loan_amnt reconciliation ===\n"
          f"  source {source.loan_amnt.sum():,.2f} | fact {fact.loan_amnt.sum():,.2f} "
          f"| difference {delta:,.2f}")
    ok &= abs(delta) < 0.005
    return bool(ok)


def main() -> None:
    df = pd.read_parquet(SRC)
    print(f"source: {len(df):,} rows, {df.shape[1]} columns\n")

    dim_date, dim_grade, dim_purpose, dim_profile = build_dimensions(df)
    fact = build_fact(df, dim_date, dim_grade, dim_purpose, dim_profile)

    named = [("dim_date", dim_date, "date_key"), ("dim_grade", dim_grade, "grade_key"),
             ("dim_purpose", dim_purpose, "purpose_key"),
             ("dim_loan_profile", dim_profile, "profile_key")]

    if not verify(fact, df, [(k, d) for _, d, k in named]):
        raise SystemExit("\n*** NOTHING WRITTEN: verification failed. ***")

    OUT.mkdir(parents=True, exist_ok=True)
    for name, d, _ in named:
        d.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8")
        print(f"  wrote {name}.csv ({len(d)} rows)")
    fact.to_parquet(OUT / "fct_loan.parquet", index=False)
    print(f"  wrote fct_loan.parquet ({len(fact):,} rows, {fact.shape[1]} columns)")


if __name__ == "__main__":
    main()
