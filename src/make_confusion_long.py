"""Reshapes reports/facts/facts_confusion.csv (wide, one row per model) into the long
2x2 form a Power BI confusion-matrix visual needs: one row per cell.

Reads the counts already in facts_confusion.csv (XGB_walkforward @ threshold 0.31,
verified against docs/FACTS.md by src/run_facts.py) rather than recomputing them from
the model - this is a pure reshape of an already-verified artifact, not a new
derivation. Project convention: positive = default = reject.

Run with: python -m src.make_confusion_long
"""
from pathlib import Path

import pandas as pd

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "facts"

# Order matters: Reject before Approve, Default before Paid within each.
CELLS = [
    {"predicted": "Reject (predict default)", "actual": "Default", "cell_type": "TP",
     "count_col": "TP_rejected_defaulted", "business_meaning": "Loss avoided"},
    {"predicted": "Reject (predict default)", "actual": "Paid", "cell_type": "FP",
     "count_col": "FP_rejected_paid", "business_meaning": "Interest forgone"},
    {"predicted": "Approve (predict pay)", "actual": "Default", "cell_type": "FN",
     "count_col": "FN_approved_defaulted", "business_meaning": "Principal lost"},
    {"predicted": "Approve (predict pay)", "actual": "Paid", "cell_type": "TN",
     "count_col": "TN_approved_paid", "business_meaning": "Correct approval"},
]


def main():
    wide = pd.read_csv(REPORTS_DIR / "facts_confusion.csv")
    row = wide[wide["model"] == "XGB_walkforward"].iloc[0]

    total = sum(int(row[c["count_col"]]) for c in CELLS)
    assert total == int(row["N"]), f"cell sum {total} != N {row['N']}"

    long_rows = []
    for c in CELLS:
        count = int(row[c["count_col"]])
        long_rows.append({
            "predicted": c["predicted"],
            "actual": c["actual"],
            "count": count,
            "pct_of_total": count / total,
            "cell_type": c["cell_type"],
            "business_meaning": c["business_meaning"],
        })

    long_df = pd.DataFrame(long_rows)

    out_path = REPORTS_DIR / "facts_confusion_long.csv"
    long_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(long_df)} rows, total={total})")
    print(long_df.to_string(index=False))


if __name__ == "__main__":
    main()
