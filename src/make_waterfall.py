"""Reshapes reports/facts/facts_financial.csv into the 3-row long form a Power BI
waterfall visual needs, decomposing net gain into avoided loss minus forgone interest.

Reads avoided_loss and lost_interest from the existing facts_financial.csv (XGB_walkforward
@ threshold 0.31, verified against docs/FACTS.md by src/run_facts.py) rather than
recomputing them from the model - this is a pure reshape of an already-verified artifact.
Validates avoided_loss - lost_interest == net_gain before writing.

Run with: python -m src.make_waterfall
"""
from pathlib import Path

import pandas as pd

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "facts"
TOLERANCE = 0.01  # dollars, rounding tolerance


def main():
    financial = pd.read_csv(REPORTS_DIR / "facts_financial.csv")
    row = financial[financial["model"] == "XGB_walkforward"].iloc[0]

    avoided_loss = float(row["avoided_loss"])
    lost_interest = float(row["lost_interest"])
    net_gain = float(row["net_gain"])

    recomputed_net_gain = avoided_loss - lost_interest
    diff = recomputed_net_gain - net_gain
    if abs(diff) > TOLERANCE:
        raise RuntimeError(
            f"avoided_loss - lost_interest ({recomputed_net_gain:,.2f}) != "
            f"net_gain ({net_gain:,.2f}); diff={diff:,.4f}, tol={TOLERANCE}"
        )

    waterfall_df = pd.DataFrame([
        {"step_order": 1, "category": "Avoided Loss", "value": round(avoided_loss, 2), "type": "increase"},
        {"step_order": 2, "category": "Forgone Interest", "value": round(-lost_interest, 2), "type": "decrease"},
        {"step_order": 3, "category": "Net Gain", "value": round(net_gain, 2), "type": "total"},
    ])

    out_path = REPORTS_DIR / "facts_waterfall.csv"
    waterfall_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(waterfall_df)} rows)")
    print(f"avoided_loss - lost_interest = {recomputed_net_gain:,.2f} (diff from net_gain: {diff:+.6f})")
    print(waterfall_df.to_string(index=False))


if __name__ == "__main__":
    main()
