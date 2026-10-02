"""
CLI for src.monitor.monitor_batch.

Loads a batch of raw records from a CSV or parquet file, runs the drift
monitor against the training baseline, and prints the verdict. Exits with a
non-zero code on "drift_unexplained", so that a CI step can alert on it
(the retraining trigger).

Run with: python -m src.run_monitor --batch path/to/batch.csv
(or .parquet)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from src.monitor import monitor_batch


def load_batch(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(
        description="Runs the PSI drift monitor against a batch of records."
    )
    parser.add_argument(
        "--batch", required=True, type=Path,
        help="Path to a CSV or parquet file with raw records (same schema as data/processed/*.parquet).",
    )
    args = parser.parse_args()

    if not args.batch.exists():
        print(f"ERROR: batch not found at {args.batch}")
        sys.exit(2)

    batch = load_batch(args.batch)
    result = monitor_batch(batch)

    print(f"n={result['n']}  verdict={result['verdict']}")
    if "message" in result:
        print(result["message"])
    if result.get("bands"):
        print(f"bands={result['bands']}")
    if result.get("n_unexplained") is not None:
        print(f"n_unexplained={result['n_unexplained']}  score_psi={result.get('score_psi')}")
    if result.get("table") is not None:
        table = result["table"]
        unexplained = table[(table["psi"] > 0.25) & (table["cause"] == "")]
        if len(unexplained):
            print("Features with critical PSI and no known cause:")
            print(unexplained[["feature", "psi", "band"]].to_string(index=False))

    if result["verdict"] == "drift_unexplained":
        print("RETRAIN TRIGGER: drift_unexplained -> the retraining policy should fire "
              "(see docs/MODEL_CARD.md section 10).")
        sys.exit(1)
    print("OK: no retraining trigger.")


if __name__ == "__main__":
    main()
