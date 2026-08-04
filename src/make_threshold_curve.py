"""Generates the threshold -> (profit, confusion matrix) curve for XGB_walkforward on the
test set, to feed an interactive threshold slider on a web dashboard.

Recomputed directly from models/xgb_final.joblib and data/processed/{train,test}.parquet
using the exact same encoding as run_all.py / notebooks 06-13 (src.features.build_features +
src.features.prepare_X, reindexed to the training matrix's columns - see src/run_facts.py's
score_with, same pattern reused here). Never hand-typed: every number in the output JSON
comes from scoring the model and sweeping src.economics.profit_at_threshold over the grid.

Project convention: positive = default = reject. Approve when y_prob < threshold.

Before writing the JSON, validates the row at threshold 0.31 against the published numbers
in docs/FACTS.md (approval 96.24%, default-among-approved 14.05%, TP=3855, FP=6789,
FN=38234, TN=233909) and aborts with RuntimeError on any mismatch.

Run with: python -m src.make_threshold_curve
"""
import json
from pathlib import Path

import joblib
import numpy as np

from src.data import load_split, FEATURE_SET, CATEGORICAL_COLS
from src.economics import compute_interest_loss, profit_at_threshold, optimal_threshold
from src.features import build_features, prepare_X

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports" / "facts"
MODELS_DIR = REPO_ROOT / "models"

THRESHOLDS = [round(i / 100, 2) for i in range(5, 96)]  # 0.05 .. 0.95 step 0.01

# Reference values from docs/FACTS.md, threshold 0.31 (XGB_walkforward, test, N=282,787).
VALIDATION_THRESHOLD = 0.31
FACTS_REFERENCE = {
    "approval_rate": 96.24, "default_approved": 14.05,
    "TP": 3855, "FP": 6789, "FN": 38234, "TN": 233909,
}
_CHECKS = []  # (label, computed, expected, abs_tol)


def check(label, computed, expected, tol):
    _CHECKS.append((label, float(computed), float(expected), tol))


def score_with(model, df_feat, feature_columns):
    """Same encode-then-reindex pattern as src/run_facts.py's score_with."""
    X = prepare_X(df_feat, FEATURE_SET, CATEGORICAL_COLS)
    X = X.reindex(columns=feature_columns, fill_value=0)
    return model.predict_proba(X)[:, 1]


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train = build_features(load_split("train"))
    test = build_features(load_split("test"))
    X_train_columns = prepare_X(train, FEATURE_SET, CATEGORICAL_COLS).columns

    xgb_model = joblib.load(MODELS_DIR / "xgb_final.joblib")

    y_test = test["target"].values
    y_prob = score_with(xgb_model, test, X_train_columns)

    interest_test, loss_test = compute_interest_loss(test)
    interest_test = interest_test.values
    loss_test = loss_test.values

    curve = []
    for t in THRESHOLDS:
        approved = y_prob < t
        rejected = ~approved
        TP = int((rejected & (y_test == 1)).sum())
        FP = int((rejected & (y_test == 0)).sum())
        FN = int((approved & (y_test == 1)).sum())
        TN = int((approved & (y_test == 0)).sum())

        profit = profit_at_threshold(y_test, y_prob, t, interest_test, loss_test)
        n_approved = TN + FN
        approval_rate = 100.0 * n_approved / len(y_test)
        default_approved = 100.0 * FN / n_approved if n_approved > 0 else float("nan")

        curve.append({
            "t": t, "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "profit": profit, "approval_rate": approval_rate,
            "default_approved": default_approved,
        })

    # --- Validation gate against docs/FACTS.md at threshold 0.31 ---
    row_031 = next(r for r in curve if r["t"] == VALIDATION_THRESHOLD)
    check("approval_rate @0.31", row_031["approval_rate"], FACTS_REFERENCE["approval_rate"], 0.01)
    check("default_approved @0.31", row_031["default_approved"], FACTS_REFERENCE["default_approved"], 0.01)
    check("TP @0.31", row_031["TP"], FACTS_REFERENCE["TP"], 0)
    check("FP @0.31", row_031["FP"], FACTS_REFERENCE["FP"], 0)
    check("FN @0.31", row_031["FN"], FACTS_REFERENCE["FN"], 0)
    check("TN @0.31", row_031["TN"], FACTS_REFERENCE["TN"], 0)

    failures = [(l, c, e, tol) for (l, c, e, tol) in _CHECKS if abs(c - e) > tol]
    if failures:
        lines = [f"  - {l}: computed={c!r} vs FACTS.md={e!r} (diff={c - e:+.6f}, tol={tol})"
                 for (l, c, e, tol) in failures]
        raise RuntimeError(
            "DIVERGENCE from docs/FACTS.md at threshold 0.31:\n" + "\n".join(lines)
        )

    print(f"VALIDATED @ threshold 0.31 against docs/FACTS.md ({len(_CHECKS)} checks passed):")
    print(f"  approval_rate={row_031['approval_rate']:.4f}%  "
          f"default_approved={row_031['default_approved']:.4f}%  "
          f"TP={row_031['TP']} FP={row_031['FP']} FN={row_031['FN']} TN={row_031['TN']}")
    print()

    # --- Profit-maximizing threshold, cross-checked via src.economics.optimal_threshold ---
    best_row = max(curve, key=lambda r: r["profit"])
    best_t_econ, best_profit_econ = optimal_threshold(
        y_test, y_prob, interest_test, loss_test, thresholds=np.array(THRESHOLDS)
    )
    if abs(best_row["t"] - best_t_econ) > 1e-9 or abs(best_row["profit"] - best_profit_econ) > 0.01:
        raise RuntimeError(
            "DIVERGENCE: max(curve) threshold/profit does not match "
            f"economics.optimal_threshold: curve=({best_row['t']}, {best_row['profit']:,.2f}) "
            f"vs optimal_threshold=({best_t_econ}, {best_profit_econ:,.2f})"
        )
    print(f"Profit-maximizing threshold on the 0.05-0.95 grid: t={best_row['t']} "
          f"-> profit=${best_row['profit']:,.2f} "
          f"(approval_rate={best_row['approval_rate']:.2f}%, "
          f"default_approved={best_row['default_approved']:.2f}%)")
    print()

    out_path = REPORTS_DIR / "threshold_curve.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(curve, f, indent=2)
    print(f"Wrote {out_path} ({len(curve)} rows)")

    print()
    print("First 3 rows:")
    for r in curve[:3]:
        print(" ", r)
    print("Last 3 rows:")
    for r in curve[-3:]:
        print(" ", r)


if __name__ == "__main__":
    main()
