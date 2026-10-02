"""Standalone proof that src/ is self-sufficient: trains the frozen final model using only
src.data, src.features, src.economics and src.models, and reproduces the test-set profit
reported in notebook 12 ($242,230,710.89 for XGB_walkforward at threshold 0.31).

Run with: python -m src.verify_pipeline
"""
from src.data import load_split, FEATURE_SET, CATEGORICAL_COLS
from src.economics import compute_interest_loss, profit_at_threshold
from src.features import assert_matriz_finita, build_features, prepare_X
from src.models import build_xgb_final

REFERENCE_PROFIT_XGB = 242230710.89
THRESH_XGB = 0.31


def main():
    train = load_split("train")
    train_feat = build_features(train)
    X_train = prepare_X(train_feat, FEATURE_SET, CATEGORICAL_COLS)
    assert_matriz_finita(X_train, "train")   # fail loudly on NaN/Inf
    y_train = train_feat["target"].values

    model = build_xgb_final()
    model.fit(X_train, y_train)

    test = load_split("test")
    test_feat = build_features(test)
    X_test = prepare_X(test_feat, FEATURE_SET, CATEGORICAL_COLS)

    # This reindex is the SAME construction that once broke single-record scoring on the
    # serving path (drop_first=True + reindex with a fill_value that coincides with a
    # legitimate value). Here it is safe -- but by a property of the DATA, not of the
    # code: each full split contains every category, so get_dummies picks the same base
    # on both sides and the reindex adds or drops nothing. Measured on 2026-08-31: 0
    # columns missing, 0 dropped. The check below enforces that property instead of
    # trusting it; if someone runs this on a subset, it fails loudly instead of
    # producing a plausible but wrong profit.
    _faltando = [c for c in X_train.columns if c not in X_test.columns]
    _sobrando = [c for c in X_test.columns if c not in X_train.columns]
    if _faltando or _sobrando:
        raise RuntimeError(
            f"Test encoding diverged from train before the reindex. "
            f"Missing in test (the reindex would invent them as 0): {_faltando}; "
            f"present only in test (the reindex would drop them): {_sobrando}. "
            "Running on a subset that does not contain every category produces "
            "exactly this."
        )

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    assert_matriz_finita(X_test, "test")   # fail loudly on NaN/Inf
    y_test = test_feat["target"].values

    interest_test, loss_test = compute_interest_loss(test_feat)
    y_prob_test = model.predict_proba(X_test)[:, 1]
    profit = profit_at_threshold(y_test, y_prob_test, THRESH_XGB, interest_test.values, loss_test.values)

    diff = profit - REFERENCE_PROFIT_XGB
    print(f"Profit reproduced using only src/: $ {profit:,.2f}")
    print(f"Reference profit (notebook 12): $ {REFERENCE_PROFIT_XGB:,.2f}")
    print(f"Difference: $ {diff:,.4f}")

    if abs(diff) > 0.01:
        raise RuntimeError(f"DIVERGENCE of $ {diff:,.2f} - src/ does not reproduce notebook 12.")
    print("OK: src/ is self-sufficient and reproduces the test result exactly.")


if __name__ == "__main__":
    main()
