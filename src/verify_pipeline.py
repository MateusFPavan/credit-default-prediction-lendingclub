"""Standalone proof that src/ is self-sufficient: trains the frozen final model using only
src.data, src.features, src.economics and src.models, and reproduces the test-set profit
reported in notebook 12 ($242,230,710.89 for XGB_walkforward at threshold 0.31).

Run with: python -m src.verify_pipeline
"""
from src.data import load_split, FEATURE_SET, CATEGORICAL_COLS
from src.economics import compute_interest_loss, profit_at_threshold
from src.features import build_features, prepare_X
from src.models import build_xgb_final

REFERENCE_LUCRO_XGB = 242230710.89
THRESH_XGB = 0.31


def main():
    train = load_split("train")
    train_feat = build_features(train)
    X_train = prepare_X(train_feat, FEATURE_SET, CATEGORICAL_COLS)
    y_train = train_feat["target"].values

    model = build_xgb_final()
    model.fit(X_train, y_train)

    test = load_split("test")
    test_feat = build_features(test)
    X_test = prepare_X(test_feat, FEATURE_SET, CATEGORICAL_COLS)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    y_test = test_feat["target"].values

    interest_test, loss_test = compute_interest_loss(test_feat)
    y_prob_test = model.predict_proba(X_test)[:, 1]
    profit = profit_at_threshold(y_test, y_prob_test, THRESH_XGB, interest_test.values, loss_test.values)

    diff = profit - REFERENCE_LUCRO_XGB
    print(f"Lucro reproduzido apenas com src/: $ {profit:,.2f}")
    print(f"Lucro de referencia (notebook 12): $ {REFERENCE_LUCRO_XGB:,.2f}")
    print(f"Diferenca: $ {diff:,.4f}")

    if abs(diff) > 0.01:
        raise RuntimeError(f"DIVERGENCIA de $ {diff:,.2f} - src/ nao reproduz o notebook 12.")
    print("OK: src/ e autossuficiente e reproduz exatamente o resultado do teste.")


if __name__ == "__main__":
    main()
