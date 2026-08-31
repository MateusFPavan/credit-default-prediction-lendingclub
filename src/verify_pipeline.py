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
    assert_matriz_finita(X_train, "train")   # P-011
    y_train = train_feat["target"].values

    model = build_xgb_final()
    model.fit(X_train, y_train)

    test = load_split("test")
    test_feat = build_features(test)
    X_test = prepare_X(test_feat, FEATURE_SET, CATEGORICAL_COLS)

    # P-046: este reindex e a MESMA construcao que foi o bug P-043 no caminho de serving
    # (drop_first=True + reindex com fill_value que coincide com um valor legitimo). Aqui
    # ele e seguro -- mas por propriedade do DADO, nao do codigo: o split inteiro contem
    # todas as categorias, entao get_dummies escolhe a mesma base dos dois lados e o
    # reindex nao acrescenta nem descarta nada. Medido em 2026-08-31: 0 colunas ausentes,
    # 0 descartadas. As duas linhas abaixo travam essa propriedade em vez de confiar nela;
    # se alguem rodar isto sobre um subconjunto, quebra alto em vez de produzir um profit
    # plausivel e errado.
    _faltando = [c for c in X_train.columns if c not in X_test.columns]
    _sobrando = [c for c in X_test.columns if c not in X_train.columns]
    if _faltando or _sobrando:
        raise RuntimeError(
            f"Encoding de test divergiu do de train antes do reindex. "
            f"Ausentes em test (o reindex inventaria com 0): {_faltando}; "
            f"presentes so em test (o reindex descartaria): {_sobrando}. "
            "Rodar sobre um subconjunto que nao contenha todas as categorias produz "
            "exatamente isto. Ver P-046 e P-043."
        )

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    assert_matriz_finita(X_test, "test")   # P-011
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
