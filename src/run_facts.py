"""Generates Task 4.2's data layer for the Power BI dashboard: model facts recomputed
directly from models/xgb_final.joblib (+ models/logistic_baseline.joblib) and
data/processed/{train,test}.parquet - never copied from docs/FACTS.md.

docs/FACTS.md is the checksheet, not the source: after every number below is computed
from the model and the parquets, this script diffs each one against the corresponding
value already published in docs/FACTS.md section 5 ("Final results"). A divergence beyond
rounding tolerance aborts the run with a RuntimeError naming exactly which number and by
how much - the code generates the numbers, FACTS.md only confirms they still match what
was previously verified there.

Writes 4 CSVs to reports/facts/. Run with: python -m src.run_facts
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, brier_score_loss

from src.data import load_split, FEATURE_SET, CATEGORICAL_COLS
from src.economics import compute_interest_loss, profit_at_threshold
from src.features import build_features, prepare_X

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "facts"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

THRESH_XGB = 0.31
THRESH_M1 = 0.38

# Reference values transcribed from docs/FACTS.md section 5, for verification only. The
# CSVs below are never populated from this dict - it exists solely so the recomputation
# can check itself against the numbers docs/FACTS.md already published.
FACTS_REFERENCE = {
    "xgb_auc": 0.6846, "xgb_brier": 0.1205, "xgb_profit": 242230710.89,
    "xgb_pct_approved": 96.24, "xgb_default_among_approved": 14.0492,
    "m1_auc": 0.6847, "m1_brier": 0.1210, "m1_profit": 235936408.63,
    "m1_pct_approved": 98.88, "m1_default_among_approved": 14.6184,
    "m0b_profit": 233202813.06,
    "confusion_TP": 3855, "confusion_TN": 233909, "confusion_FN": 38234, "confusion_FP": 6789,
    "net_gain": 9027897.83, "avoided_loss": 32151026.71, "lost_interest": 23123128.88,
    "grade_A_auc": 0.6477, "grade_A_default": 5.42, "grade_A_n": 70100,
    "grade_B_auc": 0.6048, "grade_B_default": 11.89, "grade_B_n": 91645,
    "grade_C_auc": 0.5965, "grade_C_default": 19.44, "grade_C_n": 77348,
    "grade_D_auc": 0.5889, "grade_D_default": 26.21, "grade_D_n": 32668,
    "grade_E_auc": 0.5818, "grade_E_default": 32.84, "grade_E_n": 9424,
    "grade_F_auc": 0.5878, "grade_F_default": 42.42, "grade_F_n": 1358,
    "grade_G_auc": 0.5854, "grade_G_default": 46.31, "grade_G_n": 244,
    "q1_auc": 0.6480, "q1_default": 19.09, "q1_income": 32845, "q1_n": 71696,
    "q2_auc": 0.6722, "q2_default": 15.56, "q2_income": 52952, "q2_n": 71580,
    "q3_auc": 0.6881, "q3_default": 13.62, "q3_income": 75406, "q3_n": 72906,
    "q4_auc": 0.6974, "q4_default": 11.01, "q4_income": 141672, "q4_n": 66605,
}

_CHECKS = []  # (label, computed, expected, abs_tol), populated as the script runs


def check(label, computed, expected, tol):
    _CHECKS.append((label, float(computed), float(expected), tol))


def score_with(model, df_feat, feature_columns):
    X = prepare_X(df_feat, FEATURE_SET, CATEGORICAL_COLS)
    X = X.reindex(columns=feature_columns, fill_value=0)
    return model.predict_proba(X)[:, 1]


def ks_statistic(y_true, y_prob):
    """Max distance between the score CDFs of goods (target=0) and bads (target=1)."""
    good = y_prob[y_true == 0]
    bad = y_prob[y_true == 1]
    return float(ks_2samp(bad, good).statistic)


def model_metrics(name, y_true, y_prob, threshold, interest, loss):
    approved = y_prob < threshold
    return {
        "model": name,
        "threshold": threshold,
        "auc_roc": roc_auc_score(y_true, y_prob),
        "ks": ks_statistic(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "pct_approved": approved.mean() * 100,
        "default_among_approved": y_true[approved].mean() * 100,
        "profit": profit_at_threshold(y_true, y_prob, threshold, interest, loss),
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train = build_features(load_split("train"))
    test = build_features(load_split("test"))
    X_train_columns = prepare_X(train, FEATURE_SET, CATEGORICAL_COLS).columns

    xgb_model = joblib.load(MODELS_DIR / "xgb_final.joblib")
    m1_model = joblib.load(MODELS_DIR / "logistic_baseline.joblib")

    y_test = test["target"].values
    y_prob_xgb = score_with(xgb_model, test, X_train_columns)
    y_prob_m1 = score_with(m1_model, test, X_train_columns)

    interest_test, loss_test = compute_interest_loss(test)
    interest_test = interest_test.values
    loss_test = loss_test.values

    # --- 1. facts_metrics.csv ---
    xgb_row = model_metrics("XGB_walkforward", y_test, y_prob_xgb, THRESH_XGB, interest_test, loss_test)
    m1_row = model_metrics("M1_logistic", y_test, y_prob_m1, THRESH_M1, interest_test, loss_test)
    metrics_df = pd.DataFrame([xgb_row, m1_row])
    metrics_df.to_csv(REPORTS_DIR / "facts_metrics.csv", index=False)

    check("XGB AUC-ROC", xgb_row["auc_roc"], FACTS_REFERENCE["xgb_auc"], 1e-4)
    check("XGB Brier", xgb_row["brier"], FACTS_REFERENCE["xgb_brier"], 1e-4)
    check("XGB pct approved", xgb_row["pct_approved"], FACTS_REFERENCE["xgb_pct_approved"], 0.01)
    check("XGB default among approved", xgb_row["default_among_approved"], FACTS_REFERENCE["xgb_default_among_approved"], 0.01)
    check("XGB profit", xgb_row["profit"], FACTS_REFERENCE["xgb_profit"], 1.0)
    check("M1 AUC-ROC", m1_row["auc_roc"], FACTS_REFERENCE["m1_auc"], 1e-4)
    check("M1 Brier", m1_row["brier"], FACTS_REFERENCE["m1_brier"], 1e-4)
    check("M1 pct approved", m1_row["pct_approved"], FACTS_REFERENCE["m1_pct_approved"], 0.01)
    check("M1 default among approved", m1_row["default_among_approved"], FACTS_REFERENCE["m1_default_among_approved"], 0.01)
    check("M1 profit", m1_row["profit"], FACTS_REFERENCE["m1_profit"], 1.0)

    # --- 2. facts_confusion.csv --- (positive = default = reject)
    approved_xgb = y_prob_xgb < THRESH_XGB
    rejected_xgb = ~approved_xgb
    TP = int((rejected_xgb & (y_test == 1)).sum())
    TN = int((approved_xgb & (y_test == 0)).sum())
    FN = int((approved_xgb & (y_test == 1)).sum())
    FP = int((rejected_xgb & (y_test == 0)).sum())
    confusion_df = pd.DataFrame([{
        "model": "XGB_walkforward", "threshold": THRESH_XGB,
        "TP_rejected_defaulted": TP, "TN_approved_paid": TN,
        "FN_approved_defaulted": FN, "FP_rejected_paid": FP,
        "N": TP + TN + FN + FP,
    }])
    confusion_df.to_csv(REPORTS_DIR / "facts_confusion.csv", index=False)

    check("Confusion TP", TP, FACTS_REFERENCE["confusion_TP"], 0)
    check("Confusion TN", TN, FACTS_REFERENCE["confusion_TN"], 0)
    check("Confusion FN", FN, FACTS_REFERENCE["confusion_FN"], 0)
    check("Confusion FP", FP, FACTS_REFERENCE["confusion_FP"], 0)

    # --- 3. facts_financial.csv ---
    profit_model = xgb_row["profit"]
    profit_m0b = float(interest_test[y_test == 0].sum() - loss_test[y_test == 1].sum())
    net_gain = profit_model - profit_m0b
    avoided_loss = float(loss_test[rejected_xgb & (y_test == 1)].sum())
    lost_interest = float(interest_test[rejected_xgb & (y_test == 0)].sum())

    financial_df = pd.DataFrame([{
        "model": "XGB_walkforward", "threshold": THRESH_XGB,
        "profit_model": profit_model, "profit_approve_all_m0b": profit_m0b,
        "net_gain": net_gain, "avoided_loss": avoided_loss, "lost_interest": lost_interest,
        "decomposition_check_avoided_minus_lost": avoided_loss - lost_interest,
    }])
    financial_df.to_csv(REPORTS_DIR / "facts_financial.csv", index=False)

    check("M0b profit", profit_m0b, FACTS_REFERENCE["m0b_profit"], 1.0)
    check("Net gain", net_gain, FACTS_REFERENCE["net_gain"], 1.0)
    check("Avoided loss", avoided_loss, FACTS_REFERENCE["avoided_loss"], 1.0)
    check("Lost interest", lost_interest, FACTS_REFERENCE["lost_interest"], 1.0)
    check("Net gain == avoided-lost decomposition", net_gain, avoided_loss - lost_interest, 0.01)

    # --- 4. facts_subgroups.csv ---
    subgroup_rows = []
    for g in sorted(test["grade"].dropna().unique()):
        mask = (test["grade"] == g).values
        auc_g = roc_auc_score(y_test[mask], y_prob_xgb[mask]) if len(np.unique(y_test[mask])) > 1 else np.nan
        subgroup_rows.append({
            "dimension": "grade", "segment": g.upper(), "n": int(mask.sum()),
            "default_rate_pct": y_test[mask].mean() * 100, "auc_roc": auc_g,
            "mean_income": np.nan,
        })

    income_quartile = pd.qcut(test["annual_inc"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    for q in income_quartile.cat.categories:
        mask = (income_quartile == q).values
        auc_q = roc_auc_score(y_test[mask], y_prob_xgb[mask])
        subgroup_rows.append({
            "dimension": "income_quartile", "segment": q, "n": int(mask.sum()),
            "default_rate_pct": y_test[mask].mean() * 100, "auc_roc": auc_q,
            "mean_income": test.loc[mask, "annual_inc"].mean(),
        })

    subgroups_df = pd.DataFrame(subgroup_rows)
    subgroups_df.to_csv(REPORTS_DIR / "facts_subgroups.csv", index=False)

    grade_lookup = {r["segment"]: r for r in subgroup_rows if r["dimension"] == "grade"}
    for g in "ABCDEFG":
        r = grade_lookup[g]
        check(f"Grade {g} AUC", r["auc_roc"], FACTS_REFERENCE[f"grade_{g}_auc"], 1e-4)
        check(f"Grade {g} default rate", r["default_rate_pct"], FACTS_REFERENCE[f"grade_{g}_default"], 0.01)
        check(f"Grade {g} N", r["n"], FACTS_REFERENCE[f"grade_{g}_n"], 0)

    q_lookup = {r["segment"]: r for r in subgroup_rows if r["dimension"] == "income_quartile"}
    for i, q in enumerate(["Q1", "Q2", "Q3", "Q4"], start=1):
        r = q_lookup[q]
        check(f"{q} AUC", r["auc_roc"], FACTS_REFERENCE[f"q{i}_auc"], 1e-4)
        check(f"{q} default rate", r["default_rate_pct"], FACTS_REFERENCE[f"q{i}_default"], 0.01)
        check(f"{q} mean income", r["mean_income"], FACTS_REFERENCE[f"q{i}_income"], 1.0)
        check(f"{q} N", r["n"], FACTS_REFERENCE[f"q{i}_n"], 0)

    # --- Verification gate ---
    failures = [(l, c, e, t) for (l, c, e, t) in _CHECKS if abs(c - e) > t]
    if failures:
        lines = [f"  - {l}: computed={c!r} vs FACTS.md={e!r} (diff={c - e:+.6f}, tol={t})"
                 for (l, c, e, t) in failures]
        raise RuntimeError("DIVERGENCE from docs/FACTS.md:\n" + "\n".join(lines))

    print(f"VERIFIED against FACTS.md ({len(_CHECKS)} checks passed)")
    print()
    print("=== CSVs written to reports/facts/ ===")
    for f in sorted(REPORTS_DIR.glob("*.csv")):
        print(" -", f.relative_to(REPORTS_DIR.parent.parent))


if __name__ == "__main__":
    main()
