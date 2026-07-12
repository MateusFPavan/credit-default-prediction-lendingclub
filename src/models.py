"""Frozen model configurations for the credit-default-prediction-lendingclub project.

Both builders return unfitted estimators with the exact hyperparameters selected and
frozen in notebooks 06-11 (M1) and notebook 10-11 (XGB_walkforward). Nothing here is
tuned or adjusted - changing either configuration would break reproducibility of the
final test result (notebook 12).
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def build_logistic_pipeline():
    """Build the frozen M1 pipeline: StandardScaler (to be fit on train only) + LogisticRegression.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted. Fit directly on a raw (unscaled) feature matrix; the scaler inside the
        pipeline is fit on whatever data .fit() receives, so callers must fit only on the
        training split and call .predict_proba on other splits without refitting.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000, random_state=42)),
    ])


def build_xgb_final():
    """Build the frozen XGB_walkforward configuration (winner of the notebook 10 walk-forward
    hyperparameter search, carried through calibration checks in notebook 11 and confirmed as
    final on the held-out test in notebook 12).

    n_jobs=1 is deliberate, not a default: notebook 11 found XGBoost fits are only
    bit-for-bit reproducible across runs with a fixed thread count, and the training row
    order must not be reshuffled before fitting (also established in notebook 11).

    Returns
    -------
    xgboost.XGBClassifier
        Unfitted.
    """
    return XGBClassifier(
        max_depth=8,
        learning_rate=0.03,
        n_estimators=600,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.6,
        random_state=42,
        eval_metric="logloss",
        n_jobs=1,
    )
