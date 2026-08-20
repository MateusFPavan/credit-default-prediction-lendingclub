"""
Pure inference (scoring) logic for the Lending Club default model.

Separated from the API on purpose (BEST_PRACTICES Part F1: deterministic inference,
separate from training). Reuses the official encoding pipeline (src.features), NEVER
reimplements the encoding by hand -- reconstructing the encoding produces scores that are close but
not identical (lesson from Phase 1).

Scoring contract (the same one already used by run_facts.py and make_threshold_curve.py):
    normalize dates -> build_features(df) -> prepare_X(df, FEATURE_SET, CATEGORICAL_COLS)
    -> reindex on the trained booster's columns -> predict_proba[:, 1]

Date normalization: in the project's parquets, issue_d and earliest_cr_line are
always datetime64. An HTTP request delivers these dates as a string (JSON has
no date type). build_features (line 31) and prepare_X (line 66) use the
.dt accessor, so the column MUST arrive as datetime in BOTH. The conversion lives here, in
score_frame, and not in the API: whoever consumes the column (score_frame, via build_features
and prepare_X) is the one that guarantees the type -- this way both the API AND the drift monitor (4.4), which
also calls score_frame on batches possibly read from CSV, are shielded in one place.
errors='coerce' turns an impossible date into NaT (a clean error later),
instead of raising a 500.

The training columns come from model.get_booster().feature_names, so that
scoring depends only on the .joblib artifact and not on train.parquet (gitignored).
"""
from __future__ import annotations

import functools
from pathlib import Path

import joblib
import pandas as pd

from src.data import FEATURE_SET, CATEGORICAL_COLS

# official operational threshold (verify_pipeline.THRESH_XGB = 0.31)
OPERATIONAL_THRESHOLD = 0.31

# columns the pipeline uses via .dt and that an HTTP request delivers as a string
DATETIME_COLS = ("issue_d", "earliest_cr_line")

_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "xgb_final.joblib"


@functools.lru_cache(maxsize=1)
def load_model(model_path: str | None = None):
    """Loads the serialized XGBoost once (cached)."""
    path = Path(model_path) if model_path else _MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. "
            "Run `python run_all.py` to generate models/xgb_final.joblib."
        )
    return joblib.load(path)


def _trained_columns(model) -> list[str]:
    """The 90 one-hot columns in the exact training order, from the artifact itself."""
    return list(model.get_booster().feature_names)


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures the date columns are datetime before encoding."""
    for c in DATETIME_COLS:
        if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def score_frame(df: pd.DataFrame, model=None, threshold: float = OPERATIONAL_THRESHOLD) -> pd.DataFrame:
    """
    Scores a DataFrame already in the raw/clean schema (same as data/processed/*.parquet).

    Returns a DataFrame with columns: probability_default, decision.
    Deterministic: same input -> same output (it's training that has the
    XGBoost reproducibility gotchas, not inference).
    """
    from src.features import build_features, prepare_X  # late import: avoids a cycle
    from src.cleaning import clean_record

    if model is None:
        model = load_model()

    df = _normalize_dates(df.copy())
    df = clean_record(df)  # applies notebook 03's sentinels/flags/medians
    df_feat = build_features(df)
    X = prepare_X(df_feat, FEATURE_SET, CATEGORICAL_COLS)
    X = X.reindex(columns=_trained_columns(model), fill_value=0)

    proba = model.predict_proba(X)[:, 1]
    out = pd.DataFrame(index=df.index)
    out["probability_default"] = proba
    out["decision"] = ["reject" if p >= threshold else "approve" for p in proba]
    return out


def score_records(records: list[dict], model=None, threshold: float = OPERATIONAL_THRESHOLD) -> list[dict]:
    """List-of-dicts version (what the API uses). One row per input record."""
    df = pd.DataFrame.from_records(records)
    scored = score_frame(df, model=model, threshold=threshold)
    return scored.to_dict(orient="records")
