"""
Pure inference (scoring) logic for the Lending Club default model.

Separated from the API on purpose (BEST_PRACTICES Part F1: deterministic inference,
separate from training). Reuses the official encoding pipeline (src.features), NEVER
reimplements the encoding by hand -- reconstructing the encoding produces scores that are close but
not identical (lesson from Phase 1).

Scoring contract (the same one already used by run_facts.py and make_threshold_curve.py):
    normalize dates -> build_features(df) -> prepare_X(df, FEATURE_SET, CATEGORICAL_COLS,
    drop_first=False) -> reindex on the trained booster's columns -> predict_proba[:, 1]

    drop_first=False is REQUIRED here and is not a stylistic choice (bug P-043,
    2026-08-30). With drop_first=True the set of one-hot columns produced depends on
    which categories happen to be present in the batch; on a single-row request that is
    zero columns, and the reindex below then fills all 16 one-hot columns with 0 - so
    every applicant was scored as the base category, and the same record scored
    differently depending on batch composition. With drop_first=False the produced
    columns are named per present category and the reindex onto the trained list
    reproduces the training encoding exactly, base category included.

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

KNOWN GAP (P-044): a category never seen at training encodes as all-zeros, i.e. it is
silently scored as the base category. That is sklearn's OneHotEncoder(handle_unknown=
'ignore') behaviour, and it is NOT detectable from the artifact alone: the base category
of each column has no column either (drop_first removed it at training time), so
'unknown category' and 'base category' are indistinguishable here. A first attempt at
warning was written and removed on 2026-08-30 because it fired on every base category -
application_type in particular has ZERO trained columns, so every single request would
have warned. Telling them apart requires the training vocabulary frozen to disk, the way
_cleaning_stats.json already freezes the medians. Tracked as P-044.
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
    Deterministic AND row-independent: the score of a row does not depend on the other
    rows in the frame. Scoring a record alone and scoring it inside a batch give the same
    probability (guaranteed by drop_first=False + reindex; see the module docstring and
    tests/test_features.py::test_REGRESSAO_*).
    """
    from src.features import build_features, prepare_X  # late import: avoids a cycle
    from src.cleaning import clean_record

    if model is None:
        model = load_model()

    df = _normalize_dates(df.copy())
    df = clean_record(df)  # applies notebook 03's sentinels/flags/medians
    df_feat = build_features(df)
    X = prepare_X(df_feat, FEATURE_SET, CATEGORICAL_COLS, drop_first=False)
    # fill_value=False (nao 0): as unicas colunas que o reindex ACRESCENTA sao dummies
    # ausentes -- as numericas de FEATURE_SET sempre existem, senao prepare_X ja teria
    # levantado KeyError. Preencher com False mantem o dtype bool e faz a matriz de
    # inferencia ficar identica a de treino em colunas, ordem, valores E dtype.
    X = X.reindex(columns=_trained_columns(model), fill_value=False)

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
