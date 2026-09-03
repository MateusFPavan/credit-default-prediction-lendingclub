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
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import FEATURE_SET, CATEGORICAL_COLS

# official operational threshold (verify_pipeline.THRESH_XGB = 0.31)
OPERATIONAL_THRESHOLD = 0.31

# columns the pipeline uses via .dt and that an HTTP request delivers as a string
DATETIME_COLS = ("issue_d", "earliest_cr_line")

_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "xgb_final.joblib"
_CATEGORY_STATS_PATH = Path(__file__).parent / "_category_stats.json"

log = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _training_vocabulary() -> dict:
    """The training split's category vocabulary, frozen to disk (P-044).

    Closes the gap named in the module docstring: until this file existed, an unseen
    category was indistinguishable from the BASE category *from the artifact*, because
    drop_first removed the base's column at training time. That is why a warning written
    on 2026-08-30 fired on every single request and had to be removed. With the vocabulary
    frozen the distinction exists: the base IS in this list; an unseen value is not.

    Same pattern as _cleaning_stats.json, same reason: a population-dependent statistic
    cannot be recomputed at inference. Returns {} if the file is absent, so an older
    checkout degrades to the previous silent behaviour instead of crashing.
    """
    if not _CATEGORY_STATS_PATH.exists():
        return {}
    return json.loads(_CATEGORY_STATS_PATH.read_text(encoding="utf-8")).get("categories", {})


def _warn_unseen_categories(df: pd.DataFrame) -> None:
    """Logs one warning per genuinely-unseen category value. Silent otherwise.

    Silent on the happy path BY CONSTRUCTION, not by hope: every legitimate value is in
    the frozen list, base category included. That is the whole difference from the
    2026-08-30 attempt, which compared produced columns against trained columns -- and the
    base has no trained column, so application_type (zero trained columns) made every
    request warn.

    Warns instead of raising, deliberately: raising would kill the drift monitor on one
    dirty row in a batch, and the score is still produced (the value encodes as all-zeros,
    i.e. as the base category). The caller needs to know it happened; it does not need the
    batch aborted.
    """
    for coluna, conhecidas in _training_vocabulary().items():
        if coluna not in df.columns:
            continue
        novas = sorted(set(df[coluna].dropna().astype(str).unique()) - set(conhecidas))
        if novas:
            log.warning(
                "categoria nao vista no treino em %r: %s -- sera pontuada como a "
                "categoria-base. Conhecidas: %s (P-044)", coluna, novas, conhecidas,
            )


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
    _warn_unseen_categories(df)   # P-044, antes do cleaning tocar as colunas
    df = clean_record(df)  # applies notebook 03's sentinels/flags/medians
    df_feat = build_features(df)
    X = prepare_X(df_feat, FEATURE_SET, CATEGORICAL_COLS, drop_first=False)
    # fill_value=False (nao 0): as unicas colunas que o reindex ACRESCENTA sao dummies
    # ausentes -- as numericas de FEATURE_SET sempre existem, senao prepare_X ja teria
    # levantado KeyError. Preencher com False mantem o dtype bool e faz a matriz de
    # inferencia ficar identica a de treino em colunas, ordem, valores E dtype.
    X = X.reindex(columns=_trained_columns(model), fill_value=False)

    proba = model.predict_proba(X)[:, 1]

    # P-011: probabilidade nao-finita nunca e legitima e nao da pra "tratar" depois --
    # qualquer consumidor (a API, o threshold, o PSI do monitor) precisa de um numero.
    # Silencioso por medicao, nao por esperanca: sobre as 172.988 linhas do treino a
    # saida veio sem NaN, sem Inf e toda dentro de [0,1] (min 0.0009279759, max
    # 0.7852590084). Este e o guard do lado de SERVING; o lado do treino esta em
    # features.assert_matriz_finita, e a origem provavel (renda ou total_acc zerados,
    # que o contrato ainda aceita) esta em P-047.
    if not np.isfinite(proba).all():
        ruins = [int(i) for i in np.flatnonzero(~np.isfinite(proba))[:10]]
        raise ValueError(
            f"predict_proba devolveu {int((~np.isfinite(proba)).sum())} valor(es) nao "
            f"finito(s); primeiras posicoes: {ruins}. Entrada com annual_inc ou total_acc "
            "zerados produz Inf nas razoes de build_features. Ver P-011 e P-047."
        )

    out = pd.DataFrame(index=df.index)
    out["probability_default"] = proba
    out["decision"] = ["reject" if p >= threshold else "approve" for p in proba]
    return out


def score_records(records: list[dict], model=None, threshold: float = OPERATIONAL_THRESHOLD) -> list[dict]:
    """List-of-dicts version (what the API uses). One row per input record."""
    df = pd.DataFrame.from_records(records)
    scored = score_frame(df, model=model, threshold=threshold)
    return scored.to_dict(orient="records")
