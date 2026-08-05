"""
Logica pura de inferencia (scoring) para o modelo de default do Lending Club.

Separada da API de proposito (BOAS_PRATICAS Parte F1: inferencia deterministica,
separada do treino). Reusa o pipeline oficial de encoding (src.features), NUNCA
reimplementa o encoding a mao -- reconstruir o encoding gera scores proximos mas
nao identicos (licao da Fase 1).

Contrato de scoring (o mesmo ja usado por run_facts.py e make_threshold_curve.py):
    normalizar datas -> build_features(df) -> prepare_X(df, FEATURE_SET, CATEGORICAL_COLS)
    -> reindex nas colunas do booster treinado -> predict_proba[:, 1]

Normalizacao de datas: nos parquets do projeto, issue_d e earliest_cr_line sao
sempre datetime64. Uma requisicao HTTP entrega essas datas como string (JSON nao
tem tipo data). build_features (linha 31) e prepare_X (linha 66) usam o accessor
.dt, entao a coluna PRECISA chegar datetime nos DOIS. A conversao mora aqui, em
score_frame, e nao na API: quem consome a coluna (score_frame, via build_features
e prepare_X) e quem garante o tipo -- assim a API E o monitor de drift (4.4), que
tambem chama score_frame sobre lotes possivelmente lidos de CSV, ficam blindados
de uma vez. errors='coerce' transforma data impossivel em NaT (erro limpo depois),
em vez de estourar 500.

As colunas de treino vem de model.get_booster().feature_names, para que o
scoring dependa apenas do artefato .joblib e nao de train.parquet (gitignored).
"""
from __future__ import annotations

import functools
from pathlib import Path

import joblib
import pandas as pd

from src.data import FEATURE_SET, CATEGORICAL_COLS

# threshold operacional oficial (verify_pipeline.THRESH_XGB = 0.31)
OPERATIONAL_THRESHOLD = 0.31

# colunas que o pipeline usa via .dt e que uma requisicao HTTP entrega como string
DATETIME_COLS = ("issue_d", "earliest_cr_line")

_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "xgb_final.joblib"


@functools.lru_cache(maxsize=1)
def load_model(model_path: str | None = None):
    """Carrega o XGBoost serializado uma unica vez (cacheado)."""
    path = Path(model_path) if model_path else _MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado em {path}. "
            "Rode `python run_all.py` para gerar models/xgb_final.joblib."
        )
    return joblib.load(path)


def _trained_columns(model) -> list[str]:
    """As 90 colunas one-hot na ordem exata do treino, do proprio artefato."""
    return list(model.get_booster().feature_names)


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Garante que as colunas de data sao datetime antes do encoding."""
    for c in DATETIME_COLS:
        if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def score_frame(df: pd.DataFrame, model=None, threshold: float = OPERATIONAL_THRESHOLD) -> pd.DataFrame:
    """
    Pontua um DataFrame ja no schema cru/limpo (mesmo de data/processed/*.parquet).

    Retorna um DataFrame com colunas: probability_default, decision.
    Deterministica: mesma entrada -> mesma saida (o treino e que tem as
    pegadinhas de reprodutibilidade do XGBoost, nao a inferencia).
    """
    from src.features import build_features, prepare_X  # import tardio: evita ciclo
    from src.cleaning import clean_record

    if model is None:
        model = load_model()

    df = _normalize_dates(df.copy())
    df = clean_record(df)  # aplica sentinelas/flags/medianas do notebook 03
    df_feat = build_features(df)
    X = prepare_X(df_feat, FEATURE_SET, CATEGORICAL_COLS)
    X = X.reindex(columns=_trained_columns(model), fill_value=0)

    proba = model.predict_proba(X)[:, 1]
    out = pd.DataFrame(index=df.index)
    out["probability_default"] = proba
    out["decision"] = ["reject" if p >= threshold else "approve" for p in proba]
    return out


def score_records(records: list[dict], model=None, threshold: float = OPERATIONAL_THRESHOLD) -> list[dict]:
    """Versao lista-de-dicts (o que a API usa). Uma linha por registro de entrada."""
    df = pd.DataFrame.from_records(records)
    scored = score_frame(df, model=model, threshold=threshold)
    return scored.to_dict(orient="records")
