"""
Monitor de drift para producao (tarefa 4.4).

Extensao fina do PSI da Fase 1 -- NAO reimplementa o motor. Reusa
fit_binnings/psi_table/compute_psi/psi_band de src.psi e o pipeline de
score de src.run_psi. O que ha de novo aqui, e so isto:

  1. drift de um LOTE de entrada (registros crus, como a API recebe) contra o
     baseline de treino, passando pelo mesmo clean_record/build_features/score
     que a Fase 2 usa -- em vez de comparar parquets estaticos entre si.
  2. drift do SCORE de saida (distribuicao das probabilidades do lote vs treino).
  3. um PISO DE AMOSTRA: abaixo de N linhas, PSI e ruido, entao o monitor recusa
     dar veredito em vez de reportar um numero espurio. O piso e medido
     empiricamente (ver o bloco de verificacao), nao chutado.

Interpretacao: PSI alto nao e sempre alarme. O projeto ja documenta (scope §8,
reports/psi/README) que initial_list_status tem PSI~0.48 por mudanca GENUINA de
politica (fractional->whole), enquanto o sentinela do rollout 2012 infla PSI de
forma ESPURIA. O monitor anota a causa quando conhecida, para separar deriva
real de artefato -- o diferencial de maturidade do Projeto 4.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import load_split, FEATURE_SET, CATEGORICAL_COLS
from src.psi import fit_binnings, psi_table, compute_psi, psi_band, STABLE_MAX, ATTENTION_MAX
from src import run_psi as _rp

# piso de amostra: preenchido pela verificacao empirica (bloco 8c). Ate la, valor
# conservador padrao; o bloco 8c confirma/ajusta e grava o valor medido aqui.
MIN_BATCH_FOR_PSI = 200  # medido (bloco 9c)

# nome da coluna de score, herdado de run_psi (nao transcrito)
SCORE_COL = getattr(_rp, "SCORE_COL", "model_score")

# causas conhecidas de PSI alto (scope §8 / reports/psi/README) -- anotacao interpretativa
KNOWN_CAUSES = {
    # deriva REAL mas esperada: mudanca de politica da plataforma (scope §8)
    "initial_list_status": "deriva GENUINA de politica (fractional->whole, 82%->41%); "
                           "sinal real, nao artefato",
    # NAO-deriva: artefato definicional do corte temporal dos splits. train/val/test
    # sao cortados por issue_d, entao PSI alto e garantido por construcao e nao e
    # acionavel (reports/psi/README documenta como 'definitional, not drift').
    "issue_d": "artefato definicional: splits sao cortados por issue_d, PSI alto por "
               "construcao, nao e deriva acionavel",
    "earliest_cr_line": "correlacionado ao corte temporal por issue_d; PSI alto e "
                        "reflexo do mesmo artefato definicional, nao deriva acionavel",
}


def _prepare_baseline():
    """
    Baseline = view CLEAN da Fase 1: train com build_features + encode_datetime_cols,
    scorado, e SO ENTAO filtrado por era_pre_2012 == 0 (mesma ordem que run_psi.py:
    train_clean_enc = train_enc[train_enc["era_pre_2012"] == 0]).

    Clean, nao raw: todo lote de producao e 100% era_pre_2012==0 (decisao do Bloco 6),
    entao comparar contra o baseline raw (30% pre-2012) inflaria o PSI das ~26 colunas
    de rollout por construcao, com zero deriva real. Clean e a comparacao com sentido.
    """
    df = load_split("train")
    df_feat = _rp.build_features(df)
    scored = _rp.score_df(_rp_model(), df_feat)
    base = _rp.encode_datetime_cols(df_feat.copy())
    if SCORE_COL not in base.columns:
        base[SCORE_COL] = scored
    # filtro CLEAN -- depois de features/encode/score, igual ao run_psi
    base = base[base["era_pre_2012"] == 0].copy()
    return base

def _rp_model():
    from src.scoring import load_model
    return load_model()


def monitor_batch(batch: pd.DataFrame, baseline: pd.DataFrame | None = None) -> dict:
    """
    Calcula PSI de um lote de registros (crus ou ja limpos) contra o baseline de
    treino, feature a feature + o score. Devolve um dict com a tabela e o veredito.

    Se o lote for menor que MIN_BATCH_FOR_PSI, retorna veredito 'insufficient_sample'
    sem numeros de PSI (que seriam ruido).
    """
    from src.scoring import score_frame
    from src.cleaning import clean_record
    from src.features import build_features

    n = len(batch)
    if n < MIN_BATCH_FOR_PSI:
        return {
            "n": n, "verdict": "insufficient_sample",
            "message": (f"lote de {n} linhas < piso de {MIN_BATCH_FOR_PSI}; "
                        "PSI seria ruido. Acumule mais registros antes de avaliar drift."),
            "table": None,
        }

    if baseline is None:
        baseline = _prepare_baseline()

    # preparar o lote pelo MESMO caminho da producao
    b = clean_record(batch.copy())
    b = build_features(b)
    b = _rp.encode_datetime_cols(b) if hasattr(_rp, "encode_datetime_cols") else b
    b[SCORE_COL] = score_frame(batch.copy())["probability_default"].to_numpy()

    feature_cols = [c for c in FEATURE_SET if c in baseline.columns and c in b.columns]
    feature_cols_with_score = feature_cols + ([SCORE_COL] if SCORE_COL in baseline.columns else [])

    binnings = fit_binnings(baseline, feature_cols_with_score, categorical_cols=CATEGORICAL_COLS)
    table = psi_table(binnings, baseline, b, feature_cols_with_score)

    # anotar causa conhecida e classificar
    if "feature" in table.columns and "cause" not in table.columns:
        table["cause"] = table["feature"].map(KNOWN_CAUSES).fillna("")

    # veredito agregado: quantas features em cada faixa
    bands = table["band"].value_counts().to_dict() if "band" in table.columns else {}
    n_critical = int((table["psi"] > ATTENTION_MAX).sum()) if "psi" in table.columns else 0
    # criticos SEM causa conhecida = os que merecem atencao real
    unexplained = table[(table["psi"] > ATTENTION_MAX) & (table["cause"] == "")] \
        if {"psi", "cause"}.issubset(table.columns) else table.iloc[0:0]

    verdict = "stable"
    if len(unexplained) > 0:
        verdict = "drift_unexplained"
    elif n_critical > 0:
        verdict = "drift_explained"  # ha PSI alto, mas com causa conhecida (ex.: politica)

    score_psi = None
    if SCORE_COL in feature_cols_with_score and "feature" in table.columns:
        row = table[table["feature"] == SCORE_COL]
        if len(row): score_psi = float(row["psi"].iloc[0])

    return {
        "n": n, "verdict": verdict, "bands": bands,
        "n_critical": n_critical, "n_unexplained": int(len(unexplained)),
        "score_psi": score_psi, "table": table,
    }
