"""
Limpeza de registro para inferencia (porte do notebook 03 para producao).

Reproduz -- nao redecide -- as decisoes de docs/cleaning_decisions.md, para que
um registro cru (vindo da API ou de um lote de monitoramento) receba o MESMO
tratamento de missing/sentinela/flags que o pipeline de treino aplicou. A prova
de fidelidade e a verificacao bit-a-bit contra os parquets (ver o bloco de teste).

Nao toca no notebook 03 nem no run_all.py: e um porte para consumo em producao,
verificado por igualdade, nao uma segunda pipeline paralela com decisoes proprias.

Estatisticas que dependem da populacao (as 6 medianas dos esparsos) sao LIDAS de
src/_cleaning_stats.json (congeladas do train.parquet), nunca recalculadas -- um
registro novo nao tem populacao para calcular mediana, e recalcular divergiria do
treino (a classe de erro que este projeto evita).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data import CATEGORICAL_COLS

# --- tabelas de decisao (docs/cleaning_decisions.md), NAO redecididas aqui ---

# MNAR / ausencia-informativa: sentinela 999 (preserva ordenacao "maior = mais seguro")
# cada uma tem flag propria *_missing (1 se a fonte veio nula)
SENTINEL_999_WITH_FLAG = {
    "mths_since_last_delinq": "mths_since_last_delinq_missing",
    "mths_since_last_record": "mths_since_last_record_missing",
    "mths_since_recent_bc_dlq": "mths_since_recent_bc_dlq_missing",
    "mths_since_recent_revol_delinq": "mths_since_recent_revol_delinq_missing",
    "mths_since_last_major_derog": "mths_since_last_major_derog_missing",
    "mths_since_recent_inq": "mths_since_recent_inq_missing",
}
# contadores / emp_length: sentinela -1 (sem ordenacao "maior=seguro"), com flag propria
SENTINEL_NEG1_WITH_FLAG = {
    "num_tl_120dpd_2m": "num_tl_120dpd_2m_missing",
    "emp_length_anos": "emp_length_missing",
}
# rollout 2012 estrutural: sentinela -1, coberto por era_pre_2012 (SEM flag propria)
SENTINEL_NEG1_ROLLOUT = [
    "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim", "avg_cur_bal", "mort_acc",
    "acc_open_past_24mths", "total_bal_ex_mort", "total_bc_limit",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl",
    "num_accts_ever_120_pd", "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats",
    "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts",
    "num_rev_tl_bal_gt_0", "num_sats", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "tot_hi_cred_lim",
    "total_il_high_credit_limit",
]
# rollout + estrutural empilhado: sentinela 999 (preserva ordenacao), SEM flag propria
# (exceto mths_since_recent_inq, ja tratada em SENTINEL_999_WITH_FLAG)
SENTINEL_999_ROLLOUT = [
    "bc_util", "bc_open_to_buy", "percent_bc_gt_75", "mths_since_recent_bc",
    "mo_sin_old_il_acct",
]
# esparsos: flag agregada sparse_bureau_missing (OR) + mediana congelada
SPARSE_COLS = [
    "pub_rec_bankruptcies", "revol_util", "chargeoff_within_12_mths",
    "collections_12_mths_ex_med", "tax_liens", "dti",
]

_STATS = json.loads((Path(__file__).parent / "_cleaning_stats.json").read_text())
_SPARSE_MED = _STATS["sparse_medians"]


def clean_record(raw) -> pd.DataFrame:
    """
    Recebe um registro cru (dict ou DataFrame) e devolve um DataFrame limpo, com os
    sentinelas, flags e imputacoes do notebook 03 aplicados. Colunas ausentes no
    input crua sao tratadas como nulas (o cliente nao mandou aquele campo de bureau).
    """
    if isinstance(raw, dict):
        df = pd.DataFrame.from_records([raw])
    else:
        df = raw.copy()

    # 1) flags *_missing (calcular ANTES de preencher a fonte) -- 1 se fonte ausente/nula.
    #    logica explicita, sem negacao de bool: se a coluna-fonte nao existe no input,
    #    a flag e 1 para todas as linhas; se existe, a flag e 1 onde ela e nula.
    for src, flag in {**SENTINEL_999_WITH_FLAG, **SENTINEL_NEG1_WITH_FLAG}.items():
        if flag not in df.columns:
            if src in df.columns:
                df[flag] = df[src].isna().astype("int64")
            else:
                df[flag] = 1

    # 2) flag agregada dos esparsos: 1 se QUALQUER esparso ausente/nulo
    if "sparse_bureau_missing" not in df.columns:
        any_missing = pd.Series(False, index=df.index)
        for c in SPARSE_COLS:
            any_missing = any_missing | (df[c].isna() if c in df.columns else True)
        df["sparse_bureau_missing"] = any_missing.astype("int64")

    # 3) era_pre_2012: registro novo -> sempre 0 (bureau moderno sempre disponivel)
    if "era_pre_2012" not in df.columns:
        df["era_pre_2012"] = 0

    # 4) aplicar sentinelas
    for src in {**SENTINEL_999_WITH_FLAG}:
        if src not in df.columns: df[src] = 999.0
        else: df[src] = df[src].fillna(999.0)
    for src in SENTINEL_999_ROLLOUT:
        if src not in df.columns: df[src] = 999.0
        else: df[src] = df[src].fillna(999.0)
    for src in {**SENTINEL_NEG1_WITH_FLAG}:
        if src not in df.columns: df[src] = -1.0
        else: df[src] = df[src].fillna(-1.0)
    for src in SENTINEL_NEG1_ROLLOUT:
        if src not in df.columns: df[src] = -1.0
        else: df[src] = df[src].fillna(-1.0)

    # 5) esparsos: mediana congelada do treino
    for c in SPARSE_COLS:
        med = _SPARSE_MED[c]
        if c not in df.columns: df[c] = med
        else: df[c] = df[c].fillna(med)

    # 6) funded_amnt: redundante com loan_amnt (cleaning §REVISAR); derivar se ausente
    if "funded_amnt" not in df.columns and "loan_amnt" in df.columns:
        df["funded_amnt"] = df["loan_amnt"]

    # 7) contadores de evento raro sem sentinela no notebook 03 (nunca nulos no treino).
    #    Para um registro novo que os omita, default 0 -- fundamentado: sao 0 em >99,5%
    #    das linhas e forcar 0 move o score em media 0,00003 (verificado nesta sessao).
    #    Ausencia de evento = 0 e o significado do campo, nao um chute.
    for c in ("acc_now_delinq", "delinq_amnt"):
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = df[c].fillna(0.0)

    # 8) delinq_2yrs / pub_rec: default 0 quando omitidos. Fundamentado por impacto
    #    isolado (forcar 0 move o score >0,01 em apenas 1,57% e 2,23% dos casos,
    #    ~10x menor que inq_last_6mths, que por isso e obrigatorio na API, nao aqui).
    for c in ("delinq_2yrs", "pub_rec"):
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = df[c].fillna(0.0)

    # 9) coercao numerica defensiva: um Optional omitido no schema chega como None e
    #    o pandas cria coluna 'object', que o XGBoost recusa. Forca numerico onde a
    #    coluna deveria ser numerica; nao toca nas categoricas nem nas de data.
    _non_numeric = set(CATEGORICAL_COLS) | {"issue_d", "earliest_cr_line"}
    for c in df.columns:
        if c not in _non_numeric and df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df
