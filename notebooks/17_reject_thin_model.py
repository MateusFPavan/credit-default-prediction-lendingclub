# ============================================================================
# Fase 2a - Bloco 1: thin model (features compartilhadas) + parcelling (varreduras)
# ============================================================================
# Decisoes de metodo (ja tomadas, ver docs/reject_inference_roadmap.md):
#   - thin model = features compartilhadas entre aprovados e recusados (o unico
#     alicerce possivel: recusados so tem 4 campos utilizaveis).
#   - 3 features nesta etapa: amount, dti, emp_length. Geografia fica pro 2o experimento.
#   - thin model = Logistic Regression (interpretavel; XGB nao brilha com 3 features).
#   - parcelling com DUAS varreduras (n_bands x multiplier) -- labels dos recusados
#     sao desconhecidos por construcao; mostrar sensibilidade as premissas e o ponto.
#
# Nomes reais conectados ao repo (resolvidos, nao adivinhados):
#   aprovados : src.data.load_split("train") -> loan_amnt, dti, emp_length_anos
#               (emp_length_anos ja vem com sentinela -1 p/ ausente, notebook 03).
#   recusados : data/processed/reject/rejected.parquet/app_year=*/*.parquet (Fase 1),
#               lido via DuckDB (hive_partitioning=true) -> amount_requested, dti,
#               emp_length_raw (texto "10+ years" etc., parseado aqui com a MESMA
#               funcao parse_emp_length usada no notebook 03 para os aprovados).
#
# src.economics (funcao de lucro) NAO e usado neste bloco -- so treina o thin model
# e roda a varredura de parcelling. CI-EX, Kickout e lucro vem em blocos seguintes.
#
# Bloco 1c (SUBSTITUI o 1b): tratamento de dti dos RECUSADOS por mecanismo, nao
# exclusao cega (diagnostico em scratch_diag_dti_reject.py / scratch_diag_dti_parte2.py):
#   - dti == -1 ('-1%')          -> sentinela MNAR (mesma convencao do -1 em
#                                    emp_length_anos) -> flag dti_missing + imputa mediana.
#   - dti == 100 ('100%')        -> censura a direita (>=100%, 170x o vizinho mais
#                                    proximo) -> flag dti_censored, valor mantido em 100.
#   - dti in {9999,99999,199998} -> sentinela redundante (0,3%, ja coberto pelo -1)
#                                    -> DESCARTADO.
#   - cauda 100-1000 (excl. picos) -> dado real extremo -> MANTIDO.
#   - so dti dos recusados recebe tratamento neste bloco; amount e emp_length ficam
#     como estao (cada um pelo seu mecanismo, em blocos futuros).
#
# Bloco 1d (ajuste ao 1c): dti_missing/dti_censored SAEM das features do thin model.
#   Motivo medido: essas flags sao sempre 0 nos aprovados (nao existem la), entao tem
#   variancia zero no conjunto de treino -> Logistic Regression nao tem gradiente pra
#   aprender peso nenhum (coef ficou exatamente 0.0 no 1c, confirmado). Informacao que
#   so existe nos recusados nao pode ser aprendida por um modelo treinado so nos
#   aprovados -- o canal certo pra essa informacao e a atribuicao de label (parcelling),
#   nao o modelo. As flags continuam no dataframe dos recusados como METADADO
#   (rej_flags), usadas so no parcelling: dti_censored (dti>=100%, risco extremo por
#   definicao) recebe um multiplicador ADICIONAL, varrido (nunca fixado, mesma logica
#   de honestidade das outras varreduras -- labels dos recusados sao desconhecidos).
#   Criterio de teste: se variar censored_extra nao mudar a bad rate do grupo censored,
#   a flag nao se justifica no parcelling -> volta pro caminho (b) (dti tratado basta).

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_split

REJECT_GLOB = "data/processed/reject/rejected.parquet/app_year=*/*.parquet"

SHARED = ["amount", "dti", "emp_length"]  # thin model (1d: flags saem, viram metadado)


def parse_emp_length(v):
    """Identica a notebooks/03_build_processed.ipynb (celula 9) -- mesma convencao
    usada para gerar emp_length_anos nos aprovados. Portada aqui, nao reimplementada
    de memoria: '< 1 year' -> 0, 'n years'/'n year' -> n, '10+ years' -> 10, NaN mantem NaN."""
    if pd.isna(v):
        return np.nan
    v = v.strip()
    if v == "< 1 year":
        return 0.0
    if v == "10+ years":
        return 10.0
    digits = "".join(ch for ch in v if ch.isdigit())
    return float(digits) if digits else np.nan


def treat_dti_rejected(df, col="dti"):
    """
    Tratamento de dti dos RECUSADOS por mecanismo (Bloco 1c, substitui a limpeza
    ingenua do Bloco 1b). Diagnostico medido em scratch_diag_dti_reject.py /
    scratch_diag_dti_parte2.py:
      - dti == -1  -> missing sentinela ('-1%', mesma convencao de emp_length_anos):
                      flag dti_missing=1, valor imputado (mediana dos validos).
      - dti == 9999/99999/199998 -> sentinela redundante (0,3%, ja coberto pelo -1
                      via dti_missing): linhas DESCARTADAS.
      - dti == 100 -> censura a direita ('100%', pico de ~170x a vizinhanca):
                      flag dti_censored=1, valor mantido em 100 (piso).
      - resto (incl. cauda real 100-1000) -> mantido como esta.
    Retorna df tratado (com dti_missing/dti_censored) + dict de contagens.
    """
    n0 = len(df)
    counts = {}
    redund = df[col].isin([9999.0, 99999.0, 199998.0])
    counts["descartados_9999"] = int(redund.sum())
    df = df[~redund].copy()
    is_missing = (df[col] == -1)
    counts["missing_flag_-1"] = int(is_missing.sum())
    df["dti_missing"] = is_missing.astype(int)
    is_censored = (df[col] == 100)
    counts["censored_flag_100"] = int(is_censored.sum())
    df["dti_censored"] = is_censored.astype(int)
    valid = df.loc[(df[col] > 0) & (df[col] < 100), col]
    median_valid = float(valid.median())
    counts["mediana_imputada"] = median_valid
    df.loc[is_missing, col] = median_valid
    counts["restaram"] = len(df)
    counts["removidas_total"] = n0 - len(df)
    print("[TRATAMENTO dti recusados]")
    for k, v in counts.items():
        vv = f"{v:,}" if isinstance(v, int) else f"{v:.2f}"
        print(f"  {k:<20}: {vv}")
    return df, counts


def load_approved_shared():
    """Aprovados (split 'train', 172.988 linhas, ate 2013) -> X[SHARED], y, issue_d.
    dti dos aprovados ja e saudavel (sem sentinela, p99~33) -- nao recebe tratamento."""
    df_raw = load_split("train")
    issue_d_raw = df_raw["issue_d"]
    df = pd.DataFrame({
        "amount": df_raw["loan_amnt"].astype(float),
        "dti": df_raw["dti"].astype(float),
        "emp_length": df_raw["emp_length_anos"].astype(float),  # ja sentinela -1 (notebook 03)
        "target": df_raw["target"].astype(int),
    })
    X = df[SHARED]
    y = df["target"].to_numpy()
    issue_d = issue_d_raw
    return X, y, issue_d


def load_rejected_shared():
    """Recusados (Parquet particionado da Fase 1, 27.648.741 linhas) -> X[SHARED], rej_flags.
    emp_length_raw parseado com a mesma parse_emp_length dos aprovados; sentinela -1
    para ausente, mesma convencao de emp_length_anos. dti tratado por mecanismo (Bloco 1c).
    dti_missing/dti_censored NAO entram em X (1d) -- saem como rej_flags, metadado
    usado so no parcelling."""
    con = duckdb.connect()
    rel = f"read_parquet('{REJECT_GLOB}', hive_partitioning=true)"
    df_raw = con.execute(
        f"SELECT amount_requested, dti, emp_length_raw FROM {rel}"
    ).fetchdf()
    emp_length = df_raw["emp_length_raw"].apply(parse_emp_length)
    df = pd.DataFrame({
        "amount": df_raw["amount_requested"].astype(float),
        "dti": df_raw["dti"].astype(float),
        "emp_length": emp_length.fillna(-1.0),
    })
    df, counts = treat_dti_rejected(df)
    X = df[SHARED]
    rej_flags = df[["dti_missing", "dti_censored"]].reset_index(drop=True)
    X = X.reset_index(drop=True)
    return X, rej_flags, counts


# --- Thin model: Logistic nas features compartilhadas ------------------------
def train_thin_model(X, y):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(X, y)
    return model


# --- Parcelling (1d): multiplicador base + multiplicador ADICIONAL para censored ---
def parcelling_labels_grouped(model, X_appr, y_appr, X_rej, rej_flags,
                               n_bands, base_mult, censored_extra):
    """
    Parcelling com multiplicador de bad rate ajustado por grupo.
    - base_mult: multiplicador aplicado a todos os recusados (varrido: 1.0/1.5/2.0).
    - censored_extra: multiplicador ADICIONAL para recusados com dti_censored=1
      (varrido: 1.0 = sem efeito, 1.25, 1.5). censored_extra=1.0 recupera o caminho (b).
    rej_flags: DataFrame alinhado a X_rej com coluna 'dti_censored' (0/1).
    """
    p_appr = model.predict_proba(X_appr)[:, 1]
    p_rej = model.predict_proba(X_rej)[:, 1]
    edges = np.quantile(p_appr, np.linspace(0, 1, n_bands + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    appr_band = np.digitize(p_appr, edges[1:-1])
    rej_band = np.digitize(p_rej, edges[1:-1])

    band_bad = {}
    for b in range(n_bands):
        m = appr_band == b
        band_bad[b] = y_appr[m].mean() if m.sum() > 0 else y_appr.mean()

    censored = rej_flags["dti_censored"].to_numpy()
    rng = np.random.default_rng(42)
    inferred = np.empty(len(X_rej), dtype=int)
    for i, b in enumerate(rej_band):
        mult = base_mult * (censored_extra if censored[i] == 1 else 1.0)
        p_bad = min(band_bad[b] * mult, 1.0)
        inferred[i] = int(rng.random() < p_bad)
    return inferred, band_bad


# --- Varredura tripla: faixas x base_mult x censored_extra ---------------------
def sweep_grouped(model, X_appr, y_appr, X_rej, rej_flags,
                   bands_list=(5, 10, 20),
                   base_list=(1.0, 1.5, 2.0),
                   censored_extra_list=(1.0, 1.25, 1.5)):
    """censored_extra=1.0 e o controle (equivale a NAO tratar o censored -> caminho b)."""
    print(f"{'faixas':>7}{'base':>6}{'cens_x':>8}{'bad_rate':>12}{'bad_rate_censored':>20}")
    results = {}
    cens_mask = rej_flags["dti_censored"].to_numpy() == 1
    for nb in bands_list:
        for base in base_list:
            for cx in censored_extra_list:
                inf, _ = parcelling_labels_grouped(model, X_appr, y_appr, X_rej,
                                                    rej_flags, nb, base, cx)
                overall = float(inf.mean())
                cens_rate = float(inf[cens_mask].mean()) if cens_mask.any() else float("nan")
                results[(nb, base, cx)] = (overall, cens_rate)
                print(f"{nb:>7}{base:>6}{cx:>8}{overall:>12.4f}{cens_rate:>20.4f}")
    print("\n[TESTE censored] Se a coluna bad_rate_censored NAO mudar com cens_x, "
          "a flag censored nao agrega -> voltar ao caminho (b).")
    return results


if __name__ == "__main__":
    print("[Fase 2a - Bloco 1d] thin model (3 features) + parcelling com flag roteada")

    print("\nCarregando aprovados (split 'train')...")
    X_appr, y_appr, issue_d_appr = load_approved_shared()
    print(f"  {len(X_appr):,} linhas | bad rate {y_appr.mean()*100:.2f}%")
    print(X_appr.describe())

    print("\nCarregando recusados (Parquet particionado, 27.648.741 linhas esperadas)...")
    X_rej, rej_flags, rej_counts = load_rejected_shared()
    print(f"  {len(X_rej):,} linhas | censored (dti>=100%): {int(rej_flags['dti_censored'].sum()):,}")
    print(X_rej.describe())

    print("\nTreinando thin model (Logistic Regression, 3 features -- flags fora)...")
    thin = train_thin_model(X_appr, y_appr)
    coefs = dict(zip(SHARED, thin.named_steps["clf"].coef_[0]))
    print("  Coeficientes (padronizados):", coefs)

    print("\nRodando varredura tripla de parcelling (faixas x base_mult x censored_extra)...")
    res = sweep_grouped(thin, X_appr, y_appr, X_rej, rej_flags)

    print("\n[Fase 2a - Bloco 1d] concluido.")
