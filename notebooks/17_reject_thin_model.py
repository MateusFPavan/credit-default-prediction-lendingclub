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
#
# Bloco 2: varredura estendida do base_mult ate 5.0 (literatura generica sugeria 2-5x).
#   Achado: 4.0/5.0 saturam bad_rate_censored em 1.0 e levam a bad_rate GERAL a ~70% da
#   populacao recusada inteira -- implausivel (o multiplicador e aplicado uniformemente
#   por banda, entao em valores altos estoura o teto ate nas bandas boas).
#
# Bloco 3 (CORRIGE o Bloco 2): o multiplicador deve ser CALIBRADO pela razao real entre
#   bad rates, nao varrido as cegas no range generico 1-5 da literatura (SAS Enterprise
#   Miner: bad rate aprovados 8% -> recusados 13% = fator ~1.5, calibrado pelo dado, nao
#   chutado). Aqui bad rate aprovados = 12,43% -> faixa plausivel de multiplicador fica
#   em ~1.5-2.5 (bad rate recusados 18-31%). Sem os labels reais dos recusados nao da pra
#   calibrar o fator "certo" (diferente de Hugo Lopes 2018, que tinha labels e escolheu
#   por AUC) -- so mostrar sensibilidade em torno do plausivel. Varredura corrigida:
#   base_mult em {1.0, 1.5, 2.0, 2.5, 3.0}; 4.0/5.0 ficam registrados no roadmap como
#   "testado, satura, implausivel", nao re-rodados aqui.
#
# Bloco 4: emp_length dos recusados tratado pela CONVENCAO que o projeto ja usa em
#   emp_length_anos nos aprovados (confirmado no repo: train.parquet tem emp_length_anos
#   escala 0-10 com sentinela -1, e emp_length_missing como flag companheira, 0 mismatches
#   entre as duas). Substitui o parse_emp_length ad-hoc do Bloco 1 por um mapeamento
#   explicito (EMP_MAP) que segue a mesma convencao:
#   - '< 1 year' -> 0: MANTIDO como sinal real (nao tratado como missing). Defesa: Fed da
#     Filadelfia (Jagtiani & Lam) achou tempo de emprego como o fator MAIS importante da
#     recusa no Lending Club (~88% de importancia relativa vs ~6% amount, ~5% dti) -- e
#     esperado que recusados sejam dominados por emprego curto, e' o mecanismo causal da
#     recusa, nao anomalia de formulario. Nao ha marcador de sentinela distinto pra esse
#     valor (diferente do dti, onde '-1%'/'9999%' eram prova direta de sentinela).
#   - '10+ years' -> 10: mesma codificacao dos aprovados, SEM flag de censura separada
#     (diferente do dti_censored: aqui ja existe convencao previa no projeto, entao a
#     regra e seguir o que ja existe, nao inventar uma flag nova que quebraria a simetria
#     de features com os aprovados).
#   - None -> -1 + flag emp_length_missing: mesma convencao MNAR do projeto.
#
# Bloco 6: fecha a exploracao das 4 features compartilhadas (dti, emp_length, amount,
#   state -- state fica pra quando virar feature, diagnostico em scratch_diag_state.py
#   confirmou limpo: 50 estados+DC, 22 nulos/27,6M, zero territorios/invalidos). amount
#   NAO tem mecanismo (diagnostico em scratch_diag_amount_emplen.py: sem sentinela, sem
#   pico de censura, distribuicao de numeros redondos humana normal) -- so filtro simples:
#   amount<=0 e logicamente impossivel (1.288 linhas, 0,005%), investigar mais seria
#   cerimonia, nao rigor.
#
# Bloco 8 (REMOVE censored_extra, introduzido no 1d): teste (ii) do Bloco 1d rodado em
#   notebooks/18_profit_evaluation.py (Etapa 3b, comparacao por LUCRO a taxa de aceitacao
#   fixa -- comparacao justa entre estrategias, ver Bloco 7b). Resultado: diferenca
#   com_censored - sem_censored = 0,00 em TODAS as 24 combinacoes testadas (2 base_mult x
#   3 LGD x 4 taxas). Causa medida: a taxa de aceitacao fixa aceita os de MENOR PD: os
#   censored (dti>=100%) ja sao empurrados ao pior risco pelo VALOR dti=100 (Bloco 1c) +
#   base_mult, entao nunca entram no grupo aceito -- o multiplicador extra so atua sobre
#   emprestimos ja rejeitados, nunca muda o lucro. O sinal da censura ja esta no VALOR;
#   o multiplicador em cima e redundante. Teste (i) (bad rate) tinha passado -- a
#   disciplina do projeto e nao fechar no teste facil, so no que importa (lucro).
#   Decisao: REMOVIDO. Parcelling volta a UM multiplicador (base_mult), mantendo a faixa
#   corrigida do Bloco 3 (1.0-3.0). As flags (dti_missing, dti_censored,
#   emp_length_missing) continuam no dataframe como METADADO documental -- registram o
#   mecanismo e ja atuam via o VALOR tratado -- mas nenhuma vira multiplicador no
#   parcelling. Ver docs/reject_inference_roadmap.md, decisao do Bloco 1d atualizada
#   para FECHADA.

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


EMP_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}


def treat_emp_length_rejected(df, raw_col="emp_length_raw", out_col="emp_length"):
    """
    Converte emp_length_raw (texto) para numerico 0-10, seguindo a MESMA convencao
    ja usada em emp_length_anos nos aprovados (confirmado no repo, Bloco 4):
      - '< 1 year' -> 0 (sinal real, mantido -- ver defesa no cabecalho do arquivo).
      - '1'..'9 years' -> 1..9.
      - '10+ years' -> 10 (sem flag de censura; mesma codificacao dos aprovados).
      - None/ausente -> -1 + flag emp_length_missing (convencao MNAR do projeto).
    """
    s = df[raw_col]
    mapped = s.map(EMP_MAP)
    is_missing = mapped.isna()  # None ou qualquer valor fora do mapa
    counts = {
        "lt1_year_0": int((mapped == 0).sum()),
        "10plus_10": int((mapped == 10).sum()),
        "missing_-1": int(is_missing.sum()),
        "total": len(df),
    }
    df[out_col] = mapped.fillna(-1).astype(int)
    df["emp_length_missing"] = is_missing.astype(int)
    print("[TRATAMENTO emp_length recusados]")
    for k, v in counts.items():
        print(f"  {k:<14}: {v:,}")
    unmapped = df.loc[is_missing & s.notna(), raw_col].value_counts().head()
    if len(unmapped):
        print("  [ATENCAO] valores textuais nao mapeados (verificar):")
        print(unmapped)
    return df, counts


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


def treat_amount_rejected(df, col="amount"):
    """
    amount dos recusados NAO tem mecanismo de excecao (diagnostico:
    scratch_diag_amount_emplen.py -- sem sentinela, sem censura, distribuicao normal de
    numeros redondos humanos). So filtro simples: amount<=0 e logicamente impossivel.
    """
    n0 = len(df)
    df = df[df[col] > 0].copy()
    print(f"[TRATAMENTO amount] removidas {n0 - len(df):,} linhas (amount<=0); "
          f"restaram {len(df):,}")
    return df


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
    amount filtrado (amount<=0, Bloco 6); dti tratado por mecanismo (Bloco 1c); emp_length
    tratado pela convencao dos aprovados (Bloco 4). dti_missing/dti_censored/
    emp_length_missing NAO entram em X (1d/4) -- saem como rej_flags, metadado usado so
    no parcelling."""
    con = duckdb.connect()
    rel = f"read_parquet('{REJECT_GLOB}', hive_partitioning=true)"
    df_raw = con.execute(
        f"SELECT amount_requested, dti, emp_length_raw FROM {rel}"
    ).fetchdf()
    df = pd.DataFrame({
        "amount": df_raw["amount_requested"].astype(float),
        "dti": df_raw["dti"].astype(float),
        "emp_length_raw": df_raw["emp_length_raw"],
    })
    df = treat_amount_rejected(df)
    df, dti_counts = treat_dti_rejected(df)
    df, emp_counts = treat_emp_length_rejected(df)
    X = df[SHARED]
    rej_flags = df[["dti_missing", "dti_censored", "emp_length_missing"]].reset_index(drop=True)
    X = X.reset_index(drop=True)
    return X, rej_flags, {"dti": dti_counts, "emp_length": emp_counts}


# --- Thin model: Logistic nas features compartilhadas ------------------------
def train_thin_model(X, y):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(X, y)
    return model


# --- Parcelling (Bloco 8: simplificado, censored_extra removido -- nao agregava lucro) ---
def parcelling_labels(model, X_appr, y_appr, X_rej, n_bands, base_mult):
    """
    Parcelling com UM multiplicador de bad rate, aplicado uniformemente por banda.
    - base_mult: varrido 1.0-3.0 (faixa CORRIGIDA e plausivel pra este dataset, Bloco 3
      -- calibrada pela razao real bad rate aprovados/recusados, nao um chute cego no
      range 1-5 da literatura generica. Ver docs/reject_inference_roadmap.md, secao
      Fase 2a, "correcao do multiplicador". 4.0/5.0 testados no Bloco 2, saturam e
      produzem bad_rate geral ~70% -- implausivel, mantido so como registro).
    Sem censored_extra (Bloco 8: removido, teste de lucro reprovou -- ver cabecalho).
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

    rng = np.random.default_rng(42)
    inferred = np.array([
        int(rng.random() < min(band_bad[b] * base_mult, 1.0)) for b in rej_band
    ])
    return inferred, band_bad


# --- Varredura dupla: faixas x base_mult -----------------------------------------
def sweep(model, X_appr, y_appr, X_rej,
          bands_list=(5, 10, 20), base_list=(1.0, 1.5, 2.0, 2.5, 3.0)):
    print(f"{'faixas':>7}{'base':>6}{'bad_rate_inferida':>20}")
    results = {}
    for nb in bands_list:
        for base in base_list:
            inferred, _ = parcelling_labels(model, X_appr, y_appr, X_rej, nb, base)
            rate = float(inferred.mean())
            results[(nb, base)] = rate
            print(f"{nb:>7}{base:>6}{rate:>20.4f}")
    vals = list(results.values())
    print(f"\n[ESTABILIDADE] bad rate inferida varia de {min(vals):.4f} a {max(vals):.4f} "
          f"(amplitude {max(vals)-min(vals):.4f})")
    return results


if __name__ == "__main__":
    print("[Fase 2a - Bloco 8] parcelling simplificado (censored_extra removido, nao agregava lucro)")

    print("\nCarregando aprovados (split 'train')...")
    X_appr, y_appr, issue_d_appr = load_approved_shared()
    print(f"  {len(X_appr):,} linhas | bad rate {y_appr.mean()*100:.2f}%")
    print(X_appr.describe())

    print("\nCarregando recusados (Parquet particionado, 27.648.741 linhas esperadas)...")
    X_rej, rej_flags, rej_counts = load_rejected_shared()
    print(f"  {len(X_rej):,} linhas | censored (dti>=100%): {int(rej_flags['dti_censored'].sum()):,}"
          f" | emp_length_missing: {int(rej_flags['emp_length_missing'].sum()):,}"
          f" (flags mantidas como metadado, nao roteadas no parcelling -- Bloco 8)")
    print(X_rej.describe())

    print("\nTreinando thin model (Logistic Regression, 3 features -- flags fora)...")
    thin = train_thin_model(X_appr, y_appr)
    coefs = dict(zip(SHARED, thin.named_steps["clf"].coef_[0]))
    print("  Coeficientes (padronizados):", coefs)

    print("\nRodando varredura de parcelling (faixas x base_mult, sem censored_extra)...")
    res = sweep(thin, X_appr, y_appr, X_rej)

    print("\n[Fase 2a - Bloco 8] concluido.")
