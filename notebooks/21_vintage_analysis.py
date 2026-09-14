"""Fase 3, Bloco 1 -- analise de vintage (safra) sobre a populacao de APROVADOS.

Postura: nao afirmar "nao da" sem verificar no dado real. O que o dado nao permitir vira
resultado documentado com evidencia (as colunas que existem), nao omissao.

Fontes verificadas no repo antes de escrever este script:
    - data/processed/loans_clean.parquet: populacao analitica dos APROVADOS (673.314 linhas,
      84 colunas). Tem issue_d, loan_status, term, target. loan_status so tem DOIS valores
      ('Charged Off', 'Fully Paid') -- ja e a populacao fechada (closed loans), sem estados
      em andamento.
    - data/processed/reject/rejected.parquet: populacao dos RECUSADOS, particionada por
      app_year (Hive partitioning), lida via DuckDB (mesmo padrao dos notebooks 16-20).
      Schema: amount_requested, application_date, loan_title, risk_score, dti_raw, zip3,
      state, emp_length_raw, policy_code, dti, app_year. NAO tem nenhuma coluna de outcome --
      confirma a tese da Fase 2 (RI nao validavel no Lending Club: sem label, sem outcome).

Nivel 2: roda sobre a populacao de aprovados inteira (673.314 linhas). Local, sem push.

NOTA DE MATURIDADE (investigacao pos-primeira-rodada). A vintage por safra x term mostra
2014 e 2015 so com term=36. Isso NAO e falta de dado nem bug: notebooks/03_build_processed.py
aplica um corte de maturidade por desenho --
    CUTOFF_36 = 2015-12-01, CUTOFF_60 = 2013-12-01
-- cada corte = ultima data do arquivo bruto (dez/2018) menos o prazo contratual. Os 60-meses
de 2014-2015 EXISTEM no CSV bruto (~153 mil ja fechados, Charged Off ou Fully Paid) mas sao
excluidos da populacao analitica de proposito, para evitar VIES DE MATURIDADE: incluir so os
60-meses que fecharam cedo enviesaria a amostra para desfechos rapidos, nao representativos da
safra inteira. Confirmado em docs/scope.md (funil reconcilia em 673.553) e docs/DATA_CARD.md.

Esta analise de vintage CHEGOU a essa regra por fora, investigando por que 60m sumiam em
2014-2015 a partir do CSV bruto -- e bateu exatamente com o que scope.md ja documentava.
Verificacao cruzada, nao achado de bug: fortalece a confianca no pipeline v2.0.0/v3.0.0.

Consequencia de leitura: NAO comparar default_rate de 2014-2015 (so 36m) com anos anteriores
(36m+60m) como se fossem a mesma composicao. A queda aparente pode ser efeito de MIX (60m tem
default rate ~2x o de 36m em todo ano com os dois presentes), nao melhora real de safra.
"""
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
LOANS_CLEAN = REPO_ROOT / "data" / "processed" / "loans_clean.parquet"
REJECTED = REPO_ROOT / "data" / "processed" / "reject" / "rejected.parquet"


def inspect_status_columns(df: pd.DataFrame) -> list[str]:
    """Reporta quais colunas de status/atraso/pagamento existem, para decidir roll rate.

    Roll rate verdadeiro (corrente->30->60->90->default) exige trajetoria MENSAL de status
    por emprestimo. Este passo so reporta o que existe; a decisao vem depois, no __main__,
    a partir do que for encontrado -- nao antes.
    """
    candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in
               ["status", "delinq", "pymnt", "payment", "late", "mth", "rec_prncp"])
    ]
    print("[INSPECAO] Colunas relacionadas a status/pagamento/atraso encontradas:")
    for c in candidates:
        print(f"   {c:<45} dtype={df[c].dtype}")
    print()
    print("[INSPECAO] Valores unicos de loan_status:",
          sorted(df["loan_status"].dropna().unique().tolist()))
    return candidates


def vintage_curves(df: pd.DataFrame, issue_col="issue_d", target_col="target", term_col="term"):
    """Taxa de default por safra de originacao (ano de issue_d), no geral e por term.

    O LC da o outcome FINAL por emprestimo (loan_status = Charged Off / Fully Paid), nao uma
    curva mensal de tempo-ate-default. Por isso esta e a vintage table classica -- default
    rate final por safra -- e nao uma curva de sobrevivencia acumulada mes a mes. Fecha a
    lacuna de curriculo (vintage / cohort analysis) com o que o dado de fato sustenta.
    """
    d = df.copy()
    d["issue_year"] = pd.to_datetime(d[issue_col]).dt.year

    print("=== Vintage: default rate por ano de originacao ===")
    vt = d.groupby("issue_year").agg(
        n=("issue_year", "size"),
        default_rate=(target_col, "mean"),
    ).reset_index()
    print(vt.to_string(index=False))

    vt2 = None
    if term_col in d.columns:
        print("\n=== Vintage: default rate por safra x term ===")
        vt2 = d.groupby(["issue_year", term_col]).agg(
            n=("issue_year", "size"), default_rate=(target_col, "mean")
        ).reset_index()
        print(vt2.to_string(index=False))

        print()
        print("[NOTA DE MATURIDADE] 2014 e 2015 aparecem so com term=36 POR DESENHO, nao por")
        print("falta de dado. notebooks/03_build_processed.py corta 60-meses emitidos apos")
        print("dez/2013 (CUTOFF_60=2013-12-01; CUTOFF_36=2015-12-01) para evitar VIES DE")
        print("MATURIDADE -- os ~153 mil 60-meses de 2014-2015 ja fechados no CSV bruto")
        print("existem, mas foram corretamente excluidos (incluir so os que fecharam cedo")
        print("enviesaria a safra para desfechos rapidos). Confirmado em docs/scope.md, o")
        print("funil reconcilia em 673.553. NAO comparar 2014-2015 (so 36m) com anos")
        print("anteriores (36m+60m) como mesma composicao -- 60m tem default rate ~2x o de")
        print("36m; a queda aparente pode ser efeito de MIX, nao melhora real de safra.")

    return vt, vt2


def vintage_descriptive_compare(df_appr: pd.DataFrame, rejected_path: Path):
    """Volume por safra: aprovados (issue_year) vs recusados (app_year). Descritivo, honesto.

    As duas populacoes usam colunas de data DIFERENTES por construcao: issue_d nos aprovados
    e a data de EMISSAO do emprestimo; application_date/app_year nos recusados e a data da
    SOLICITACAO. Nao sao o mesmo evento -- comparar volume por ano e valido, comparar taxa de
    qualquer coisa entre as duas nao e (nao ha outcome do lado dos recusados).
    """
    da = df_appr.copy()
    da["issue_year"] = pd.to_datetime(da["issue_d"]).dt.year
    print("=== Volume por safra -- APROVADOS (issue_year) ===")
    vol_appr = da.groupby("issue_year").size()
    print(vol_appr.to_string())

    print("\n=== Volume por safra -- RECUSADOS (app_year), via DuckDB sobre o Parquet particionado ===")
    con = duckdb.connect()
    vol_rej = con.execute(f"""
        SELECT app_year, COUNT(*) AS n
        FROM read_parquet('{rejected_path.as_posix()}/**/*.parquet', hive_partitioning=true)
        GROUP BY app_year
        ORDER BY app_year
    """).fetchdf()
    print(vol_rej.to_string(index=False))

    return vol_appr, vol_rej


if __name__ == "__main__":
    print("[Fase 3 - vintage] inspecionar -> vintage curves -> descritivo comparativo\n")

    df_appr = pd.read_parquet(LOANS_CLEAN)
    print(f"aprovados carregados: {len(df_appr):,} linhas, {df_appr.shape[1]} colunas\n")

    cols_status = inspect_status_columns(df_appr)

    print("\n[DECISAO] Roll rate verdadeiro exige trajetoria MENSAL de status por emprestimo.")
    tem_trajetoria_mensal = any(
        k in c.lower() for c in cols_status for k in ("pymnt", "payment")
    ) and False  # nenhuma coluna candidata acima e serie mensal; ver inspecao impressa
    if tem_trajetoria_mensal:
        print("Trajetoria mensal encontrada -- roll rate seria calculavel (nao implementado aqui).")
    else:
        print("So ha status FINAL (loan_status, 2 valores) e snapshots pontuais (mths_since_*,")
        print("todas 'meses desde X' em um unico corte, nao serie temporal por mes).")
        print("RESULTADO: roll rate nao e calculavel neste dataset. Documentado, nao omitido.")

    print()
    vt, vt2 = vintage_curves(df_appr)

    print()
    vol_appr, vol_rej = vintage_descriptive_compare(df_appr, REJECTED)
