import duckdb, os

# caminho do Parquet dos recusados (mesmo da Fase 1 / notebook 17)
G = "data/processed/reject/rejected.parquet/app_year=*/*.parquet".replace("\\", "/")
con = duckdb.connect()
rel = f"read_parquet('{G}', hive_partitioning=true)"

# --- BLOCO 1: o dti_raw cru existe no Parquet? Se nao, ler do gzip. -----------
cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {rel}").fetchall()]
print("Colunas disponiveis:", cols)
DTI = "dti_raw" if "dti_raw" in cols else "dti"
print(f"Analisando coluna: {DTI}")

# --- BLOCO 2: quantos batem em cada faixa critica ----------------------------
print("\n=== Contagem por faixa de dti (numerico) ===")
q = f"""
SELECT
  COUNT(*) FILTER (WHERE dti < 0)                        AS negativos,
  COUNT(*) FILTER (WHERE dti = 0)                        AS zero,
  COUNT(*) FILTER (WHERE dti > 0 AND dti < 100)          AS entre_0_100,
  COUNT(*) FILTER (WHERE dti = 100)                      AS exatamente_100,
  COUNT(*) FILTER (WHERE dti > 100 AND dti <= 1000)      AS entre_100_1000,
  COUNT(*) FILTER (WHERE dti > 1000)                     AS acima_1000,
  COUNT(*) FILTER (WHERE dti IS NULL)                    AS nulos,
  COUNT(*)                                               AS total
FROM {rel}
"""
for k, v in zip([d[0] for d in con.execute(q).description], con.execute(q).fetchone()):
    print(f"  {k:<18}: {v:,}")

# --- BLOCO 3: os valores mais frequentes acima de 100 (procura sentinela) -----
print("\n=== Top 15 valores de dti MAIS FREQUENTES acima de 100 ===")
q = f"""
SELECT dti, COUNT(*) c FROM {rel}
WHERE dti > 100 GROUP BY dti ORDER BY c DESC LIMIT 15
"""
for dti_val, c in con.execute(q).fetchall():
    print(f"  dti={dti_val:>15} : {c:,}")

# --- BLOCO 4: amostra do dti_raw CRU (string) para esses casos ---------------
if "dti_raw" in cols:
    print("\n=== Amostra do dti_raw CRU (string) onde dti > 100 ===")
    q = f"""
    SELECT dti_raw, COUNT(*) c FROM {rel}
    WHERE dti > 100 GROUP BY dti_raw ORDER BY c DESC LIMIT 15
    """
    for raw, c in con.execute(q).fetchall():
        print(f"  raw={raw!r:>20} : {c:,}")
else:
    print("\n[nota] dti_raw (string cru) nao esta no Parquet; so o dti numerico. "
          "Se precisar do cru, ler do gzip original.")

print("\n[FIM] So leitura. Reportar os 4 blocos ao Mateus. Nada alterado.")
