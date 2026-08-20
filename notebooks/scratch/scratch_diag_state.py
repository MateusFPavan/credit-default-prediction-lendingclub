import duckdb
con = duckdb.connect()
G = "data/processed/reject/rejected.parquet/app_year=*/*.parquet".replace("\\", "/")
rej = f"read_parquet('{G}', hive_partitioning=true)"
tot = con.execute(f"SELECT COUNT(*) FROM {rej}").fetchone()[0]

# 1. cardinalidade e nulos
print("=== state: cardinalidade e nulos ===")
q = f"""
SELECT
  COUNT(DISTINCT state)                                   AS distintos,
  COUNT(*) FILTER (WHERE state IS NULL)                   AS nulos,
  COUNT(*) FILTER (WHERE TRIM(CAST(state AS VARCHAR))='') AS vazios,
  COUNT(*) FILTER (WHERE LENGTH(TRIM(CAST(state AS VARCHAR))) <> 2) AS nao_2_letras,
  COUNT(*) total
FROM {rej}
"""
for name, val in zip([d[0] for d in con.execute(q).description], con.execute(q).fetchone()):
    print(f"  {name:<14}: {val:,}")

# 2. distribuicao completa (todos os valores distintos, ordenados)
print("\n=== TODOS os valores distintos de state (contagem) ===")
rows = con.execute(f"SELECT state, COUNT(*) c FROM {rej} GROUP BY state ORDER BY c DESC").fetchall()
print(f"  (total de {len(rows)} valores distintos)")
for v, c in rows:
    print(f"  {str(v):>6}: {c:>12,} ({100*c/tot:5.2f}%)")

# 3. checar valores fora dos 50 estados + DC (territorios, invalidos)
print("\n=== valores que NAO sao um dos 50 estados + DC ===")
US_STATES = {
 'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA',
 'ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK',
 'OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY','DC'}
outside = [(v, c) for v, c in rows if str(v) not in US_STATES]
if outside:
    print("  Fora dos 50+DC (possiveis territorios/invalidos):")
    for v, c in outside:
        print(f"    {str(v):>6}: {c:,}")
else:
    print("  Nenhum -- todos os valores sao estados validos + DC.")

print("\n[FIM] So leitura. Reportar ao Mateus.")
