import duckdb
G = "data/processed/reject/rejected.parquet/app_year=*/*.parquet".replace("\\", "/")
con = duckdb.connect()
rel = f"read_parquet('{G}', hive_partitioning=true)"

# ============ amount_requested ============
print("=" * 60)
print("AMOUNT_REQUESTED")
print("=" * 60)
q = f"""
SELECT
  MIN(amount_requested) mn, MAX(amount_requested) mx,
  AVG(amount_requested) avg, MEDIAN(amount_requested) med,
  COUNT(*) FILTER (WHERE amount_requested <= 0)      AS zero_ou_neg,
  COUNT(*) FILTER (WHERE amount_requested IS NULL)   AS nulos,
  COUNT(*) total
FROM {rel}
"""
r = con.execute(q).fetchone()
for name, val in zip([d[0] for d in con.execute(q).description], r):
    print(f"  {name:<12}: {val:,}" if isinstance(val, (int, float)) and val == int(val) else f"  {name:<12}: {val}")

print("\n  Top 15 valores mais frequentes de amount_requested:")
for v, c in con.execute(f"SELECT amount_requested, COUNT(*) c FROM {rel} GROUP BY amount_requested ORDER BY c DESC LIMIT 15").fetchall():
    print(f"    {v:>12} : {c:,}")

print("\n  Top 10 MAIORES valores distintos:")
for v, c in con.execute(f"SELECT amount_requested, COUNT(*) c FROM {rel} GROUP BY amount_requested ORDER BY amount_requested DESC LIMIT 10").fetchall():
    print(f"    {v:>12} : {c:,}")

# ============ emp_length_raw ============
print("\n" + "=" * 60)
print("EMP_LENGTH_RAW (texto)")
print("=" * 60)
print("  Distribuicao dos valores de emp_length_raw:")
for v, c in con.execute(f"SELECT emp_length_raw, COUNT(*) c FROM {rel} GROUP BY emp_length_raw ORDER BY c DESC LIMIT 30").fetchall():
    print(f"    {v!r:>16} : {c:,}")

q = f"""
SELECT
  COUNT(*) FILTER (WHERE emp_length_raw IS NULL)                 AS nulos,
  COUNT(*) FILTER (WHERE TRIM(CAST(emp_length_raw AS VARCHAR))='') AS vazios,
  COUNT(*) total
FROM {rel}
"""
r = con.execute(q).fetchone()
print(f"\n  nulos={r[0]:,}  vazios={r[1]:,}  total={r[2]:,}")

print("\n[FIM] So leitura. Reportar amount e emp_length ao Mateus.")
