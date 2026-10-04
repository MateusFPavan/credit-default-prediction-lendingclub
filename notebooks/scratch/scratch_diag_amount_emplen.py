# Read-only diagnostic: amount_requested range/frequent values and the emp_length_raw value
# distribution in the rejected Parquet. Provenance for notebooks/17_reject_thin_model.py.
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
  COUNT(*) FILTER (WHERE amount_requested <= 0)      AS zero_or_neg,
  COUNT(*) FILTER (WHERE amount_requested IS NULL)   AS nulls,
  COUNT(*) total
FROM {rel}
"""
r = con.execute(q).fetchone()
for name, val in zip([d[0] for d in con.execute(q).description], r):
    print(f"  {name:<12}: {val:,}" if isinstance(val, (int, float)) and val == int(val) else f"  {name:<12}: {val}")

print("\n  Top 15 most frequent amount_requested values:")
for v, c in con.execute(f"SELECT amount_requested, COUNT(*) c FROM {rel} GROUP BY amount_requested ORDER BY c DESC LIMIT 15").fetchall():
    print(f"    {v:>12} : {c:,}")

print("\n  Top 10 LARGEST distinct values:")
for v, c in con.execute(f"SELECT amount_requested, COUNT(*) c FROM {rel} GROUP BY amount_requested ORDER BY amount_requested DESC LIMIT 10").fetchall():
    print(f"    {v:>12} : {c:,}")

# ============ emp_length_raw ============
print("\n" + "=" * 60)
print("EMP_LENGTH_RAW (text)")
print("=" * 60)
print("  Distribution of emp_length_raw values:")
for v, c in con.execute(f"SELECT emp_length_raw, COUNT(*) c FROM {rel} GROUP BY emp_length_raw ORDER BY c DESC LIMIT 30").fetchall():
    print(f"    {v!r:>16} : {c:,}")

q = f"""
SELECT
  COUNT(*) FILTER (WHERE emp_length_raw IS NULL)                 AS nulls,
  COUNT(*) FILTER (WHERE TRIM(CAST(emp_length_raw AS VARCHAR))='') AS empty,
  COUNT(*) total
FROM {rel}
"""
r = con.execute(q).fetchone()
print(f"\n  nulls={r[0]:,}  empty={r[1]:,}  total={r[2]:,}")

print("\n[END] Read-only diagnostic; nothing was written.")
