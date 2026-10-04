# Read-only diagnostic (DTI, part 1): rejected-population dti counts by range (negative, 0,
# 100, >100, null) and frequent values above 100. Provenance for notebooks/17_reject_thin_model.py.
import duckdb, os

# Rejected-population Parquet (written by notebook 16, read by notebook 17)
G = "data/processed/reject/rejected.parquet/app_year=*/*.parquet".replace("\\", "/")
con = duckdb.connect()
rel = f"read_parquet('{G}', hive_partitioning=true)"

# --- 1: is the raw dti_raw string in the Parquet? If not, read the gzip. -----
cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {rel}").fetchall()]
print("Available columns:", cols)
DTI = "dti_raw" if "dti_raw" in cols else "dti"
print(f"Analyzing column: {DTI}")

# --- 2: row counts in each critical dti range --------------------------------
print("\n=== Row count by dti range (numeric) ===")
q = f"""
SELECT
  COUNT(*) FILTER (WHERE dti < 0)                        AS negative,
  COUNT(*) FILTER (WHERE dti = 0)                        AS zero,
  COUNT(*) FILTER (WHERE dti > 0 AND dti < 100)          AS between_0_100,
  COUNT(*) FILTER (WHERE dti = 100)                      AS exactly_100,
  COUNT(*) FILTER (WHERE dti > 100 AND dti <= 1000)      AS between_100_1000,
  COUNT(*) FILTER (WHERE dti > 1000)                     AS above_1000,
  COUNT(*) FILTER (WHERE dti IS NULL)                    AS nulls,
  COUNT(*)                                               AS total
FROM {rel}
"""
for k, v in zip([d[0] for d in con.execute(q).description], con.execute(q).fetchone()):
    print(f"  {k:<18}: {v:,}")

# --- 3: most frequent values above 100 (looking for a sentinel) -------------
print("\n=== Top 15 MOST FREQUENT dti values above 100 ===")
q = f"""
SELECT dti, COUNT(*) c FROM {rel}
WHERE dti > 100 GROUP BY dti ORDER BY c DESC LIMIT 15
"""
for dti_val, c in con.execute(q).fetchall():
    print(f"  dti={dti_val:>15} : {c:,}")

# --- 4: sample of the RAW dti_raw string for those rows ---------------------
if "dti_raw" in cols:
    print("\n=== Sample of RAW dti_raw (string) where dti > 100 ===")
    q = f"""
    SELECT dti_raw, COUNT(*) c FROM {rel}
    WHERE dti > 100 GROUP BY dti_raw ORDER BY c DESC LIMIT 15
    """
    for raw, c in con.execute(q).fetchall():
        print(f"  raw={raw!r:>20} : {c:,}")
else:
    print("\n[NOTE] dti_raw (raw string) is not in the Parquet; only the numeric dti is. "
          "Read the original gzip if the raw value is needed.")

print("\n[END] Read-only diagnostic; nothing was written.")
