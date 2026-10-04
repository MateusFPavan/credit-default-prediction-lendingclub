# Read-only diagnostic (DTI, part 2): raw dti_raw strings behind dti == 100 and dti < 0, and
# the distribution around 100. Provenance for notebooks/17_reject_thin_model.py.
import duckdb
G = "data/processed/reject/rejected.parquet/app_year=*/*.parquet".replace("\\", "/")
con = duckdb.connect()
rel = f"read_parquet('{G}', hive_partitioning=true)"

# --- A: raw dti_raw strings where dti == 100 ---------------------------------
print("=== RAW dti_raw where dti == 100 (top 15) ===")
q = f"""
SELECT dti_raw, COUNT(*) c FROM {rel}
WHERE dti = 100 GROUP BY dti_raw ORDER BY c DESC LIMIT 15
"""
for raw, c in con.execute(q).fetchall():
    print(f"  raw={raw!r:>16} : {c:,}")

# --- B: raw dti_raw strings of the negative values ---------------------------
print("\n=== RAW dti_raw where dti < 0 (top 15) ===")
q = f"""
SELECT dti_raw, COUNT(*) c FROM {rel}
WHERE dti < 0 GROUP BY dti_raw ORDER BY c DESC LIMIT 15
"""
for raw, c in con.execute(q).fetchall():
    print(f"  raw={raw!r:>16} : {c:,}")

# --- C: do the negatives collapse to a single value? range -------------------
print("\n=== Negative-value statistics ===")
q = f"SELECT MIN(dti), MAX(dti), COUNT(DISTINCT dti) FROM {rel} WHERE dti < 0"
mn, mx, ndist = con.execute(q).fetchone()
print(f"  min={mn} max={mx} distinct_values={ndist}")

# --- D: does dti == 100 look like a cap? inspect the 95-105 neighborhood -----
print("\n=== Distribution around 100 (95 to 105) ===")
q = f"""
SELECT ROUND(dti) band, COUNT(*) c FROM {rel}
WHERE dti >= 95 AND dti <= 105 GROUP BY ROUND(dti) ORDER BY band
"""
for band, c in con.execute(q).fetchall():
    print(f"  dti~{band:>4} : {c:,}")

print("\n[END] Read-only diagnostic; nothing was written.")
