# Read-only diagnostic: cardinality, nulls and validity of the state column in the rejected
# Parquet. Provenance for notebooks/17_reject_thin_model.py.
import duckdb
con = duckdb.connect()
G = "data/processed/reject/rejected.parquet/app_year=*/*.parquet".replace("\\", "/")
rej = f"read_parquet('{G}', hive_partitioning=true)"
tot = con.execute(f"SELECT COUNT(*) FROM {rej}").fetchone()[0]

# 1. cardinality and nulls
print("=== state: cardinality and nulls ===")
q = f"""
SELECT
  COUNT(DISTINCT state)                                   AS n_distinct,
  COUNT(*) FILTER (WHERE state IS NULL)                   AS nulls,
  COUNT(*) FILTER (WHERE TRIM(CAST(state AS VARCHAR))='') AS empty,
  COUNT(*) FILTER (WHERE LENGTH(TRIM(CAST(state AS VARCHAR))) <> 2) AS not_2_letters,
  COUNT(*) total
FROM {rej}
"""
for name, val in zip([d[0] for d in con.execute(q).description], con.execute(q).fetchone()):
    print(f"  {name:<14}: {val:,}")

# 2. full distribution (every distinct value, sorted by count)
print("\n=== ALL distinct state values (count) ===")
rows = con.execute(f"SELECT state, COUNT(*) c FROM {rej} GROUP BY state ORDER BY c DESC").fetchall()
print(f"  ({len(rows)} distinct values in total)")
for v, c in rows:
    print(f"  {str(v):>6}: {c:>12,} ({100*c/tot:5.2f}%)")

# 3. values outside the 50 states + DC (territories, invalid codes)
print("\n=== values that are NOT one of the 50 states + DC ===")
US_STATES = {
 'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA',
 'ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK',
 'OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY','DC'}
outside = [(v, c) for v, c in rows if str(v) not in US_STATES]
if outside:
    print("  Outside 50+DC (possible territories/invalid codes):")
    for v, c in outside:
        print(f"    {str(v):>6}: {c:,}")
else:
    print("  None -- every value is a valid state or DC.")

print("\n[END] Read-only diagnostic; nothing was written.")
