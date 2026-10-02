# Read-only diagnostic: emp_length distribution, rejected (emp_length_raw) vs approved
# (emp_length_anos). Provenance for notebooks/17_reject_thin_model.py.
import duckdb
con = duckdb.connect()

# rejected population
G = "data/processed/reject/rejected.parquet/app_year=*/*.parquet".replace("\\", "/")
rej = f"read_parquet('{G}', hive_partitioning=true)"
print("=== emp_length REJECTED (emp_length_raw) ===")
tot_r = con.execute(f"SELECT COUNT(*) FROM {rej}").fetchone()[0]
for v, c in con.execute(f"SELECT emp_length_raw, COUNT(*) c FROM {rej} GROUP BY emp_length_raw ORDER BY c DESC").fetchall():
    print(f"  {str(v):>12}: {c:>12,} ({100*c/tot_r:5.2f}%)")

# approved population: loans_clean.parquet, column emp_length_anos (numeric, -1 sentinel)
# emp_length_anos: 0 = "<1 year", 1..9 = years, 10 = "10+ years", -1 = missing.
print("\n=== emp_length APPROVED (emp_length_anos) ===")
appr_path = "data/processed/loans_clean.parquet".replace("\\", "/")
appr = f"read_parquet('{appr_path}')"
try:
    tot_a = con.execute(f"SELECT COUNT(*) FROM {appr}").fetchone()[0]
    for v, c in con.execute(f"SELECT emp_length_anos, COUNT(*) c FROM {appr} GROUP BY emp_length_anos ORDER BY emp_length_anos").fetchall():
        label = "<1 year" if v == 0 else ("10+ years" if v == 10 else ("MISSING" if v == -1 else f"{v} years"))
        print(f"  {label:>12} (val={v:>3}): {c:>10,} ({100*c/tot_a:5.2f}%)")
except Exception as e:
    print(f"  [CHECK] Adjust the approved-population path/column. Error: {e}")

print("\n[COMPARISON] If '<1 year' is ~10-20% among approved and 83% among rejected, "
      "the gap points to (a) a real signal (short tenure -> rejection) OR (b) a disguised form default. "
      "If the shares are similar, it simply reflects the population.")
