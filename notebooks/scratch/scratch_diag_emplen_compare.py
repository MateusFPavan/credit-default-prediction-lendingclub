import duckdb
con = duckdb.connect()

# recusados
G = "data/processed/reject/rejected.parquet/app_year=*/*.parquet".replace("\\", "/")
rej = f"read_parquet('{G}', hive_partitioning=true)"
print("=== emp_length RECUSADOS (emp_length_raw) ===")
tot_r = con.execute(f"SELECT COUNT(*) FROM {rej}").fetchone()[0]
for v, c in con.execute(f"SELECT emp_length_raw, COUNT(*) c FROM {rej} GROUP BY emp_length_raw ORDER BY c DESC").fetchall():
    print(f"  {str(v):>12}: {c:>12,} ({100*c/tot_r:5.2f}%)")

# aprovados: loans_clean.parquet, coluna emp_length_anos (numerica, sentinela -1)
# emp_length_anos: 0 = "<1 year", 1..9 = anos, 10 = "10+ years", -1 = missing.
print("\n=== emp_length APROVADOS (emp_length_anos) ===")
appr_path = "data/processed/loans_clean.parquet".replace("\\", "/")
appr = f"read_parquet('{appr_path}')"
try:
    tot_a = con.execute(f"SELECT COUNT(*) FROM {appr}").fetchone()[0]
    for v, c in con.execute(f"SELECT emp_length_anos, COUNT(*) c FROM {appr} GROUP BY emp_length_anos ORDER BY emp_length_anos").fetchall():
        label = "<1 year" if v == 0 else ("10+ years" if v == 10 else ("MISSING" if v == -1 else f"{v} years"))
        print(f"  {label:>12} (val={v:>3}): {c:>10,} ({100*c/tot_a:5.2f}%)")
except Exception as e:
    print(f"  [VERIFICAR NO REPO] ajustar caminho/coluna dos aprovados. Erro: {e}")

print("\n[COMPARACAO] Se '<1 year' for ~10-20% nos aprovados e 83% nos recusados, "
      "a diferenca sugere (a) sinal real (emprego curto -> recusa) OU (b) default disfarcado. "
      "Se for parecido nas duas, e' so a natureza da populacao. Reportar ao Mateus.")
