import duckdb
G = "data/processed/reject/rejected.parquet/app_year=*/*.parquet".replace("\\", "/")
con = duckdb.connect()
rel = f"read_parquet('{G}', hive_partitioning=true)"

# --- BLOCO A: dti_raw cru dos casos dti == 100 -------------------------------
print("=== dti_raw CRU onde dti == 100 (top 15) ===")
q = f"""
SELECT dti_raw, COUNT(*) c FROM {rel}
WHERE dti = 100 GROUP BY dti_raw ORDER BY c DESC LIMIT 15
"""
for raw, c in con.execute(q).fetchall():
    print(f"  raw={raw!r:>16} : {c:,}")

# --- BLOCO B: dti_raw cru dos negativos --------------------------------------
print("\n=== dti_raw CRU onde dti < 0 (top 15) ===")
q = f"""
SELECT dti_raw, COUNT(*) c FROM {rel}
WHERE dti < 0 GROUP BY dti_raw ORDER BY c DESC LIMIT 15
"""
for raw, c in con.execute(q).fetchall():
    print(f"  raw={raw!r:>16} : {c:,}")

# --- BLOCO C: os negativos batem num valor unico? amplitude ------------------
print("\n=== Estatistica dos negativos ===")
q = f"SELECT MIN(dti), MAX(dti), COUNT(DISTINCT dti) FROM {rel} WHERE dti < 0"
mn, mx, ndist = con.execute(q).fetchone()
print(f"  min={mn} max={mx} valores_distintos={ndist}")

# --- BLOCO D: dti=100 tem cara de teto? ver vizinhanca 95-105 ----------------
print("\n=== Distribuicao na vizinhanca de 100 (95 a 105) ===")
q = f"""
SELECT ROUND(dti) faixa, COUNT(*) c FROM {rel}
WHERE dti >= 95 AND dti <= 105 GROUP BY ROUND(dti) ORDER BY faixa
"""
for faixa, c in con.execute(q).fetchall():
    print(f"  dti~{faixa:>4} : {c:,}")

print("\n[FIM] So leitura. Reportar A, B, C, D ao Mateus.")
