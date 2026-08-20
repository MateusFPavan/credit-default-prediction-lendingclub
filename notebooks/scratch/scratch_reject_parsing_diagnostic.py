import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("reject_parsing_diagnostic")
    .config("spark.sql.session.timeZone", "UTC")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

REPO = r"C:\Users\Avell\Documents\Projetos\credit-default-prediction-lendingclub"
RAW_GZ = os.path.join(REPO, "data", "raw", "rejected_2007_to_2018Q4.csv.gz")

# The column that should be numeric; a non-numeric value there flags a broken row.
RISK_COL = "Risk_Score"
AMT_COL = "Amount Requested"

# A value is "numeric-or-null" if it is null or matches a decimal pattern.
NUMERIC_RE = r"^\s*-?\d+(\.\d+)?\s*$"

def numeric_or_null(colname):
    c = F.col(f"`{colname}`")
    return c.isNull() | c.rlike(NUMERIC_RE)

# ---------------------------------------------------------------------------
# PARSER A — current settings (header, no special quote handling)
# ---------------------------------------------------------------------------
dfA = (
    spark.read
    .option("header", True)
    .option("inferSchema", False)
    .csv(RAW_GZ)
)
totalA = dfA.count()

badA = dfA.filter(~numeric_or_null(RISK_COL))
n_badA = badA.count()

print("==================== CONTAGEM (parser atual) ====================")
print(f"Total de linhas               : {totalA:,}")
print(f"Risk_Score NAO-numerico       : {n_badA:,}")
print(f"Proporcao                     : {100.0*n_badA/totalA:.6f}%")

# also check Amount Requested being non-numeric (a shift usually breaks more than one col)
badA_amt = dfA.filter(~numeric_or_null(AMT_COL)).count()
print(f"Amount Requested NAO-numerico : {badA_amt:,}")

# ---------------------------------------------------------------------------
# PARSER B — adjusted quote/escape handling for nested double-quotes ("" as escape)
#            + multiLine so a quoted field spanning odd content is kept together.
# ---------------------------------------------------------------------------
dfB = (
    spark.read
    .option("header", True)
    .option("inferSchema", False)
    .option("multiLine", True)
    .option("quote", '"')
    .option("escape", '"')      # doubled double-quote ("") is the CSV-standard escape
    .csv(RAW_GZ)
)
# multiLine changes partitioning; count may take a bit longer.
totalB = dfB.count()
n_badB = dfB.filter(~numeric_or_null(RISK_COL)).count()
badB_amt = dfB.filter(~numeric_or_null(AMT_COL)).count()

print("\n=============== COMPARACAO DE PARSERS (aspas ajustadas) ===============")
print(f"Total linhas   | atual: {totalA:,}   ajustado: {totalB:,}")
print(f"Risk_Score bad | atual: {n_badA:,}    ajustado: {n_badB:,}")
print(f"Amount   bad   | atual: {badA_amt:,}    ajustado: {badB_amt:,}")
if totalB == totalA and n_badB < n_badA:
    print(">> O parsing ajustado RECUPEROU linhas sem perder contagem. Preferir na origem.")
elif totalB != totalA:
    print(">> ATENCAO: o parser ajustado mudou a contagem total de linhas. "
          "Reportar ambos os numeros ao Mateus; nao assumir qual esta certo.")
else:
    print(">> O parsing ajustado NAO reduziu as linhas quebradas. "
          "O residuo e' provavelmente corrupcao real; try_cast e' o caminho para o residuo.")

# ---------------------------------------------------------------------------
# EXEMPLOS — mostrar ate 5 linhas problematicas do parser atual, colunas cruas
# ---------------------------------------------------------------------------
print("\n==================== EXEMPLOS (parser atual) ====================")
sample_bad = badA.limit(5).collect()
for i, row in enumerate(sample_bad, 1):
    d = row.asDict()
    print(f"\n--- linha problematica #{i} ---")
    for k, v in d.items():
        vs = (v[:120] + "...") if isinstance(v, str) and len(v) > 120 else v
        print(f"    {k!r}: {vs!r}")

print("\n[FIM DO DIAGNOSTICO] Nada foi escrito em disco. Reporte os tres blocos ao Mateus.")
