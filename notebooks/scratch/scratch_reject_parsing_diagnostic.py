# Read-only diagnostic: row count and non-numeric Risk_Score/Amount, default vs quote/escape parsing.
# Provenance for the parser options of notebook 16, whose output feeds notebooks/17_reject_thin_model.py.
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

# Repo root: CREDIT_REPO if set, otherwise derived from the working directory
# (repo root, notebooks/ or notebooks/scratch/).
REPO = os.environ.get("CREDIT_REPO") or os.path.abspath(
    os.path.join(os.getcwd(), *([".."] * {"notebooks": 1, "scratch": 2}.get(os.path.basename(os.getcwd()), 0)))
)
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

print("==================== COUNTS (current parser) ====================")
print(f"Total rows                    : {totalA:,}")
print(f"Risk_Score non-numeric        : {n_badA:,}")
print(f"Proportion                    : {100.0*n_badA/totalA:.6f}%")

# also check Amount Requested being non-numeric (a shift usually breaks more than one col)
badA_amt = dfA.filter(~numeric_or_null(AMT_COL)).count()
print(f"Amount Requested non-numeric  : {badA_amt:,}")

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

print("\n=============== PARSER COMPARISON (adjusted quote/escape) ===============")
print(f"Total rows     | current: {totalA:,}   adjusted: {totalB:,}")
print(f"Risk_Score bad | current: {n_badA:,}    adjusted: {n_badB:,}")
print(f"Amount   bad   | current: {badA_amt:,}    adjusted: {badB_amt:,}")
if totalB == totalA and n_badB < n_badA:
    print(">> Adjusted parsing RECOVERED the broken rows with no change in row count. Prefer fixing at the source.")
elif totalB != totalA:
    print(">> WARNING: the adjusted parser changed the total row count. "
          "Record both numbers; do not assume which one is correct.")
else:
    print(">> Adjusted parsing did NOT reduce the broken rows. "
          "The remainder is likely genuine corruption; handle it with try_cast.")

# ---------------------------------------------------------------------------
# EXAMPLES - up to 5 problematic rows from the current parser, raw columns
# ---------------------------------------------------------------------------
print("\n==================== EXAMPLES (current parser) ====================")
sample_bad = badA.limit(5).collect()
for i, row in enumerate(sample_bad, 1):
    d = row.asDict()
    print(f"\n--- problematic row #{i} ---")
    for k, v in d.items():
        vs = (v[:120] + "...") if isinstance(v, str) and len(v) > 120 else v
        print(f"    {k!r}: {vs!r}")

print("\n[END] Read-only diagnostic; nothing was written to disk.")
