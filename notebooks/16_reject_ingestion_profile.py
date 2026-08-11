# Databricks notebook source
# MAGIC %md
# MAGIC # Fase 1 — Reject population: ingestion, count reconciliation, and profiling
# MAGIC
# MAGIC Extends the credit-default project (v2.0.0, approved-loan model) toward the
# MAGIC **rejected-applicant population**, the half of the Lending Club data the approved
# MAGIC model never saw. This notebook does the ingestion and descriptive profiling only.
# MAGIC Reject inference, the thin model, and the profit comparison are later phases.
# MAGIC
# MAGIC **Why Spark here (and not on the approved pipeline).** The approved book is
# MAGIC ~2.26M rows / 1.67 GB; pandas handles it in minutes and Spark would be decoration.
# MAGIC The rejected file is a single gzip of ~27M rows: a `.csv.gz` is **not splittable**,
# MAGIC so it is read once on a single thread and immediately rewritten as **partitioned
# MAGIC Parquet**, after which every downstream aggregation (per year, per score band, and
# MAGIC the alignment against the approved book) runs partitioned. The justification comes
# MAGIC from the volume of the half that closes the model's #1 limitation, not from a wish
# MAGIC to use the tool.
# MAGIC
# MAGIC **Source.** `rejected_2007_to_2018Q4.csv.gz`, Kaggle dataset
# MAGIC `wordsforthewise/lending-club`, CC0-1.0. Same package as the approved file.
# MAGIC
# MAGIC **Runs in two places, same code.** Developed locally (fast, no quota); validated
# MAGIC once on Databricks Free Edition (serverless) and exported from there. The only
# MAGIC difference is the paths block below.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 — Environment check and paths
# MAGIC Detects local-vs-Databricks and sets paths. Fails loudly if the raw file is absent,
# MAGIC rather than proceeding on an assumption.

# COMMAND ----------

import os
import sys

# --- detect environment -----------------------------------------------------
# On Databricks a `spark` session already exists and `dbutils` is defined.
try:
    dbutils  # noqa: F821
    ON_DATABRICKS = True
except NameError:
    ON_DATABRICKS = False

if ON_DATABRICKS:
    # Free Edition stores data in a Unity Catalog Volume, not the legacy DBFS.
    # Adjust <catalog>/<schema>/<volume> to your workspace (see the import step
    # in the doc block Mateus received). Example layout:
    BASE = "/Volumes/workspace/default/credit_reject"
    RAW_GZ = f"{BASE}/rejected_2007_to_2018Q4.csv.gz"
    PROC = f"{BASE}/processed"
    APPROVED_CLEAN = f"{BASE}/loans_clean.parquet"
    REPORTS = f"{BASE}/reports"
    # `spark` is provided by the Databricks runtime.
else:
    # Local development inside the repo.
    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder
        .appName("reject_fase1")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    REPO = r"C:\Users\Avell\Documents\Projetos\credit-default-prediction-lendingclub"
    RAW_GZ = os.path.join(REPO, "data", "raw", "rejected_2007_to_2018Q4.csv.gz")
    PROC = os.path.join(REPO, "data", "processed", "reject")
    APPROVED_CLEAN = os.path.join(REPO, "data", "processed", "loans_clean.parquet")
    REPORTS = os.path.join(REPO, "reports", "reject")

spark.sparkContext.setLogLevel("WARN")

# --- guard: raw file must exist ---------------------------------------------
def _exists(path):
    if ON_DATABRICKS:
        try:
            dbutils.fs.ls(path)  # noqa: F821
            return True
        except Exception:
            return False
    return os.path.exists(path)

if not _exists(RAW_GZ):
    msg = f"[STOP] Raw file not found at: {RAW_GZ}"
    if not ON_DATABRICKS:
        raw_dir = os.path.dirname(RAW_GZ)
        listing = os.listdir(raw_dir) if os.path.isdir(raw_dir) else "(dir missing)"
        msg += f"\nContents of {raw_dir}: {listing}"
    raise FileNotFoundError(msg)

print("Environment:", "Databricks" if ON_DATABRICKS else "local")
print("Raw gz:", RAW_GZ)
print("Processed out:", PROC)
print("Approved clean:", APPROVED_CLEAN, "(exists:", _exists(APPROVED_CLEAN), ")")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 — Read the gzip (single pass), robust parsing, auditable corrupt-record net
# MAGIC The rejected file has one row (of 27,648,741) whose free-text Loan Title contains
# MAGIC nested double-quotes ("") that misalign the CSV parser. Two layers handle it:
# MAGIC (1) explicit quote/escape + multiLine so the standard CSV escape ("" = one ") is
# MAGIC honored and the row parses correctly at the source, no data lost; (2) PERMISSIVE
# MAGIC mode with a `_corrupt_record` column as an auditable safety net — we then COUNT how
# MAGIC many rows the parser still flags, proving in-dataframe that the fix caught
# MAGIC everything, rather than assuming it did. The column is dropped before profiling so
# MAGIC counts and profiles are unaffected.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType,
)

# In some Spark versions the _corrupt_record column is only populated when column pruning
# is disabled. Set it defensively; harmless if already default.
spark.conf.set("spark.sql.csv.parser.columnPruning.enabled", "false")

CORRUPT_COL = "_corrupt_record"

# We read all source columns as strings first (deliberate casts happen in Cell 4), plus the
# corrupt-record capture column. Building the schema explicitly is what lets PERMISSIVE
# route malformed rows into _corrupt_record instead of silently nulling fields.
SOURCE_COLS = [
    "Amount Requested", "Application Date", "Loan Title", "Risk_Score",
    "Debt-To-Income Ratio", "Zip Code", "State", "Employment Length", "Policy Code",
]
schema = StructType(
    [StructField(c, StringType(), True) for c in SOURCE_COLS]
    + [StructField(CORRUPT_COL, StringType(), True)]
)

df_raw = (
    spark.read
    .option("header", True)
    .option("multiLine", True)
    .option("quote", '"')
    .option("escape", '"')       # CSV-standard: "" inside a quoted field is one literal "
    .option("mode", "PERMISSIVE")
    .option("columnNameOfCorruptRecord", CORRUPT_COL)
    .schema(schema)
    .csv(RAW_GZ)
)

print("Columns read (incl. corrupt-record net):")
for c in df_raw.columns:
    print("   ", repr(c))

# Materialize once so the corrupt-record audit is valid (Spark disallows querying only the
# corrupt column straight from a raw file). Keep a handle to unpersist right after.
df_cached = df_raw.cache()

n_corrupt = df_cached.filter(F.col(CORRUPT_COL).isNotNull()).count()
print(f"\n[AUDIT] Rows flagged as corrupt after robust parsing: {n_corrupt:,}")
if n_corrupt == 0:
    print("[AUDIT] OK — zero malformed rows remain. Parsing fix verified in-dataframe.")
else:
    print("[AUDIT] STOP — corrupt rows remain. Sample below. Report to Mateus, do not treat.")
    df_cached.filter(F.col(CORRUPT_COL).isNotNull()).select(CORRUPT_COL).show(5, truncate=120)

PARSE_AUDIT = {"corrupt_rows_after_fix": int(n_corrupt),
               "parser_options": {"multiLine": True, "quote": '"', "escape": '"',
                                  "mode": "PERMISSIVE"}}
print("\nParse audit recorded for manifest:", PARSE_AUDIT)

# Drop the audit column, then release the cache immediately — it was only needed for the
# audit above, and holding 27.6M free-text rows in heap starves the Cell-4 write.
df_raw = df_cached.drop(CORRUPT_COL)
df_cached.unpersist()
print("[MEM] Audit cache released (unpersist).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 — Count reconciliation (corrects the [CONFIRMAR] in the decision docs)
# MAGIC The decision documents mark the rejected volume as ~27M, unmeasured. This is the
# MAGIC measured number. Report it back to the document generator so the docs get corrected.

# COMMAND ----------

n_reject = df_raw.count()
print(f"[MEASURED] Rejected population row count: {n_reject:,}")

# raw file size, for the manifest
if ON_DATABRICKS:
    size_bytes = sum(f.size for f in dbutils.fs.ls(RAW_GZ))  # noqa: F821
else:
    size_bytes = os.path.getsize(RAW_GZ)
print(f"[MEASURED] Raw .gz size on disk: {size_bytes:,} bytes ({size_bytes/1024/1024:.2f} MiB)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 — Normalize columns and write partitioned Parquet (Arrow, no driver dicts)
# MAGIC On Databricks the write is native Spark. On Windows (local) we avoid Hadoop/winutils
# MAGIC by writing via pandas/pyarrow — but instead of accumulating Python dicts per row
# MAGIC (which caused an OOM), we enable Arrow-based conversion and write one year-partition
# MAGIC at a time via toPandas() (columnar, compact). Large years are sub-sliced by month so
# MAGIC no single toPandas() exceeds one month of rows. Result on disk: Parquet partitioned
# MAGIC by app_year (month slices become multiple files inside each year dir; Spark reads
# MAGIC them back transparently).

# COMMAND ----------

from pyspark.sql import functions as F

# Enable Arrow: makes toPandas() transfer columnar/compact instead of row objects.
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")

# --- column normalization (unchanged logic) --------------------------------
RENAME = {
    "Amount Requested": "amount_requested",
    "Application Date": "application_date",
    "Loan Title": "loan_title",
    "Risk_Score": "risk_score",
    "Debt-To-Income Ratio": "dti_raw",
    "Zip Code": "zip3",
    "State": "state",
    "Employment Length": "emp_length_raw",
    "Policy Code": "policy_code",
}
df = df_raw
for src, dst in RENAME.items():
    if src in df.columns:
        df = df.withColumnRenamed(src, dst)

if "amount_requested" in df.columns:
    df = df.withColumn("amount_requested", F.col("amount_requested").cast("double"))
if "risk_score" in df.columns:
    df = df.withColumn("risk_score", F.col("risk_score").cast("double"))
if "application_date" in df.columns:
    df = df.withColumn("application_date", F.to_date("application_date", "yyyy-MM-dd"))
    df = df.withColumn("app_year", F.year("application_date"))
    df = df.withColumn("app_month", F.month("application_date"))
if "dti_raw" in df.columns:
    df = df.withColumn("dti", F.regexp_replace(F.col("dti_raw"), "%", "").cast("double"))

reject_parquet = (os.path.join(PROC, "rejected.parquet") if not ON_DATABRICKS
                  else f"{PROC}/rejected.parquet")

if ON_DATABRICKS:
    (df.drop("app_month").write.mode("overwrite").partitionBy("app_year").parquet(reject_parquet))
    df = spark.read.parquet(reject_parquet)
    print("Wrote partitioned Parquet (native Spark) to:", reject_parquet)
    print("Row count after write:", df.count())
else:
    import shutil
    import pyarrow as pa
    import pyarrow.parquet as pq
    try:
        import psutil
        vm = psutil.virtual_memory()
        total_gb = vm.total / (1024**3)
        avail_gb = vm.available / (1024**3)
    except Exception:
        total_gb = avail_gb = None

    # Calibrate driver memory and the per-slice row ceiling from available RAM.
    # (Driver memory must be set before the JVM starts; if the session is already up this
    #  is a no-op — report it so Mateus can restart the kernel with more if needed.)
    if total_gb is not None:
        print(f"[MEM] total={total_gb:.1f} GB | available={avail_gb:.1f} GB")
        # aim to keep each toPandas() slice comfortably under ~0.75 GB of pandas memory
        ROWS_PER_SLICE = 400_000 if avail_gb < 4 else (800_000 if avail_gb < 8 else 1_500_000)
    else:
        print("[MEM] psutil unavailable; using conservative slice ceiling.")
        ROWS_PER_SLICE = 400_000
    print(f"[MEM] ROWS_PER_SLICE = {ROWS_PER_SLICE:,}")

    if os.path.isdir(reject_parquet):
        shutil.rmtree(reject_parquet)
    os.makedirs(reject_parquet, exist_ok=True)

    years = [r["app_year"] for r in
             df.select("app_year").distinct().orderBy("app_year").collect()]
    print("Years to write:", years)

    total_written = 0
    for yr in years:
        year_dir = os.path.join(reject_parquet, f"app_year={yr}")
        os.makedirs(year_dir, exist_ok=True)
        dyr = df.filter(F.col("app_year").eqNullSafe(yr))
        n_yr = dyr.count()

        # Decide slicing: whole year if small, else by month.
        if n_yr <= ROWS_PER_SLICE:
            slices = [("all", dyr)]
        else:
            months = [r["app_month"] for r in
                      dyr.select("app_month").distinct().orderBy("app_month").collect()]
            slices = [(f"m{m:02d}", dyr.filter(F.col("app_month").eqNullSafe(m)))
                      for m in months]

        part_ix = 0
        for slabel, sdf in slices:
            n_s = sdf.count()
            if n_s == 0:
                continue
            # Arrow-based, columnar transfer; drop the helper month/year partition cols.
            pdf = sdf.drop("app_year", "app_month").toPandas()
            # If a single month still exceeds the ceiling, split the pandas frame by index.
            if len(pdf) > ROWS_PER_SLICE:
                for start in range(0, len(pdf), ROWS_PER_SLICE):
                    chunk = pdf.iloc[start:start + ROWS_PER_SLICE]
                    tbl = pa.Table.from_pandas(chunk, preserve_index=False)
                    pq.write_table(tbl, os.path.join(year_dir, f"part-{part_ix:04d}.parquet"))
                    part_ix += 1
                    total_written += len(chunk)
            else:
                tbl = pa.Table.from_pandas(pdf, preserve_index=False)
                pq.write_table(tbl, os.path.join(year_dir, f"part-{part_ix:04d}.parquet"))
                part_ix += 1
                total_written += len(pdf)
            del pdf  # free the frame before the next slice
        print(f"  app_year={yr}: {n_yr:,} rows written in {part_ix} file(s)")

    print(f"Wrote partitioned Parquet (pyarrow via Arrow) to: {reject_parquet}")
    print(f"Total rows written: {total_written:,}")

    # Read back for verification via DuckDB (no Hadoop needed on Windows). We do NOT reuse
    # the lazy Spark df here — that would re-read the gzip. DuckDB reads the partitioned
    # Parquet we just wrote, directly.
    import duckdb
    con = duckdb.connect()
    # forward-slashes for the glob; DuckDB is fine with them on Windows too
    glob_path = os.path.join(reject_parquet, "app_year=*", "*.parquet").replace("\\", "/")
    n_back = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{glob_path}', hive_partitioning=true)"
    ).fetchone()[0]
    print("Row count read back (DuckDB):", f"{n_back:,}")
    # keep the glob path and connection for the profiling cells
    REJECT_GLOB = glob_path

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 — Column profile and Risk_Score coverage by year
# MAGIC Local: DuckDB SQL over the partitioned Parquet (no Spark/Hadoop). Databricks: Spark.
# MAGIC Measures null rate per column and, crucially, Risk_Score presence per application
# MAGIC year — external sources indicate ~70% missing; this measures it in our file.

# COMMAND ----------

if ON_DATABRICKS:
    df = spark.read.parquet(reject_parquet)
    total = df.count()
    print("=== Column null rates (Spark) ===")
    for c, t in df.dtypes:
        nulls = df.filter(F.col(c).isNull()).count()
        print(f"   {c:<20} {t:<10} null%={100.0*nulls/total:8.4f}")
    if "risk_score" in df.columns:
        print("\n=== risk_score coverage by year (Spark) ===")
        (df.groupBy("app_year")
           .agg(F.count("*").alias("n"),
                F.count("risk_score").alias("n_risk"),
                F.round(100.0*F.count("risk_score")/F.count("*"), 2).alias("pct_present"),
                F.round(F.avg("risk_score"), 1).alias("mean_risk"))
           .orderBy("app_year").show(50, truncate=False))
else:
    import duckdb
    con = duckdb.connect()
    G = REJECT_GLOB
    rel = f"read_parquet('{G}', hive_partitioning=true)"

    total = con.execute(f"SELECT COUNT(*) FROM {rel}").fetchone()[0]
    print(f"Total rows: {total:,}")

    # column list
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {rel}").fetchall()]
    print("\n=== Column null rates (DuckDB) ===")
    for c in cols:
        q = f'SELECT COUNT(*) - COUNT("{c}") FROM {rel}'
        nulls = con.execute(q).fetchone()[0]
        print(f'   {c:<20} null%={100.0*nulls/total:8.4f}')

    # THE key measurement: risk_score presence by year
    print("\n=== risk_score coverage by application year (DuckDB) ===")
    q = f"""
        SELECT app_year,
               COUNT(*)                                         AS n,
               COUNT(risk_score)                                AS n_risk,
               ROUND(100.0*COUNT(risk_score)/COUNT(*), 2)       AS pct_present,
               ROUND(AVG(risk_score), 1)                        AS mean_risk
        FROM {rel}
        GROUP BY app_year
        ORDER BY app_year
    """
    res = con.execute(q).fetchall()
    print(f"{'year':<6}{'n':>12}{'n_risk':>12}{'pct_present':>14}{'mean_risk':>12}")
    for yr, n, nr, pct, mean in res:
        print(f"{yr:<6}{n:>12,}{nr:>12,}{pct:>13}%{str(mean):>12}")

    # overall risk_score presence, the headline number for the thin-model decision
    overall = con.execute(
        f"SELECT ROUND(100.0*COUNT(risk_score)/COUNT(*), 2) FROM {rel}"
    ).fetchone()[0]
    print(f"\n[HEADLINE] risk_score present in {overall}% of all rejected rows "
          f"(=> {100-overall:.2f}% missing). Drives the Phase-2 thin-model feature choice.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 — Comparative profile: approved vs rejected on shared dimensions
# MAGIC Descriptive only. NOTE (see roadmap finding B): the two `dti` fields are NOT on an
# MAGIC identical basis — approved dti is LC-computed excluding mortgage; rejected dti is
# MAGIC the application-time value of a DENIED application. Compare shape/order of magnitude,
# MAGIC not identical scales. emp_length IS comparable; amount is comparable.

# COMMAND ----------

if not _exists(APPROVED_CLEAN):
    print(f"[INFO] Approved clean parquet not found at {APPROVED_CLEAN}. Profiling "
          f"rejected alone; comparison runs once the approved parquet is available.")
else:
    if ON_DATABRICKS:
        appr = spark.read.parquet(APPROVED_CLEAN)
        n_appr = appr.count()
        print(f"Approved rows: {n_appr:,}")
        # (Spark aggregations analogous to below; omitted for brevity on Databricks run)
    else:
        import duckdb
        con = duckdb.connect()
        G = REJECT_GLOB
        rej = f"read_parquet('{G}', hive_partitioning=true)"
        appr_path = APPROVED_CLEAN.replace("\\", "/")
        appr = f"read_parquet('{appr_path}')"

        n_appr = con.execute(f"SELECT COUNT(*) FROM {appr}").fetchone()[0]
        is_sample = n_appr < 600_000
        print(f"Approved rows: {n_appr:,} [{'SAMPLE' if is_sample else 'FULL'}]")
        if is_sample:
            print("[WARN] Approved parquet looks like a sample (<600k). Comparison is "
                  "against the sample; regenerate full parquet for the final comparison.")

        # amount: approved loan_amnt vs rejected amount_requested
        print("\n=== Amount: approved (loan_amnt) vs rejected (amount_requested) ===")
        a = con.execute(f"SELECT ROUND(AVG(loan_amnt),2), MEDIAN(loan_amnt) FROM {appr}").fetchone()
        r = con.execute(f"SELECT ROUND(AVG(amount_requested),2), MEDIAN(amount_requested) FROM {rej}").fetchone()
        print(f"   approved: mean={a[0]:,} median={a[1]:,}")
        print(f"   rejected: mean={r[0]:,} median={r[1]:,}")

        # dti (WITH the non-comparability caveat)
        print("\n=== DTI: approved vs rejected (NOT identical basis — see roadmap B) ===")
        a = con.execute(f"SELECT ROUND(AVG(dti),2), MEDIAN(dti) FROM {appr} WHERE dti BETWEEN 0 AND 100").fetchone()
        r = con.execute(f"SELECT ROUND(AVG(dti),2), MEDIAN(dti) FROM {rej} WHERE dti BETWEEN 0 AND 100").fetchone()
        print(f"   approved: mean={a[0]} median={a[1]}  (LC-computed, excl. mortgage)")
        print(f"   rejected: mean={r[0]} median={r[1]}  (application-time, denied apps)")

        # Geography: rejected side only this phase. Approved-side addr_state is not in
        # loans_clean.parquet (dropped in notebook 03 as a non-feature); comparing requires
        # regenerating it from the raw accepted CSV, deferred to Phase 2. See roadmap E.
        print("\n=== Rejected: top 10 states (approved comparison deferred to Phase 2) ===")
        state_null = con.execute(
            f"SELECT ROUND(100.0*(COUNT(*)-COUNT(state))/COUNT(*), 4) FROM {rej}"
        ).fetchone()[0]
        print(f"   state null rate (rejected): {state_null}%")
        rows = con.execute(
            f"SELECT state, COUNT(*) c FROM {rej} GROUP BY state ORDER BY c DESC LIMIT 10"
        ).fetchall()
        for st, c in rows:
            print(f"   {st}: {c:,}")
        print("   [NOTE] Approved-vs-rejected geography comparison deferred to Phase 2 "
              "(approved addr_state must be regenerated from raw CSV).")

        # vintage shape: rejected volume by year
        print("\n=== Rejected volume by application year (vintage) ===")
        for yr, c in con.execute(
            f"SELECT app_year, COUNT(*) FROM {rej} GROUP BY app_year ORDER BY app_year").fetchall():
            print(f"   {yr}: {c:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 — Provenance manifest
# MAGIC Writes a versioned manifest so the ingestion is auditable: source, license, measured
# MAGIC row count, file size, and timestamp. This is what makes the pipeline reproducible by
# MAGIC documenting the origin, instead of re-downloading the file every run.

# COMMAND ----------

import json
import datetime

manifest = {
    "artifact": "rejected_population_ingestion",
    "phase": "1",
    "source_dataset": "kaggle: wordsforthewise/lending-club",
    "source_file": "rejected_2007_to_2018Q4.csv.gz",
    "license": "CC0-1.0",
    "measured_row_count": int(n_reject),
    "raw_gz_size_bytes": int(size_bytes),
    "partitioned_by": "app_year",
    "written_to": reject_parquet,
    "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "parse_audit": PARSE_AUDIT,
    "note": "Descriptive ingestion + profile only. No labels exist for the rejected "
            "population by construction; no inference is performed in this phase.",
}
manifest["engine"] = {
    "ingestion_write": "PySpark (local pyarrow write; native Spark on Databricks)",
    "local_read_profile": "DuckDB over partitioned Parquet (hive_partitioning=true)",
    "rationale": "Spark for scale ingestion; DuckDB for single-node local analytics "
                 "without Hadoop/winutils. Each tool where it fits.",
}

manifest_path = (os.path.join(REPORTS, "reject_manifest.json") if not ON_DATABRICKS
                 else f"{REPORTS}/reject_manifest.json")

if ON_DATABRICKS:
    dbutils.fs.put(manifest_path, json.dumps(manifest, indent=2), overwrite=True)  # noqa: F821
else:
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

print("Manifest written to:", manifest_path)
print(json.dumps(manifest, indent=2))
