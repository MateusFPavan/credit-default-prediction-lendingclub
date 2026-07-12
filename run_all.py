"""Single-command reproduction entry point for credit-default-prediction-lendingclub.

This reproduces the final model and its test result from raw data. The model-selection
experiments (baseline comparison, tuning, bootstrap) live in notebooks 06-11 and are
documented in docs/FACTS.md; they are not re-run here because they take hours and are not
needed to reproduce the delivered model.

Essential path only: raw CSV -> cleaned dataset -> temporal split -> features -> final
model -> test-result verification. Notebook 01 (data understanding) and notebook 02
(cleaning diagnostics) are intentionally skipped - both are exploratory and neither is a
prerequisite of notebook 03: 03 reads the raw CSV directly (confirmed by inspection) and
re-implements docs/cleaning_decisions.md on its own; 02's only file output is
docs/outliers_multivariados.csv, a diagnostic export nothing downstream reads.

Uses only relative paths (all paths are derived from this file's own location, never a
hardcoded absolute path).

Usage:
    python run_all.py

Prerequisite: data/raw/accepted_2007_to_2018Q4.csv must already be present (see the
error message below for where to get it - this script never downloads it automatically,
since that requires a Kaggle account/credential).

Run with the project's own virtual environment active (or invoke this script with that
venv's python interpreter directly), since it shells out to `python -m jupyter nbconvert`
using whatever interpreter is currently running this script (sys.executable).
"""
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
RAW_CSV = REPO_ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv"
KERNEL_NAME = "credit-default-prediction-lendingclub"
NOTEBOOK_TIMEOUT = 1800  # seconds; essential-path notebooks are light (no tuning loops)

REFERENCE_PROFIT = "242,230,710.89"  # for the human-readable message only; the real
# comparison is done inside src/verify_pipeline.py


def run_notebook(filename, timeout=NOTEBOOK_TIMEOUT):
    """Execute one notebook in place via nbconvert. Raises RuntimeError on failure."""
    rel_path = f"notebooks/{filename}"
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook", "--execute", "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        f"--ExecutePreprocessor.kernel_name={KERNEL_NAME}",
        rel_path,
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"notebook {filename} falhou (exit code {proc.returncode}).\n"
            f"--- stderr (final) ---\n{proc.stderr[-4000:]}"
        )


def step_feature_engineering():
    """Equivalent of notebook 05: applies src.features.build_features to the four
    processed splits and re-saves them in place. Done directly via src/ instead of
    re-executing the notebook - same function, much faster."""
    import pandas as pd
    from src.features import build_features

    processed_dir = REPO_ROOT / "data" / "processed"
    for name in ["train", "validation", "test", "transfer_60m"]:
        path = processed_dir / f"{name}.parquet"
        df = pd.read_parquet(path)
        df_feat = build_features(df)
        df_feat.to_parquet(path, index=False)
        print(f"  {name}.parquet: {df.shape} -> {df_feat.shape}")


def step_verify():
    """Runs src/verify_pipeline.py as a subprocess: trains the frozen XGB_walkforward
    config on train.parquet using only src/, scores the test set, and checks the profit
    at threshold 0.31 reproduces $242,230,710.89 exactly. Raises RuntimeError on any
    mismatch or failure - src/verify_pipeline.py itself does the exact-match check and
    exits non-zero if it fails."""
    proc = subprocess.run(
        [sys.executable, "-m", "src.verify_pipeline"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Verificacao de integridade FALHOU (exit code {proc.returncode}).\n"
            f"O lucro reproduzido nao bateu com o valor de referencia (${REFERENCE_PROFIT}).\n"
            f"--- stderr ---\n{proc.stderr[-4000:]}"
        )


# Fixed, single path - no branching. Exactly: build processed -> temporal split ->
# feature engineering -> train & serialize final model -> verify.
STEPS = [
    {"id": "03", "kind": "notebook", "file": "03_build_processed.ipynb",
     "desc": "Build processed dataset (loans_clean.parquet)"},
    {"id": "04", "kind": "notebook", "file": "04_temporal_split.ipynb",
     "desc": "Temporal split (train/validation/test/transfer_60m)"},
    {"id": "05", "kind": "src", "func": step_feature_engineering,
     "desc": "Feature engineering (src/features.build_features, applied+saved directly, replaces notebook 05)"},
    {"id": "14", "kind": "notebook", "file": "14_train_final_model.ipynb",
     "desc": "Train & serialize final model (models/*.joblib, models/model_meta.json)"},
    {"id": "verify", "kind": "src", "func": step_verify,
     "desc": "Integrity verification (src/verify_pipeline.py) - mandatory final step"},
]


def main():
    print("=" * 70)
    print("credit-default-prediction-lendingclub - reproducao do caminho essencial")
    print("=" * 70)

    if not RAW_CSV.exists():
        print()
        print("ERRO: arquivo de dados bruto nao encontrado em:")
        print(f"  {RAW_CSV.relative_to(REPO_ROOT)}")
        print()
        print("Baixe o dataset Lending Club no Kaggle:")
        print("  https://www.kaggle.com/datasets/wordsforthewise/lending-club")
        print("(requer conta/API key do Kaggle - este script nao baixa automaticamente).")
        print()
        print("Depois de baixar, coloque o arquivo como:")
        print(f"  {RAW_CSV.relative_to(REPO_ROOT)}")
        print("e rode este script novamente.")
        sys.exit(1)

    total_start = time.time()
    step_times = []

    for i, step in enumerate(STEPS, start=1):
        print()
        print(f"[{i}/{len(STEPS)}] Etapa {step['id']}: {step['desc']}")
        t0 = time.time()
        try:
            if step["kind"] == "notebook":
                run_notebook(step["file"])
            else:
                step["func"]()
        except RuntimeError as e:
            elapsed = time.time() - t0
            print(f"  FALHOU apos {elapsed:.1f}s.")
            print()
            print("=" * 70)
            print(f"PIPELINE PAROU na etapa {step['id']} ({step['desc']}).")
            print("=" * 70)
            print(str(e))
            sys.exit(1)
        elapsed = time.time() - t0
        step_times.append((step["id"], step["desc"], elapsed))
        print(f"  OK em {elapsed:.1f}s ({elapsed / 60:.1f} min).")

    total_elapsed = time.time() - total_start

    print()
    print("=" * 70)
    print("RESUMO")
    print("=" * 70)
    for step_id, desc, elapsed in step_times:
        print(f"  {step_id:>6}  {elapsed:8.1f}s  {desc}")
    print(f"\nTempo total: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print()
    print("Pipeline reproduzido com sucesso. O lucro no teste (XGB_walkforward, "
          f"threshold 0.31) bateu exatamente com ${REFERENCE_PROFIT}.")


if __name__ == "__main__":
    main()
