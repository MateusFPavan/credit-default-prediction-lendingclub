"""
CLI para src.monitor.monitor_batch (task C1.1, auditoria Boletim 12/13).

Carrega um lote de registros brutos de um CSV ou parquet, roda o monitor de
drift contra a baseline de treino, e imprime o veredito. Sai com codigo
diferente de zero em "drift_unexplained", pra um step de CI conseguir
alertar sobre isso.

Run with: python -m src.run_monitor --batch caminho/para/lote.csv
(ou .parquet)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from src.monitor import monitor_batch


def load_batch(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(
        description="Roda o monitor de drift PSI contra um lote de registros."
    )
    parser.add_argument(
        "--batch", required=True, type=Path,
        help="Caminho pra um CSV ou parquet com registros brutos (mesmo schema de data/processed/*.parquet).",
    )
    args = parser.parse_args()

    if not args.batch.exists():
        print(f"ERRO: lote nao encontrado em {args.batch}")
        sys.exit(2)

    batch = load_batch(args.batch)
    result = monitor_batch(batch)

    print(f"n={result['n']}  verdict={result['verdict']}")
    if "message" in result:
        print(result["message"])
    if result.get("bands"):
        print(f"bands={result['bands']}")
    if result.get("n_unexplained") is not None:
        print(f"n_unexplained={result['n_unexplained']}  score_psi={result.get('score_psi')}")
    if result.get("table") is not None:
        table = result["table"]
        unexplained = table[(table["psi"] > 0.25) & (table["cause"] == "")]
        if len(unexplained):
            print("Features com PSI critico sem causa conhecida:")
            print(unexplained[["feature", "psi", "band"]].to_string(index=False))

    if result["verdict"] == "drift_unexplained":
        print("RETRAIN TRIGGER: drift_unexplained -> politica de retreino deveria disparar "
              "(ver docs/MODEL_CARD.md secao 10).")
        sys.exit(1)
    print("OK: sem trigger de retreino.")


if __name__ == "__main__":
    main()
