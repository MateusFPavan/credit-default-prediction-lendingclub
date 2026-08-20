"""
Drift monitor for production (task 4.4).

Thin extension of the Phase 1 PSI -- does NOT reimplement the engine. Reuses
fit_binnings/psi_table/compute_psi/psi_band from src.psi and the scoring
pipeline from src.run_psi. What's new here is only this:

  1. drift of an incoming BATCH (raw records, as the API receives them) against the
     training baseline, going through the same clean_record/build_features/score
     Phase 2 uses -- instead of comparing static parquets against each other.
  2. drift of the output SCORE (distribution of the batch's probabilities vs training).
  3. a SAMPLE FLOOR: below N rows, PSI is noise, so the monitor refuses
     to give a verdict instead of reporting a spurious number. The floor is measured
     empirically (see the verification block), not guessed.

Interpretation: high PSI is not always an alarm. The project already documents (scope §8,
reports/psi/README) that initial_list_status has PSI~0.48 from a GENUINE policy
change (fractional->whole), while the 2012 rollout sentinel inflates PSI in a
SPURIOUS way. The monitor annotates the cause when known, to separate real
drift from artifact -- the maturity differentiator of Project 4.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import load_split, FEATURE_SET, CATEGORICAL_COLS
from src.psi import fit_binnings, psi_table, compute_psi, psi_band, STABLE_MAX, ATTENTION_MAX
from src import run_psi as _rp

# sample floor: filled in by empirical verification (block 8c). Until then, a
# conservative default value; block 8c confirms/adjusts and records the measured value here.
MIN_BATCH_FOR_PSI = 200  # measured (block 9c)

# score column name, inherited from run_psi (not transcribed)
SCORE_COL = getattr(_rp, "SCORE_COL", "model_score")

# known causes of high PSI (scope §8 / reports/psi/README) -- interpretive annotation
KNOWN_CAUSES = {
    # REAL but expected drift: platform policy change (scope §8)
    "initial_list_status": "GENUINE policy drift (fractional->whole, 82%->41%); "
                           "real signal, not artifact",
    # NOT drift: definitional artifact of the splits' temporal cut. train/val/test
    # are cut by issue_d, so high PSI is guaranteed by construction and is not
    # actionable (reports/psi/README documents this as 'definitional, not drift').
    "issue_d": "definitional artifact: splits are cut by issue_d, PSI is high by "
               "construction, not actionable drift",
    "earliest_cr_line": "correlated to the temporal cut by issue_d; high PSI is "
                        "a reflection of the same definitional artifact, not actionable drift",
}


def _prepare_baseline():
    """
    Baseline = Phase 1's CLEAN view: train with build_features + encode_datetime_cols,
    scored, and ONLY THEN filtered by era_pre_2012 == 0 (same order as run_psi.py:
    train_clean_enc = train_enc[train_enc["era_pre_2012"] == 0]).

    Clean, not raw: every production batch is 100% era_pre_2012==0 (Block 6's decision),
    so comparing against the raw baseline (30% pre-2012) would inflate the PSI of the ~26
    rollout columns by construction, with zero real drift. Clean is the comparison that makes sense.
    """
    df = load_split("train")
    df_feat = _rp.build_features(df)
    scored = _rp.score_df(_rp_model(), df_feat)
    base = _rp.encode_datetime_cols(df_feat.copy())
    if SCORE_COL not in base.columns:
        base[SCORE_COL] = scored
    # CLEAN filter -- after features/encode/score, same as run_psi
    base = base[base["era_pre_2012"] == 0].copy()
    return base

def _rp_model():
    from src.scoring import load_model
    return load_model()


def monitor_batch(batch: pd.DataFrame, baseline: pd.DataFrame | None = None) -> dict:
    """
    Computes PSI of a batch of records (raw or already clean) against the
    training baseline, feature by feature + the score. Returns a dict with the table and the verdict.

    If the batch is smaller than MIN_BATCH_FOR_PSI, returns verdict 'insufficient_sample'
    without PSI numbers (which would be noise).
    """
    from src.scoring import score_frame
    from src.cleaning import clean_record
    from src.features import build_features

    n = len(batch)
    if n < MIN_BATCH_FOR_PSI:
        return {
            "n": n, "verdict": "insufficient_sample",
            "message": (f"batch of {n} rows < floor of {MIN_BATCH_FOR_PSI}; "
                        "PSI would be noise. Accumulate more records before evaluating drift."),
            "table": None,
        }

    if baseline is None:
        baseline = _prepare_baseline()

    # prepare the batch through the SAME path as production
    b = clean_record(batch.copy())
    b = build_features(b)
    b = _rp.encode_datetime_cols(b) if hasattr(_rp, "encode_datetime_cols") else b
    b[SCORE_COL] = score_frame(batch.copy())["probability_default"].to_numpy()

    feature_cols = [c for c in FEATURE_SET if c in baseline.columns and c in b.columns]
    feature_cols_with_score = feature_cols + ([SCORE_COL] if SCORE_COL in baseline.columns else [])

    binnings = fit_binnings(baseline, feature_cols_with_score, categorical_cols=CATEGORICAL_COLS)
    table = psi_table(binnings, baseline, b, feature_cols_with_score)

    # annotate known cause and classify
    if "feature" in table.columns and "cause" not in table.columns:
        table["cause"] = table["feature"].map(KNOWN_CAUSES).fillna("")

    # aggregated verdict: how many features in each band
    bands = table["band"].value_counts().to_dict() if "band" in table.columns else {}
    n_critical = int((table["psi"] > ATTENTION_MAX).sum()) if "psi" in table.columns else 0
    # critical WITHOUT known cause = the ones that deserve real attention
    unexplained = table[(table["psi"] > ATTENTION_MAX) & (table["cause"] == "")] \
        if {"psi", "cause"}.issubset(table.columns) else table.iloc[0:0]

    verdict = "stable"
    if len(unexplained) > 0:
        verdict = "drift_unexplained"
    elif n_critical > 0:
        verdict = "drift_explained"  # there's high PSI, but with a known cause (e.g., policy)

    score_psi = None
    if SCORE_COL in feature_cols_with_score and "feature" in table.columns:
        row = table[table["feature"] == SCORE_COL]
        if len(row): score_psi = float(row["psi"].iloc[0])

    return {
        "n": n, "verdict": verdict, "bands": bands,
        "n_critical": n_critical, "n_unexplained": int(len(unexplained)),
        "score_psi": score_psi, "table": table,
    }
