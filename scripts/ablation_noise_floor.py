"""Noise floor of leave-one-feature-out ablation, and what survives it.

WHY THIS EXISTS
---------------
build_xgb_final() freezes colsample_bytree=0.6: each of the 600 trees is grown on a
random 60% of the columns, drawn from random_state AND from the column count. So
dropping any column -- even one the model never splits on -- re-randomises the whole
ensemble. A leave-one-feature-out (LOFO) delta is therefore "feature + re-draw", and
for an inert feature it is re-draw only.

This script measures the re-draw term and re-reads the LOFO ranking against it.

WHAT IT DOES
------------
1. TWIN TEST. Two features -- num_tl_120dpd_2m_missing and sparse_bureau_missing --
   returned the same profit, AUC and threshold to full float precision. The script
   refits both ablations and compares the boosters directly, not their metrics.
2. NULL BY SEED. Refits the frozen configuration with eight seeds, same rows, same
   columns. The spread is the fit-to-fit noise floor.
3. RE-READ. Recentres the 78 LOFO deltas on their median -- because the baseline is
   one draw, not the mean, so the expected delta is regression to the mean, not zero
   -- and reports what is still outside the noise.

It does NOT pick a seed. The published figure stays random_state=42; this measures
the uncertainty around it. Picking the best of N would be exactly the unadjusted
multiple comparison this project's own rules forbid.

USAGE
-----
    python scripts/ablation_noise_floor.py            # ~11 fits, about 10 minutes
    python scripts/ablation_noise_floor.py --lofo     # also regenerates the 78-row
                                                      # CSV; ~80 fits, about an hour

Run from the repository root: imports and reports/ are resolved from the working
directory.
"""
import csv
import os
import statistics as st
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.getcwd())

from src.data import load_split, FEATURE_SET, CATEGORICAL_COLS
from src.economics import compute_interest_loss, profit_at_threshold
from src.features import build_features, prepare_X
from src.models import build_xgb_final

PUBLISHED = 242230710.89
SEEDS = [42, 0, 1, 7, 13, 99, 123, 2024]
TWIN_A = "num_tl_120dpd_2m_missing"
TWIN_B = "sparse_bureau_missing"
LOFO_CSV = os.path.join("reports", "lofo_deltas.csv")

train, val, test = load_split("train"), load_split("validation"), load_split("test")
trf, vf, tf = build_features(train), build_features(val), build_features(test)
y_tr, y_v, y_te = trf["target"].values, vf["target"].values, tf["target"].values
iv, lv = compute_interest_loss(vf)
it, lt = compute_interest_loss(tf)
GRID = np.arange(0.05, 0.95, 0.01)


def fit_and_score(feats, seed=42):
    """Train on TRAIN, pick the threshold on VALIDATION, evaluate on TEST."""
    cats = [c for c in CATEGORICAL_COLS if c in feats]
    Xa = prepare_X(trf, list(feats), cats)
    model = build_xgb_final()
    model.set_params(random_state=seed)
    model.fit(Xa, y_tr)
    Xv = prepare_X(vf, list(feats), cats).reindex(columns=Xa.columns, fill_value=0)
    Xt = prepare_X(tf, list(feats), cats).reindex(columns=Xa.columns, fill_value=0)
    pv = model.predict_proba(Xv)[:, 1]
    pt = model.predict_proba(Xt)[:, 1]
    thr = float(GRID[int(np.argmax(
        [profit_at_threshold(y_v, pv, t, iv.values, lv.values) for t in GRID]))])
    return {"model": model, "cols": list(Xa.columns), "pt": pt, "thr": thr,
            "auc": roc_auc_score(y_te, pt),
            "profit": profit_at_threshold(y_te, pt, thr, it.values, lt.values)}


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def regenerate_lofo():
    """Re-run the full 78-feature LOFO and rewrite reports/lofo_deltas.csv (~1 hour)."""
    rows = []
    for i, feat in enumerate(FEATURE_SET, 1):
        r = fit_and_score([f for f in FEATURE_SET if f != feat], seed=42)
        rows.append({"feature": feat, "delta_profit": r["profit"] - base["profit"],
                     "delta_auc": r["auc"] - base["auc"], "profit": r["profit"],
                     "auc": r["auc"], "thr": r["thr"]})
        print("  [%2d/%d] %-34s delta=%15.2f" % (i, len(FEATURE_SET), feat, rows[-1]["delta_profit"]))
        with open(LOFO_CSV, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["feature", "delta_profit", "delta_auc",
                                               "profit", "auc", "thr"])
            w.writeheader()
            w.writerows(rows)


rule("GATE 0 - does the frozen configuration still reproduce the published figure?")
base = fit_and_score(FEATURE_SET, seed=42)
print("  base: %d columns  thr=%.2f  AUC=%.6f  profit=%.2f"
      % (len(base["cols"]), base["thr"], base["auc"], base["profit"]))
print("  against published %.2f: difference = %.2f" % (PUBLISHED, base["profit"] - PUBLISHED))
if abs(base["profit"] - PUBLISHED) > 1.0:
    print("  STOP: the baseline does not reproduce. Nothing below is comparable.")
    sys.exit(1)

if "--lofo" in sys.argv:
    rule("LOFO - regenerating %s" % LOFO_CSV)
    regenerate_lofo()

rule("1. TWIN TEST - two ablations that agreed to the last digit")
no_a = fit_and_score([f for f in FEATURE_SET if f != TWIN_A], seed=42)
no_b = fit_and_score([f for f in FEATURE_SET if f != TWIN_B], seed=42)
for label, r in ((TWIN_A, no_a), (TWIN_B, no_b)):
    print("  without %-28s %d cols  thr=%.2f  AUC=%.16f  profit=%.2f"
          % (label, len(r["cols"]), r["thr"], r["auc"], r["profit"]))
print("  predictions identical bitwise : %s" % np.array_equal(no_a["pt"], no_b["pt"]))
print("  max|p_a - p_b|                : %.3e"
      % float(np.max(np.abs(no_a["pt"] - no_b["pt"]))))
print("  tree dumps identical          : %s"
      % (no_a["model"].get_booster().get_dump() == no_b["model"].get_booster().get_dump()))
base_splits = base["model"].get_booster().get_score(importance_type="weight")
for r, survivor in ((no_a, TWIN_B), (no_b, TWIN_A)):
    sc = r["model"].get_booster().get_score(importance_type="weight")
    print("    splits on %-28s in that fit = %s" % (survivor, sc.get(survivor, 0)))
for c in (TWIN_A, TWIN_B):
    print("    splits on %-28s in the base  = %s" % (c, base_splits.get(c, 0)))
print("  Reading: the two 89-column matrices differ in ONE column, at the same index,")
print("  and that column is never split on in either fit -- so the two fits are the")
print("  same ensemble. The shared delta measures the re-draw, not either feature.")

rule("2. NULL BY SEED - same rows, same columns, only the draw changes")
profits = []
for seed in SEEDS:
    r = base if seed == 42 else fit_and_score(FEATURE_SET, seed=seed)
    profits.append(r["profit"])
    print("  seed %-6d thr=%.2f  AUC=%.6f  profit=%15.2f" % (seed, r["thr"], r["auc"], r["profit"]))
profits = np.array(profits)
mean, sd = float(profits.mean()), float(profits.std(ddof=1))
rank42 = int(np.sum(profits > base["profit"])) + 1
print("  mean / sd (n-1)        : %15.2f / %.2f" % (mean, sd))
print("  min / max / range      : %15.2f / %.2f / %.2f"
      % (profits.min(), profits.max(), profits.max() - profits.min()))
print("  seed 42 rank           : %d of %d (highest first), %+0.2f sd from the mean"
      % (rank42, len(profits), (base["profit"] - mean) / sd))
print("  predicted regression to the mean for any fresh draw: %.2f" % (mean - base["profit"]))
print("  NOTE: changing the seed redraws columns AND rows (subsample=0.8), while")
print("  dropping a column redraws columns only. This sd is an UPPER bound on the")
print("  re-draw noise: a delta above it is signal with room to spare; a delta below")
print("  it is NOT thereby proven to be noise.")

rule("3. RE-READING THE LOFO RANKING AGAINST THE NULL")
if not os.path.exists(LOFO_CSV):
    print("  %s not found -- run with --lofo to generate it." % LOFO_CSV)
    sys.exit(0)
deltas = []
with open(LOFO_CSV, newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        deltas.append((row["feature"], float(row["delta_profit"])))
values = [d for _, d in deltas]
median = st.median(values)
print("  n                      : %d" % len(values))
print("  median delta           : %15.2f   <- predicted above as %.2f" % (median, mean - base["profit"]))
print("  mean delta             : %15.2f" % st.mean(values))
print("  sd of the deltas       : %15.2f" % st.stdev(values))
print()
print("  centred on ZERO   , |delta| > 2 sd : %d of %d  (negative: %d)"
      % (sum(1 for v in values if abs(v) > 2 * sd), len(values),
         sum(1 for v in values if v < -2 * sd)))
print("  centred on MEDIAN , |delta - median| > 2 sd : %d of %d"
      % (sum(1 for v in values if abs(v - median) > 2 * sd), len(values)))
print()
print("  what survives, recentred on the median:")
survivors = sorted([(f, d) for f, d in deltas if abs(d - median) > 2 * sd], key=lambda t: t[1])
if not survivors:
    print("      nothing")
for f, d in survivors:
    print("      %-40s %15.2f   %+0.2f sd" % (f, d, (d - median) / sd))
print()
print("  A ranking in which almost everything falls on one side is measuring the")
print("  shift, not the effect. Centred on zero the survivors are all negative;")
print("  centred on the median, features whose REMOVAL raises profit appear too.")
