"""Within-segment AUC of the published model against the logistic baseline.

WHY THIS EXISTS
---------------
The model's AUC inside each Lending Club grade is 0.648 in grade A and between 0.582
and 0.605 in grades B to G, and it is lower for lower-income borrowers. Read alone,
that looks like a model that gets weaker where risk is higher - a defect worth
mitigating, for example by re-weighting the training data.

A within-segment AUC also depends on the segment itself: how varied its borrowers
are, and how much of their risk the available features can see. The test of which
of the two is going on is to run a second, unrelated model through the same cut. If
the logistic baseline - no sampling, no trees - shows the same gradient, the gradient
belongs to the population, and re-weighting the XGB would not remove it.

WHAT IT DOES
------------
0. GATE. Scores the test set with the published artifacts and checks the per-segment
   AUCs against reports/facts/facts_subgroups.csv and the baseline's overall AUC
   against the published 0.6847. Stops on any mismatch.
1. Per segment (7 grades, 4 income quartiles): XGB AUC, baseline AUC, and their
   difference, with a paired bootstrap inside each segment. Intervals are reported
   at 95% and, because each dimension is a family of comparisons, Bonferroni-adjusted
   (normal approximation from the bootstrap standard error).
2. The gradient in each model - grade A minus grade B, income Q4 minus Q1 - and the
   difference between the two models' gradients. If that difference's interval
   crosses zero, both models share the gradient.
3. Fit-to-fit noise: the frozen XGB configuration refitted with eight seeds, same
   rows and columns, and the per-segment AUC of each refit.

It writes reports/subgroup_auc_vs_baseline.txt and
reports/figures/subgroup_auc_vs_baseline.png itself. Nothing else is modified.

USAGE
-----
    python scripts/subgroup_auc_vs_baseline.py      # about 10 minutes

Run from the repository root.
"""
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.getcwd())

from src.data import load_split, FEATURE_SET, CATEGORICAL_COLS
from src.features import build_features, prepare_X
from src.models import build_xgb_final

OUT_TXT = os.path.join("reports", "subgroup_auc_vs_baseline.txt")
OUT_PNG = os.path.join("reports", "figures", "subgroup_auc_vs_baseline.png")
SEEDS = [42, 0, 1, 7, 13, 99, 123, 2024]
GRADES = list("ABCDEFG")
QUARTILES = ["Q1", "Q2", "Q3", "Q4"]
PUBLISHED = {"A": 0.6477, "B": 0.6048, "C": 0.5965, "D": 0.5889, "E": 0.5818,
             "F": 0.5878, "G": 0.5854,
             "Q1": 0.6480, "Q2": 0.6722, "Q3": 0.6881, "Q4": 0.6974}
M1_PUBLISHED_AUC = 0.6847
N_BOOT = 500
Z_BONF = {"grade": 2.6901, "income": 2.4977}   # two-sided, alpha 0.05 / 7 and / 4
rng = np.random.default_rng(20260930)

_fh = open(OUT_TXT, "w", encoding="utf-8", newline="\n")


def out(text=""):
    print(text)
    _fh.write(text + "\n")
    _fh.flush()


def rule(title):
    out()
    out("=" * 78)
    out(title)
    out("=" * 78)


train = build_features(load_split("train"))
test = build_features(load_split("test"))
Xa = prepare_X(train, FEATURE_SET, CATEGORICAL_COLS)
cols = Xa.columns
Xt = prepare_X(test, FEATURE_SET, CATEGORICAL_COLS).reindex(columns=cols, fill_value=0)
y_tr = train["target"].values
y = test["target"].values
grade = test["grade"].astype(str).str.upper().values
quartile = pd.qcut(test["annual_inc"], q=4, labels=QUARTILES).astype(str).values
seg_idx = {g: np.where(grade == g)[0] for g in GRADES}
seg_idx.update({q: np.where(quartile == q)[0] for q in QUARTILES})
FAMILIES = (("grade", GRADES), ("income", QUARTILES))


def auc_by_segment(p):
    return {s: roc_auc_score(y[seg_idx[s]], p[seg_idx[s]]) for s in GRADES + QUARTILES}


p_xgb = joblib.load(os.path.join("models", "xgb_final.joblib")).predict_proba(Xt)[:, 1]
p_m1 = joblib.load(os.path.join("models", "logistic_baseline.joblib")).predict_proba(Xt)[:, 1]

rule("GATE 0 - do the published artifacts reproduce the published figures?")
a_xgb = auc_by_segment(p_xgb)
failed = False
for s in GRADES + QUARTILES:
    ok = abs(a_xgb[s] - PUBLISHED[s]) <= 1e-4
    failed = failed or not ok
    out("  XGB %-3s : %.4f  published %.4f  %s" % (s, a_xgb[s], PUBLISHED[s], "ok" if ok else "MISMATCH"))
m1_overall = roc_auc_score(y, p_m1)
ok = abs(m1_overall - M1_PUBLISHED_AUC) <= 1e-4
failed = failed or not ok
out("  baseline overall AUC: %.4f  published %.4f  %s" % (m1_overall, M1_PUBLISHED_AUC, "ok" if ok else "MISMATCH"))
if failed:
    out("  STOP: the artifacts do not reproduce the published figures. Nothing below is comparable.")
    _fh.close()
    sys.exit(1)

a_m1 = auc_by_segment(p_m1)
boot = {s: {"xgb": [], "m1": [], "diff": []} for s in GRADES + QUARTILES}
for s in GRADES + QUARTILES:   # grades first: keeps the grade draws identical to the first run
    ig = seg_idx[s]
    for _ in range(N_BOOT):
        r = rng.choice(ig, size=len(ig), replace=True)
        if y[r].min() == y[r].max():
            continue
        ax = roc_auc_score(y[r], p_xgb[r])
        am = roc_auc_score(y[r], p_m1[r])
        boot[s]["xgb"].append(ax)
        boot[s]["m1"].append(am)
        boot[s]["diff"].append(ax - am)


def ci95(v):
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


for family, segs in FAMILIES:
    rule("1. %s: published XGB against the logistic baseline (bootstrap %d x, paired)"
         % (family.upper(), N_BOOT))
    out("  %-4s %7s %8s | %-21s | %-21s | %-24s | %s"
        % ("seg", "n", "default", "XGB [95% CI]", "baseline [95% CI]",
           "XGB - baseline [95% CI]", "Bonferroni"))
    for s in segs:
        n = len(seg_idx[s])
        dr = y[seg_idx[s]].mean() * 100
        lx, hx = ci95(boot[s]["xgb"])
        lm, hm = ci95(boot[s]["m1"])
        ld, hd = ci95(boot[s]["diff"])
        d = a_xgb[s] - a_m1[s]
        half = Z_BONF[family] * np.std(boot[s]["diff"], ddof=1)
        out("  %-4s %7d %7.2f%% | %.4f [%.3f, %.3f] | %.4f [%.3f, %.3f] | %+.4f [%+.3f, %+.3f] | [%+.3f, %+.3f]"
            % (s, n, dr, a_xgb[s], lx, hx, a_m1[s], lm, hm, d, ld, hd, d - half, d + half))
    out("  Bonferroni column: %d comparisons in this family, normal approximation from the"
        % len(segs))
    out("  bootstrap standard error. A segment where XGB beats the baseline should clear zero there.")

rule("2. THE GRADIENT IN EACH MODEL, and whether the two models share it")
for label, hi, lo in (("grade A minus grade B", "A", "B"), ("income Q4 minus Q1", "Q4", "Q1")):
    k = min(len(boot[hi]["xgb"]), len(boot[lo]["xgb"]))
    for name, key, a in (("XGB     ", "xgb", a_xgb), ("baseline", "m1", a_m1)):
        g = np.array(boot[hi][key][:k]) - np.array(boot[lo][key][:k])
        l, h = ci95(g)
        out("  %-22s %s %+.4f  95%% CI [%+.3f, %+.3f]" % (label, name, a[hi] - a[lo], l, h))
    dd = (np.array(boot[hi]["xgb"][:k]) - np.array(boot[lo]["xgb"][:k])) - \
         (np.array(boot[hi]["m1"][:k]) - np.array(boot[lo]["m1"][:k]))
    l, h = ci95(dd)
    out("  %-22s XGB gradient minus baseline gradient = %+.4f  95%% CI [%+.3f, %+.3f]"
        % (label, (a_xgb[hi] - a_xgb[lo]) - (a_m1[hi] - a_m1[lo]), l, h))
    out()
out("  If the last line of a pair has an interval crossing zero, both models carry the")
out("  same gradient: it belongs to the population, not to the XGB.")

rule("3. FIT-TO-FIT NOISE - eight seeds, same rows and columns")
per_seed = {}
for seed in SEEDS:
    m = build_xgb_final()
    m.set_params(random_state=seed)
    m.fit(Xa, y_tr)
    p = m.predict_proba(Xt)[:, 1]
    if seed == 42:
        out("  seed 42 refit identical to the published artifact, bit for bit: %s"
            % np.array_equal(p, p_xgb))
    per_seed[seed] = auc_by_segment(p)
    out("  seed %-5d " % seed + "  ".join("%s %.4f" % (s, per_seed[seed][s]) for s in GRADES + QUARTILES))
out()
out("  %-4s %8s %8s %8s %8s   %s" % ("seg", "mean", "sd", "min", "max", "published (seed 42) rank, 1 = highest"))
for s in GRADES + QUARTILES:
    v = np.array([per_seed[x][s] for x in SEEDS])
    rank = int(np.sum(v > per_seed[42][s])) + 1
    out("  %-4s %8.4f %8.4f %8.4f %8.4f   %d of %d" % (s, v.mean(), v.std(ddof=1), v.min(), v.max(), rank, len(SEEDS)))
for label, hi, lo in (("A - B", "A", "B"), ("Q4 - Q1", "Q4", "Q1")):
    g = np.array([per_seed[x][hi] - per_seed[x][lo] for x in SEEDS])
    out("  gradient %-7s across seeds: mean %+.4f, min %+.4f, max %+.4f" % (label, g.mean(), g.min(), g.max()))

# ---- figure ----------------------------------------------------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2"
C_XGB, C_M1 = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True,
                         gridspec_kw={"width_ratios": [7, 4]}, facecolor=SURFACE)
w = 0.38
top = 0.0
for ax, (family, segs), title in zip(axes, FAMILIES, ("By Lending Club grade (A = safest)",
                                                      "By income quartile (Q1 = lowest)")):
    ax.set_facecolor(SURFACE)
    x = np.arange(len(segs))
    for off, key, a, color, name in ((-w / 2, "xgb", a_xgb, C_XGB, "XGB (published model)"),
                                     (w / 2, "m1", a_m1, C_M1, "Logistic baseline")):
        vals = np.array([a[s] for s in segs])
        lo_ci = np.array([ci95(boot[s][key])[0] for s in segs])
        hi_ci = np.array([ci95(boot[s][key])[1] for s in segs])
        top = max(top, hi_ci.max())
        ax.bar(x + off, vals - 0.5, width=w, bottom=0.5, color=color, edgecolor=SURFACE,
               linewidth=2, label=name if family == "grade" else None, zorder=2)
        ax.errorbar(x + off, vals, yerr=[vals - lo_ci, hi_ci - vals], fmt="none",
                    ecolor=INK2, elinewidth=1, capsize=2, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n%s" % (s, format(len(seg_idx[s]), ",")) for s in segs])
    ax.set_title(title, color=INK, fontsize=11, pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", length=0)
axes[0].set_ylim(0.50, np.ceil((top + 0.015) / 0.025) * 0.025)
axes[0].set_ylabel("AUC-ROC within segment (0.5 = chance)")
fig.suptitle("Within-segment AUC: published XGB against a logistic baseline", x=0.055,
             y=0.975, ha="left", fontsize=13, fontweight="bold", color=INK)
fig.text(0.055, 0.898, "Test set (2015 loans). Error bars: 95% bootstrap intervals. "
         "Under each label: loans in the test set.", ha="left", fontsize=9.5, color=INK2)
fig.legend(loc="upper left", bbox_to_anchor=(0.048, 0.885), ncol=2, frameon=False, fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, 0.875))
fig.savefig(OUT_PNG, dpi=150, facecolor=SURFACE)
out()
out("figure: %s" % OUT_PNG)
_fh.close()
