# ============================================================================
# Phase 2a: demonstration of the illusion of improvement in reject inference
# ============================================================================
# Purpose
#   Show, with the Lending Club data itself, why reject inference (RI) cannot be validated
#   on this dataset and why any reported improvement is likely an evaluation artifact.
#
# Why RI cannot be validated here (three measured reasons)
#   1. Weak shared signal: the thin model (features common to approved and rejected
#      applications) has an out-of-sample AUC of 0.5620 +/- 0.0027 (4-fold CV, notebook
#      17), barely above 0.5. Any label inferred from this score is close to noise.
#   2. No outcome for rejected applications: no column in the repository holds a reject
#      label. Kickout/AUK, the standard RI metric, requires labeled rejects and therefore
#      cannot be computed (see docs/reject_inference.md).
#   3. No unbiased sample and no true population default rate: the metric that escapes the
#      illusion (distortion of the training default rate relative to the population rate,
#      Illusion of Improvement, arXiv 2606.18479) needs the true population rate as a
#      reference. Lending Club does not have it, because rejected applications never
#      became loans. Kozodoi (arXiv 1909.06108) avoids the problem with an unbiased sample
#      of 1,967 cases (rejects accepted on purpose to observe their outcome); there is no
#      equivalent here. The Illusion paper concludes that evaluating RI on approved loans
#      produces an illusion of improvement: extrapolation outperforms even an Oracle with
#      true labels, because evaluation on the accepted pool rewards extreme decision
#      boundaries. Any gain reported on approved loans is therefore a likely artifact, not
#      a real improvement.
#
# Inputs
#   Approved loans (split 'train') and rejected applications (partitioned Parquet,
#   27,648,741 rows), loaded through the shared loaders of notebooks/17_reject_thin_model.py.
#
# Method
#   Rather than measuring which strategy wins (parcelling vs baseline), the script measures
#   the symptom of the artifact that the Illusion paper identifies: the default rate of the
#   augmented training set (observed approved loans plus rejects with parcelling labels) is
#   inflated relative to the observed rate of approved loans, and the inflation grows with
#   base_mult (the more aggressive the RI, the larger the inflation and the stronger the
#   symptom).
#
# Measured result (10 bands; approved default rate 12.43%)
#   base_mult   training rate after RI   inflation vs approved
#     1.0              0.1417                  +0.0173
#     1.5              0.2121                  +0.0878
#     2.0              0.2825                  +0.1581
#     2.5              0.3530                  +0.2287
#     3.0              0.4235                  +0.2992
#   The inflation grows monotonically and roughly linearly with base_mult.
#
# Known limitations
#   Without the true population rate, this inflation cannot be interpreted as a bias
#   correction: it may be one, or it may be a new bias. This is why RI cannot be validated
#   here, not because parcelling does not work: parcelling is implemented and runs
#   (notebook 17), but this data does not support a valid evaluation of it.
#
# References: Illusion of Improvement (Scarone & Baeza-Yates, arXiv 2606.18479, ECML
# PKDD 2026); Kozodoi (arXiv 1909.06108, 2407.13009); Hugo Lopes 2018.

import sys
import importlib.util
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# notebooks/17_reject_thin_model.py is loaded as a module (its name starts with a digit, so
# it cannot be imported with `import`); same pattern as notebooks/18_profit_evaluation.py.
_NB17_PATH = Path(__file__).resolve().parent / "17_reject_thin_model.py"
_spec = importlib.util.spec_from_file_location("nb17_reject_thin_model", _NB17_PATH)
nb17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nb17)


def demonstrate_illusion(X_appr, y_appr, X_rej, bands_list=(10,),
                          base_list=(1.0, 1.5, 2.0, 2.5, 3.0)):
    """
    Show the symptoms of the illusion of improvement (Illusion of Improvement, 2026) on
    Lending Club data. Key metric: default rate of the training set after RI vs the default
    rate of approved loans. If parcelling inflates the training rate above the approved
    rate, that is the symptom of the artifact; inflation that grows with base_mult shows
    that "more aggressive RI" only deepens the artifact rather than delivering a real
    improvement.
    """
    base_rate = float(y_appr.mean())
    print(f"[REFERENCE] default rate of approved loans (only observable rate): {base_rate:.4f}")
    print("[LIMITATION] true population default rate: unknown (rejects have no outcome).")
    print("             Without it, the inflation could be a correction or a distortion.\n")

    thin = nb17.baseline_ignore_rejects(X_appr, y_appr)

    print(f"{'n_bands':>8}{'base_mult':>10}{'train_rate_post_RI':>20}{'inflation_vs_approved':>22}")
    results = []
    for nb in bands_list:
        for bm in base_list:
            inferred, _ = nb17.parcelling_labels(thin, X_appr, y_appr, X_rej, nb, bm)
            n_appr, n_rej = len(y_appr), len(inferred)
            train_rate = (y_appr.sum() + inferred.sum()) / (n_appr + n_rej)
            inflacao = train_rate - base_rate
            results.append((nb, bm, train_rate, inflacao))
            print(f"{nb:>8}{bm:>10}{train_rate:>20.4f}{inflacao:>+22.4f}")

    print("\n[READING] If the training rate rises with base_mult (growing positive "
          "inflation), that is the artifact symptom from Illusion of Improvement (2026): more "
          "aggressive RI inflates the training rate, creates a more extreme boundary and "
          "appears to improve on approved loans without a real gain. Without the true "
          "population rate, this inflation cannot be read as a correction.")
    return results


if __name__ == "__main__":
    print("[Phase 2a] Demonstration of the illusion of improvement in reject inference\n")

    print("Loading approved loans (split 'train')...")
    X_appr, y_appr, issue_d_appr = nb17.load_approved_shared()
    print(f"  {len(X_appr):,} rows | bad rate {y_appr.mean()*100:.2f}%")

    print("\nLoading rejected applications (partitioned Parquet, 27,648,741 rows expected)...")
    X_rej, rej_flags, rej_counts = nb17.load_rejected_shared()
    print(f"  {len(X_rej):,} rows")

    print("\n=== Why RI cannot be validated on this data ===")
    print("  1. Weak shared signal: thin model out-of-sample AUC = 0.5620 +/- 0.0027 "
          "(4-fold CV), barely above 0.5.")
    print("  2. No outcome for rejected applications: no reject label exists in the data, "
          "so Kickout/AUK cannot be computed.")
    print("  3. No unbiased sample and no true population rate: no reference to check "
          "whether RI corrects bias or creates a new one (Illusion of Improvement, 2026).")

    print("\n=== Demonstration: training default-rate inflation vs base_mult ===")
    demonstrate_illusion(X_appr, y_appr, X_rej)

    print("\n[Phase 2a] Illusion demonstration complete.")
