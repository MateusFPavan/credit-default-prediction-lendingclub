# ============================================================================
# Phase 2b: bias-aware Bayesian evaluation (Kozodoi) with a prior sweep
# ============================================================================
# PURPOSE
# Test whether bias-aware Bayesian evaluation can measure model performance on the full
# through-the-door population of Lending Club, or whether the estimate is determined by
# the prior assumed for the rejected applications.
#
# BACKGROUND
# Selection bias (MNAR): the credit model only observes the outcome (good/bad payer) of
# APPROVED applications. Standard evaluation (AUC, profit, optimal threshold) runs on that
# subset only, and the subset is biased by construction: Lending Club had already screened
# out the riskiest applicants before the model saw them. Good performance on approved
# loans does not guarantee the same performance on the through-the-door population
# (everyone who applied, approved or not).
#
# Kozodoi (arXiv 2407.13009) measures performance on the WHOLE population without reject
# labels by treating each rejected applicant's label as a LATENT VARIABLE (unknown, with an
# assumed distribution) instead of ignoring it. In the classic Bayesian sense, it starts
# from a PRIOR -- the default rate the rejected applicants are believed to have, before any
# score is seen -- and uses it to complete the metric on the full population: an assumption
# about what cannot be observed is propagated through the calculation.
#
# METHOD
# A simplified version, not the paper's full Bayesian framework (see Scope below). For a
# policy that accepts the lowest-PD `acceptance_rate` share of the pooled population
# (approved + rejected), the expected default rate of the accepted portfolio combines
#   (a) what is known: the observed label of the accepted approved applications;
#   (b) what can only be expected: the label of the accepted rejected applications under
#       the assumed prior.
# Question answered: "if X% of the rejected applicants would default, what would the
# model's actual performance be if it decided on EVERYONE who applied, not only on the
# applications Lending Club let it see?"
# The prior is swept over 0.10-0.50 at acceptance rates of 10%, 30% and 50%. Both
# populations are scored by the thin model, the only model that puts them on the same
# probability scale.
#
# INPUTS
#   - Approved applications, 'train' split (nb17.load_approved_shared).
#   - Rejected applications, partitioned Parquet, 27,648,741 rows expected
#     (nb17.load_rejected_shared).
#   - Thin model baseline_ignore_rejects from notebooks/17_reject_thin_model.py.
#
# RESULT / INTERPRETATION
# The result depends entirely on the chosen PRIOR. Kozodoi can show a performance
# "ceiling" because the prior there is close to correct: that dataset has an unbiased
# control sample of 1,967 cases, which Lending Club does not have (Phase 2a finding). On
# Lending Club the prior is UNKNOWN. This does not invalidate the Bayesian method (it is
# mathematically correct given a prior), but the sweep quantifies the dependence:
#   - Rejected applications are about 159x the approved training rows, so they make up
#     nearly all of any accepted set and the expected default rate follows the prior with
#     a slope of about 0.99.
#   - The amplitude across the prior grid is about 0.396, identical at acceptance rates of
#     10%, 30% and 50%.
# An estimate that moves this much with the prior means the Bayesian method has not
# solved the missing-label problem; it has moved it to the choice of prior. Phase 2a showed
# empirically, not by citation, that Lending Club does not provide the true population
# default rate needed to choose that prior, so the Bayesian evaluation is also
# undetermined here: it inherits the same structural limitation that ruled out validation
# in Phase 2a (no outcome for rejected applications, no unbiased sample).
#
# KNOWN LIMITATIONS / SCOPE
#   - Implements the Bayesian extension of the expected portfolio DEFAULT RATE under an
#     acceptance policy, not the paper's full framework (formal posterior inference over
#     model parameters). This version is sufficient to test the thesis (sensitivity to the
#     prior); the full simulation is out of scope.
#   - Scores come from the thin model (AUC 0.56). A weak score leaves the choice of
#     accepted applications dominated by the prior rather than by the score.
#
# References: Kozodoi (arXiv 2407.13009, 1909.06108); Illusion of Improvement (arXiv
# 2606.18479) -- the reasoning that closed Phase 2a (no true population rate, no valid
# validation) applies here as well, now tested against the Bayesian method.

import sys
import importlib.util
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load notebooks/17_reject_thin_model.py as a module (same pattern as
# notebooks/18_profit_evaluation.py and notebooks/19_illusion_demonstration.py).
_NB17_PATH = Path(__file__).resolve().parent / "17_reject_thin_model.py"
_spec = importlib.util.spec_from_file_location("nb17_reject_thin_model", _NB17_PATH)
nb17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nb17)


def bayesian_expected_default_rate(pd_appr, y_appr, pd_rej, prior_rej_bad_rate,
                                    acceptance_rate):
    """
    Estimate the EXPECTED portfolio default rate if the lowest-risk `acceptance_rate`
    share of the WHOLE population (approved + rejected) were accepted, using:
    - the observed label of the accepted approved applications;
    - the EXPECTED label of the accepted rejected applications under the PRIOR
      (prior_rej_bad_rate).

    IMPORTANT (integration): pd_appr and pd_rej must be on the SAME probability scale for
    a joint ranking to make sense (np.argsort over the concatenation). Only the thin model
    scores both populations on the same scale (the 78-feature XGB cannot score rejected
    applications; mixing the two would repeat the scale mismatch addressed in
    notebooks/18_profit_evaluation.py). For that reason pd_appr here comes from the THIN
    MODEL, not the XGB.

    prior_rej_bad_rate: the ASSUMED default rate of the rejected applications (the prior).
    This is the swept parameter, because its true value is unknown (Phase 2a: no true
    population rate).
    """
    all_pd = np.concatenate([pd_appr, pd_rej])
    n_total = len(all_pd)
    n_accept = int(n_total * acceptance_rate)
    order = np.argsort(all_pd)  # accept lowest PD first
    accepted = order[:n_accept]

    is_appr = accepted < len(pd_appr)
    appr_defaults = y_appr[accepted[is_appr]].sum()
    n_rej_accepted = int((~is_appr).sum())
    rej_defaults_expected = n_rej_accepted * prior_rej_bad_rate

    exp_rate = (appr_defaults + rej_defaults_expected) / n_accept
    return exp_rate, n_rej_accepted


def prior_sweep(pd_appr, y_appr, pd_rej,
                 prior_grid=(0.10, 0.20, 0.30, 0.40, 0.50),
                 acceptance_rates=(0.1, 0.3, 0.5)):
    """
    Sweep the PRIOR and report how much the expected portfolio default rate changes.
    Large change -> the conclusion depends on the prior -> the Bayesian method inherits
    the Lending Club limitation.
    Small change -> the Bayesian method would be robust (unlikely given the thin model's
    AUC of 0.56: a weak score leaves the accept decision dominated by the prior, not by
    the score).
    """
    appr_base = float(y_appr.mean())
    print(f"[REF] approved default rate: {appr_base:.4f}")
    print("[NOTE] true prior for rejected applications: UNKNOWN (Phase 2a), hence the sweep.\n")
    print(f"{'accept_rate':>12}{'prior':>8}{'expected_rate':>16}{'n_rej_accepted':>16}")
    results = {}
    for ar in acceptance_rates:
        for prior in prior_grid:
            rate, n_rej = bayesian_expected_default_rate(pd_appr, y_appr, pd_rej, prior, ar)
            results[(ar, prior)] = (rate, n_rej)
            print(f"{ar:>12}{prior:>8}{rate:>16.4f}{n_rej:>16,}")
        vals = [results[(ar, p)][0] for p in prior_grid]
        print(f"    -> amplitude at this accept_rate: {max(vals) - min(vals):.4f}\n")
    print("[INTERPRETATION] Large amplitude = conclusion depends on the prior = the Bayesian")
    print("method does not solve the LC problem (it only moves it to the unknown prior).")
    return results


if __name__ == "__main__":
    print("[Phase 2b] Bayesian evaluation + prior sweep\n")

    print("Loading approved applications ('train' split)...")
    X_appr, y_appr, issue_d_appr = nb17.load_approved_shared()
    print(f"  {len(X_appr):,} rows | bad rate {y_appr.mean()*100:.2f}%")

    print("\nLoading rejected applications (partitioned Parquet, 27,648,741 rows expected)...")
    X_rej, rej_flags, rej_counts = nb17.load_rejected_shared()
    print(f"  {len(X_rej):,} rows")

    print("\nTraining thin model (baseline_ignore_rejects) -- the only model that "
          "scores both populations on the same scale...")
    thin = nb17.baseline_ignore_rejects(X_appr, y_appr)
    pd_appr = thin.predict_proba(X_appr)[:, 1]
    pd_rej = thin.predict_proba(X_rej)[:, 1]

    print("\n=== PRIOR SWEEP ===")
    results = prior_sweep(pd_appr, y_appr, pd_rej)

    print("\n[Phase 2b] done.")
