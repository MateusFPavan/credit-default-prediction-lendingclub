# ============================================================================
# Phase 2a: profit-based evaluation of reject inference (Kozodoi formula, LGD sweep)
# ============================================================================
# Purpose
#   Evaluate the approved-loan model and the reject-inference strategies by expected
#   profit, replacing the simplified calculation with an implicit LGD of 100%.
#   Per-loan expected profit (Kozodoi, arXiv 2407.13009):
#     profit = PD * A * (1+i) * (1-LGD) + (1-PD) * A * (1+i) - A
#   LGD is swept over {0.5..0.9} instead of fixed: unsecured P2P loans recover 10-30%
#   (LGD 70-90%), and 0.5 is an optimistic floor. Sweeping the assumption follows the same
#   approach as the other sweeps in this project.
#
# Steps
#   1. validate_on_approved: Kozodoi profit on approved loans by LGD, checked against the
#      realized-profit reference in src.economics.
#   2. evaluate_rejected_scenario: scenario profit on rejected applications if all were
#      accepted.
#   3. run_step3: compare reject-scoring strategies at each model's own optimal threshold
#      and at fixed acceptance rates.
#
# Inputs
#   - Approved loans: data/processed/test.parquet via src.data.load_split("test"), with
#     loan_amnt (A), int_rate (i) and target. int_rate is in src.data.EXCLUDED (not a model
#     feature), but the raw column is present. It is stored as a percentage (11.27 means
#     11.27%), confirmed by reading the parquet directly, and is divided by 100 before it
#     enters the formula.
#   - Approved-loan scores: XGB_walkforward v2.0.0 (models/xgb_final.joblib), scored with
#     the same pattern as src/run_facts.py, which produces docs/FACTS.md: build_features ->
#     prepare_X -> reindex on the booster columns -> predict_proba. src.scoring.score_frame
#     is not used because it calls clean_record() before build_features, while
#     load_split("test") is already cleaned (notebook 03 sentinels applied); run_facts.py
#     does not call clean_record at this point either.
#   - Rejected applications and thin model: notebooks/17_reject_thin_model.py, loaded as a
#     module through importlib without running its __main__ block (load_approved_shared,
#     load_rejected_shared, train_thin_model, SHARED). The treatment of dti, emp_length and
#     amount lives in notebook 17 and is not duplicated here.
#
# Method decisions
#   - Reference formula: src/economics.py computes realized profit (interest and loss
#     rebuilt from installment, term, loan_amnt and total_rec_prncp, i.e. observed
#     outcomes). That is a different paradigm from the Kozodoi expected value based on model
#     PD: it implies a 100% loss of principal with no LGD parameter, and it cannot score
#     rejects, which never have total_rec_prncp. It is used only as a sanity check in
#     Step 1, on the same pd_hat and test set.
#   - Validation: on approved loans, the realized-profit reference at threshold 0.31
#     reproduces the published docs/FACTS.md figure of $242,230,710.89 byte-exactly, and
#     the Kozodoi optimum at LGD 0.5 is threshold 0.26, matching the project's known
#     optimum. This confirms the formula and the scoring pipeline.
#   - Interest rate for rejects: rejected applications never became loans, so they have
#     no real rate i. Assumption: i = mean int_rate of approved loans (split 'train', the
#     population used by the thin model) in the equivalent thin-model score band. The band
#     count is fixed at 10, the middle value of the band-count sweep (5, 10, 20) in notebook
#     17, and is not re-swept here to keep the grid of combinations tractable.
#   - PD for rejects: rejected applications have no real label. PD is the raw thin-model
#     score (Step 2, "accept all" scenario) or the parcelling-adjusted PD (Step 3,
#     base_mult and censored_extra). The parcelling logic matches notebook 17 but is
#     vectorized here and feeds the band-adjusted probability directly into the profit
#     formula instead of drawing a 0/1 label; sampling would add noise that the
#     expected-value formula does not need.
#   - Comparison at a fixed acceptance rate, not at a common numeric threshold: applying
#     the XGB threshold 0.26 (78 features, approved loans) to the thin-model PD scale
#     (3 features, rejects) is invalid because the scales differ. Measured: at 0.26, 100%
#     of the 27.5M rejects would be accepted, which is implausible given that the rejected
#     population is worse on dti and emp_length. Following KNIME and Verbraken
#     (2014), Step 3 reports the optimal threshold on each model's own scale (an upper
#     bound per model) and, as the primary comparison, profit at a fixed acceptance rate
#     (every strategy accepts the same fraction and chooses which applications), which
#     separates decision quality from acceptance volume.
#   - censored_extra test: a uniform base_mult does not change the ranking of rejects;
#     only censored_extra, applied to rejects with dti_censored = 1 (dti >= 100%),
#     reorders who enters the accepted fraction. Measured: the profit difference
#     with_censored - without_censored is 0.00 in all 24 combinations (2 base_mult x 3 LGD x
#     4 acceptance rates). Censored rejects are already pushed into the worst-risk bands
#     by their dti value of 100, so they are never accepted and the extra multiplier never
#     changes profit. censored_extra has therefore been removed from parcelling in
#     notebook 17.
#
# Known limitations
#   - Profit on rejected applications is a scenario ("if accepted"), never a measurement;
#     every printout labels it as such.
#   - The interest rate and PD of rejected applications are assumptions (see above).
#   - The metric is expected profit with a swept LGD, not a monthly discounted cash-flow
#     NAR, which would require individual payment schedules that the data does not have.

import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_split, FEATURE_SET, CATEGORICAL_COLS
from src.features import build_features, prepare_X
from src.economics import compute_interest_loss, optimal_threshold, profit_at_threshold

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# notebooks/17_reject_thin_model.py is loaded as a module (its name starts with a digit, so
# it cannot be imported with `import`). This reuses its loaders and thin model without
# running its __main__ block (protected by the if __name__ == "__main__" guard).
_NB17_PATH = Path(__file__).resolve().parent / "17_reject_thin_model.py"
_spec = importlib.util.spec_from_file_location("nb17_reject_thin_model", _NB17_PATH)
nb17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nb17)

LGD_GRID = (0.5, 0.6, 0.7, 0.8, 0.9)
THR_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
N_BANDS = 10  # middle value of the band-count sweep in notebook 17
BASE_MULT_GRID = (1.0, 1.5, 2.0, 2.5, 3.0)  # plausible multiplier range from notebook 17
CENSORED_EXTRA_GRID = (1.0, 1.25, 1.5)


# --- per-loan profit formula (Kozodoi) ---------------------------------------
def expected_profit_per_loan(pd_hat, A, i, lgd):
    """Expected profit of a single loan. Vectorized (pd_hat, A and i may be arrays)."""
    pd_hat = np.asarray(pd_hat, dtype=float)
    A = np.asarray(A, dtype=float)
    i = np.asarray(i, dtype=float)
    return (pd_hat * A * (1 + i) * (1 - lgd)
            + (1 - pd_hat) * A * (1 + i)
            - A)


def portfolio_profit(pd_hat, A, i, lgd, accept_mask=None):
    """Total expected profit of a portfolio. accept_mask: boolean mask of accepted loans.
    If None, all loans are accepted."""
    p = expected_profit_per_loan(pd_hat, A, i, lgd)
    if accept_mask is not None:
        p = p[np.asarray(accept_mask)]
    return float(np.sum(p))


def incremental_profit(baseline_profit, ri_profit):
    """Incremental profit = profit with reject inference - profit of the approved-only
    baseline. Positive means reject inference helped."""
    return ri_profit - baseline_profit


# ============================================================================
# Step 1: validate the formula on approved loans (model PD, observed rate and label)
# ============================================================================
def validate_on_approved():
    """
    Apply the Kozodoi formula to approved loans (split 'test', XGB_walkforward v2.0.0) and
    check it against the realized-profit formula already used in the project
    (src.economics), on the same pd_hat and the same test set. An optimal threshold far
    from the known 0.31/0.26 would indicate a unit or formula error.
    """
    raw_test = load_split("test")
    test_feat = build_features(raw_test.copy())

    xgb_model = joblib.load(MODELS_DIR / "xgb_final.joblib")
    trained_cols = list(xgb_model.get_booster().feature_names)
    X = prepare_X(test_feat, FEATURE_SET, CATEGORICAL_COLS).reindex(columns=trained_cols, fill_value=0)
    pd_hat = xgb_model.predict_proba(X)[:, 1]

    A = raw_test["loan_amnt"].to_numpy(dtype=float)
    i = raw_test["int_rate"].to_numpy(dtype=float) / 100.0  # percent -> fraction
    y = raw_test["target"].to_numpy()

    interest, loss = compute_interest_loss(raw_test)
    interest = interest.to_numpy()
    loss = loss.to_numpy()
    thr_realized, profit_realized = optimal_threshold(y, pd_hat, interest, loss)
    profit_at_031 = profit_at_threshold(y, pd_hat, 0.31, interest, loss)

    print("=== Step 1: profit on approved loans (Kozodoi expected value) by LGD ===")
    print(f"  [reference: realized-profit formula, src.economics, same pd_hat and test set]")
    print(f"    optimal threshold={thr_realized:.2f} | profit={profit_realized:,.2f}")
    print(f"    profit @0.31 (operating threshold)={profit_at_031:,.2f} "
          f"(docs/FACTS.md xgb_profit: 242,230,710.89)")
    print(f"\n{'LGD':>5}{'kozodoi_total_profit':>22}{'optimal_threshold':>18}")
    results = {}
    for lgd in LGD_GRID:
        profits = np.array([
            portfolio_profit(pd_hat, A, i, lgd, accept_mask=(pd_hat < t)) for t in THR_GRID
        ])
        best_idx = int(np.argmax(profits))
        best_thr, best_profit = float(THR_GRID[best_idx]), float(profits[best_idx])
        results[lgd] = (best_thr, best_profit)
        print(f"{lgd:>5}{best_profit:>22,.0f}{best_thr:>18.3f}")
    return pd_hat, A, i, y, results


# ============================================================================
# Step 2: scenario for rejected applications (rate and label are assumptions)
# ============================================================================
def evaluate_rejected_scenario():
    """
    Scenario profit on rejected applications if all of them were accepted (thresholds are
    applied in Step 3). PD = raw thin-model score; i = assumption (mean rate of the
    equivalent score band among approved loans). This is a scenario, not a measurement:
    rejected applications never became loans and have neither a real label nor a real
    rate.
    """
    print("\n[Loading thin model and treated rejected applications...]")
    X_appr, y_appr, issue_d_appr = nb17.load_approved_shared()
    X_rej, rej_flags, rej_counts = nb17.load_rejected_shared()
    thin = nb17.train_thin_model(X_appr, y_appr)

    p_appr = thin.predict_proba(X_appr)[:, 1]
    p_rej = thin.predict_proba(X_rej)[:, 1]
    A_rej = X_rej["amount"].to_numpy(dtype=float)

    # rate assumption: mean int_rate of approved loans (split 'train') in the equivalent band
    train_raw = load_split("train")
    int_rate_train = train_raw["int_rate"].to_numpy(dtype=float) / 100.0

    edges = np.quantile(p_appr, np.linspace(0, 1, N_BANDS + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    appr_band = np.digitize(p_appr, edges[1:-1])
    rej_band = np.digitize(p_rej, edges[1:-1])

    band_rate = {}
    band_bad = {}
    for b in range(N_BANDS):
        m = appr_band == b
        band_rate[b] = float(int_rate_train[m].mean()) if m.any() else float(int_rate_train.mean())
        band_bad[b] = float(y_appr[m].mean()) if m.any() else float(y_appr.mean())
    i_rej = np.array([band_rate[b] for b in rej_band])

    print("\n=== Step 2: scenario profit on rejects by LGD (accept all; assumed rate) ===")
    print(f"{'LGD':>5}{'scenario_profit':>20}")
    scenario_results = {}
    for lgd in LGD_GRID:
        profit = portfolio_profit(p_rej, A_rej, i_rej, lgd)  # accept all = scenario
        scenario_results[lgd] = profit
        print(f"{lgd:>5}{profit:>20,.0f}")
    print("[NOTE] Profit on rejects is a scenario: the rate (equivalent score band of approved "
          "loans, split 'train') and the PD (thin model, 3 features) are assumptions, not measured.")

    return {
        "thin": thin, "X_appr": X_appr, "y_appr": y_appr, "X_rej": X_rej,
        "rej_flags": rej_flags, "p_appr": p_appr, "p_rej": p_rej, "edges": edges,
        "appr_band": appr_band, "rej_band": rej_band, "band_rate": band_rate,
        "band_bad": band_bad, "i_rej": i_rej, "A_rej": A_rej,
        "scenario_results": scenario_results,
    }


# ============================================================================
# Step 3: compare reject-scoring strategies at a fixed acceptance rate
# ============================================================================
# A common numeric threshold is invalid across models with different PD scales: the XGB
# threshold 0.26 (78 features) applied to the thin model (3 features) accepts 100% of the
# rejects, which is implausible because the rejected population is worse on dti and
# emp_length. Two comparisons from the literature (KNIME; Verbraken 2014): the optimal
# threshold on each model's own scale, as an upper bound per model, and profit at a fixed
# acceptance rate as the primary comparison (same fraction accepted, each model chooses
# which applications), which separates decision quality from acceptance volume.
# ============================================================================
def parcelling_pd_rejected(rej_band, band_bad, rej_flags, base_mult, censored_extra):
    """Band-adjusted PD using the parcelling logic of notebook 17 (nb17.parcelling_labels)
    plus the censored_extra multiplier, returning the continuous probability
    (band_bad * mult) instead of a sampled label: the expected-value formula uses
    probabilities, and sampling would add unnecessary noise. Fully vectorized (no
    row-by-row Python loop, unlike the parcelling in notebook 17)."""
    band_bad_arr = np.array([band_bad[b] for b in range(len(band_bad))])
    p_bad = band_bad_arr[rej_band] * base_mult
    censored = rej_flags["dti_censored"].to_numpy() == 1
    p_bad = np.where(censored, p_bad * censored_extra, p_bad)
    return np.clip(p_bad, 0.0, 1.0)


def optimal_threshold_ownscale(pd_hat, A, i, lgd, grid=None):
    """Upper bound per model: profit-maximizing threshold on the model's own PD scale.
    Named differently from src.economics.optimal_threshold, which uses a different
    formula and paradigm (realized profit vs expected value) and must not be confused
    with it."""
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    best_t, best_p = None, -np.inf
    for t in grid:
        mask = pd_hat < t
        p = portfolio_profit(pd_hat, A, i, lgd, accept_mask=mask) if mask.any() else 0.0
        if p > best_p:
            best_p, best_t = p, float(t)
    return best_t, best_p


def profit_at_acceptance_rate(pd_hat, A, i, lgd, rate, order=None):
    """Primary comparison: accept the fraction `rate` with the lowest PD (best risks) on
    the model's own scale. Fair across strategies: the same fraction is accepted and each
    strategy chooses which applications, with no cross-model numeric threshold. `order`
    (argsort of pd_hat) can be precomputed and reused across LGD and rate values, since it
    depends only on pd_hat; this avoids repeating an argsort of 27.5M elements 60 times."""
    if order is None:
        order = np.argsort(pd_hat)
    n_accept = int(len(pd_hat) * rate)
    idx = order[:n_accept]
    return float(np.sum(expected_profit_per_loan(pd_hat[idx], A[idx], i[idx], lgd)))


RATE_GRID = (0.1, 0.2, 0.3, 0.5)
LGD_GRID_STEP3 = (0.5, 0.7, 0.9)  # representative subset (all 5 LGD values would be too dense)


def run_step3(step2_out):
    """Upper bound per model (own-scale thresholds) and primary comparison (fixed
    acceptance rate) across 5 reject-scoring strategies: the raw thin model, and
    parcelling with base_mult in {1.5, 2.0} (plausible range from notebook 17) crossed
    with censored_extra in {1.0 = no boost, 1.5 = maximum boost}. This isolates the effect
    of censored_extra: a uniform base_mult (same factor in every band) does not change the
    ranking of rejects, because multiplying everything by the same constant preserves
    order; only censored_extra, applied selectively to censored rejects, reorders who
    enters the accepted X%. This is why the fixed-rate comparison is the right test of
    whether censored_extra adds profit."""
    p_rej = step2_out["p_rej"]
    rej_band = step2_out["rej_band"]
    band_bad = step2_out["band_bad"]
    rej_flags = step2_out["rej_flags"]
    A_rej = step2_out["A_rej"]
    i_rej = step2_out["i_rej"]

    strategies = {"raw_thin (no parcelling)": p_rej}
    for base_mult in (1.5, 2.0):
        for cx, label in ((1.0, "without_censored"), (1.5, "with_censored")):
            name = f"base{base_mult}_{label}"
            strategies[name] = parcelling_pd_rejected(rej_band, band_bad, rej_flags, base_mult, cx)

    print("\n=== Step 3 (upper bound): optimal threshold on each strategy's own scale ===")
    print(f"{'strategy':>28}{'LGD':>6}{'own_threshold':>20}{'profit_at_optimum':>20}")
    for name, pd_hat in strategies.items():
        for lgd in LGD_GRID_STEP3:
            t, p = optimal_threshold_ownscale(pd_hat, A_rej, i_rej, lgd)
            print(f"{name:>28}{lgd:>6}{t:>20.3f}{p:>20,.0f}")
    print("  [reference] XGB optimal threshold on approved loans (Step 1, LGD=0.5) = 0.260; "
          "the values above should be well below it, consistent with a more compressed "
          "scale (thin model with 3 weak features vs 78 in the XGB).")

    print("\n=== Step 3 (primary comparison): profit at a fixed acceptance rate ===")
    names = list(strategies.keys())
    orders = {n: np.argsort(strategies[n]) for n in names}  # one argsort per strategy, reused
    header = f"{'LGD':>4}{'rate':>6}  " + "  ".join(f"{n:>24}" for n in names)
    print(header)
    rows = []
    for lgd in LGD_GRID_STEP3:
        for rate in RATE_GRID:
            vals = [profit_at_acceptance_rate(strategies[n], A_rej, i_rej, lgd, rate, order=orders[n])
                    for n in names]
            rows.append((lgd, rate, *vals))
            print(f"{lgd:>4}{rate:>6}  " + "  ".join(f"{v:>24,.0f}" for v in vals))

    df = pd.DataFrame(rows, columns=["lgd", "rate"] + names)

    print("\n[censored_extra test] Within each (LGD, rate), compare "
          "base{X}_with_censored against base{X}_without_censored (same base_mult, only cx changes). "
          "If with_censored is consistently higher than without_censored, the flag adds profit and "
          "is kept. If equal or lower, the flag is removed from parcelling.")
    for base_mult in (1.5, 2.0):
        without_censored = df[f"base{base_mult}_without_censored"]
        with_censored = df[f"base{base_mult}_with_censored"]
        delta = with_censored - without_censored
        print(f"  base_mult={base_mult}: with_censored - without_censored "
              f"ranges from {delta.min():,.0f} to {delta.max():,.0f} "
              f"({(delta > 0).sum()}/{len(delta)} combinations improved)")
    return df


if __name__ == "__main__":
    print("[Phase 2a] Profit-based evaluation (Kozodoi formula, LGD sweep)\n")

    pd_hat_appr, A_appr, i_appr, y_appr_test, step1_results = validate_on_approved()
    step2_out = evaluate_rejected_scenario()
    step3_df = run_step3(step2_out)

    print("\n[Phase 2a] Profit-based evaluation complete.")
