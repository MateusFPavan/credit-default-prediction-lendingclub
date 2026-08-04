"""Population Stability Index (PSI) engine for credit-default-prediction-lendingclub.

Core rule: bin edges are fit ONCE on a baseline series and frozen into a Binning object.
Every subsequent comparison (a later split, a later quarter) reuses that same Binning -
it is never refit per period. Refitting bins per period would make PSI values across
periods incomparable (each period would be measured against its own distribution
instead of against a fixed reference), which defeats the purpose of a drift metric.

Numeric columns: bins are decile edges (n_bins=10) fit on the baseline's non-sentinel
("core") values. Point masses at known sentinel values (-1, 999 - see docs/FACTS.md
section 4 and docs/cleaning_decisions.md for the rollout-2012 / MNAR mechanisms that
produce them) are carved out into their own dedicated bin before the quantile edges are
computed, so a sentinel doesn't get silently blended into a real-value decile. Decile
edges are also deduplicated (low-cardinality numerics, e.g. 0/1 flags or small integer
counts, would otherwise produce repeated quantile edges).

Categorical columns: one bin per baseline category. Any category seen in a comparison
period but absent from the baseline gets its own dedicated "__unseen__" bin, rather than
being folded into an existing category - a brand-new category is itself a signal worth
seeing in the PSI, not noise to be hidden.

Empty bins (zero mass in baseline or comparison) are floored at EPSILON before the log
ratio, per the standard PSI formula, since ln(0) is undefined and a truly empty bin
should contribute a bounded (not infinite) amount of instability.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

SENTINEL_VALUES = (-1, 999)
SENTINEL_MIN_SHARE = 0.01
N_BINS = 10
EPSILON = 1e-6

STABLE_MAX = 0.10
ATTENTION_MAX = 0.25


@dataclass
class Binning:
    """Frozen binning definition for one column, fit once on a baseline series."""
    name: str
    kind: str  # "numeric" or "categorical"
    sentinel_values: list
    edges: np.ndarray  # numeric only; None for categorical
    categories: list  # categorical only; None for numeric


def psi_band(psi_value):
    if psi_value < STABLE_MAX:
        return "stable"
    if psi_value < ATTENTION_MAX:
        return "attention"
    return "unstable"


def fit_numeric_binning(series, name=None, n_bins=N_BINS,
                         sentinel_values=SENTINEL_VALUES, sentinel_min_share=SENTINEL_MIN_SHARE):
    s = series.dropna()
    n = len(s)

    present_sentinels = []
    for sv in sentinel_values:
        share = (s == sv).mean() if n else 0.0
        if share >= sentinel_min_share:
            present_sentinels.append(sv)

    core = s[~s.isin(present_sentinels)] if present_sentinels else s

    if len(core) == 0:
        edges = np.array([-np.inf, np.inf])
    else:
        qs = np.linspace(0, 1, n_bins + 1)
        raw_edges = np.unique(core.quantile(qs).to_numpy())
        if len(raw_edges) < 2:
            edges = np.array([-np.inf, np.inf])
        else:
            edges = raw_edges.copy()
            edges[0] = -np.inf
            edges[-1] = np.inf

    return Binning(name=name, kind="numeric", sentinel_values=present_sentinels,
                    edges=edges, categories=None)


def fit_categorical_binning(series, name=None):
    categories = sorted(series.dropna().astype(str).unique().tolist())
    return Binning(name=name, kind="categorical", sentinel_values=None,
                    edges=None, categories=categories)


def fit_binning(series, name=None, categorical=False, **kwargs):
    if categorical:
        return fit_categorical_binning(series, name=name)
    return fit_numeric_binning(series, name=name, **kwargs)


def assign_bins(series, binning):
    """Return a Series of bin-label strings for `series`, using a frozen Binning."""
    if binning.kind == "categorical":
        cats = set(binning.categories)
        return series.astype(str).apply(lambda v: v if v in cats else "__unseen__")

    labels = pd.Series(index=series.index, dtype=object)
    is_sentinel = pd.Series(False, index=series.index)
    for sv in binning.sentinel_values:
        mask = series == sv
        labels[mask] = f"sentinel={sv}"
        is_sentinel = is_sentinel | mask

    remaining = series[~is_sentinel]
    if len(remaining) > 0:
        if len(binning.edges) >= 2:
            cut = pd.cut(remaining, bins=binning.edges, include_lowest=True, duplicates="drop")
            labels.loc[remaining.index] = cut.astype(str)
        else:
            labels.loc[remaining.index] = "all"

    labels[series.isna()] = "__missing__"
    return labels


def psi_from_labels(baseline_labels, comparison_labels, eps=EPSILON):
    all_labels = sorted(set(baseline_labels.unique()) | set(comparison_labels.unique()), key=str)

    n_base = len(baseline_labels)
    n_comp = len(comparison_labels)

    base_pct = baseline_labels.value_counts().reindex(all_labels, fill_value=0) / n_base
    comp_pct = comparison_labels.value_counts().reindex(all_labels, fill_value=0) / n_comp

    base_pct = base_pct.clip(lower=eps)
    comp_pct = comp_pct.clip(lower=eps)

    psi = float(((comp_pct - base_pct) * np.log(comp_pct / base_pct)).sum())
    return psi


def compute_psi(baseline_series, comparison_series, binning=None, categorical=False, **fit_kwargs):
    """Compute PSI for one column. Fits a Binning on baseline_series if none is passed."""
    if binning is None:
        binning = fit_binning(baseline_series, categorical=categorical, **fit_kwargs)
    base_labels = assign_bins(baseline_series, binning)
    comp_labels = assign_bins(comparison_series, binning)
    psi = psi_from_labels(base_labels, comp_labels)
    return psi, binning


def fit_binnings(baseline_df, feature_cols, categorical_cols=()):
    """Fit one Binning per column in feature_cols, from baseline_df, once."""
    binnings = {}
    for col in feature_cols:
        is_cat = col in categorical_cols
        binnings[col] = fit_binning(baseline_df[col], name=col, categorical=is_cat)
    return binnings


def psi_table(binnings, baseline_df, comparison_df, feature_cols):
    """PSI of comparison_df against baseline_df, using pre-fit `binnings` (never refit)."""
    rows = []
    for col in feature_cols:
        binning = binnings[col]
        base_labels = assign_bins(baseline_df[col], binning)
        comp_labels = assign_bins(comparison_df[col], binning)
        psi = psi_from_labels(base_labels, comp_labels)
        rows.append({
            "feature": col,
            "kind": binning.kind,
            "psi": psi,
            "band": psi_band(psi),
            "n_bins": len(binning.categories) if binning.kind == "categorical" else max(len(binning.edges) - 1, 1),
            "n_baseline": len(baseline_df[col].dropna()),
            "n_comparison": len(comparison_df[col].dropna()),
        })
    return pd.DataFrame(rows)
