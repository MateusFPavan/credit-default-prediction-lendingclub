"""
Testes unitarios do motor de PSI (src/psi.py) -- usado no monitor de drift de
producao (src/monitor.py) e nos relatorios trimestrais (src/run_psi.py).
Cobertura C1.2, auditoria Boletim 12/13.

Cobre: bandas, PSI ~0 entre distribuicoes identicas, PSI alto numa mudanca real,
carve-out de sentinela, categoria nunca vista, e coluna de baixa cardinalidade
(quantis duplicados) -- os quatro comportamentos documentados no docstring do
modulo, nao so o caminho feliz.
"""
import numpy as np
import pandas as pd
import pytest

from src.psi import (
    psi_band, fit_numeric_binning, fit_categorical_binning, assign_bins,
    compute_psi, STABLE_MAX, ATTENTION_MAX,
)


def test_psi_band_boundaries():
    assert psi_band(0.0) == "stable"
    assert psi_band(STABLE_MAX - 0.001) == "stable"
    assert psi_band(STABLE_MAX) == "attention"
    assert psi_band(ATTENTION_MAX - 0.001) == "attention"
    assert psi_band(ATTENTION_MAX) == "unstable"
    assert psi_band(1.0) == "unstable"


def test_compute_psi_identical_distribution_is_near_zero():
    rng = np.random.default_rng(42)
    baseline = pd.Series(rng.normal(size=5000))
    comparison = baseline.copy()
    psi, _ = compute_psi(baseline, comparison)
    assert psi < 0.01


def test_compute_psi_shifted_distribution_is_unstable():
    rng = np.random.default_rng(42)
    baseline = pd.Series(rng.normal(loc=0.0, size=5000))
    comparison = pd.Series(rng.normal(loc=3.0, size=5000))
    psi, _ = compute_psi(baseline, comparison)
    assert psi_band(psi) == "unstable"


def test_sentinel_carved_into_its_own_bin():
    """Sentinela (999) no piso minimo de share nao pode se misturar nos bins de quantil."""
    core = pd.Series(np.linspace(0, 100, 990))
    sentinel = pd.Series([999] * 10)  # 10/1000 = 1% = exatamente SENTINEL_MIN_SHARE
    baseline = pd.concat([core, sentinel], ignore_index=True)

    binning = fit_numeric_binning(baseline, name="x")
    assert 999 in binning.sentinel_values

    labels = assign_bins(baseline, binning)
    assert (labels == "sentinel=999").sum() == 10


def test_unseen_category_gets_its_own_bin():
    baseline = pd.Series(["a", "b", "a", "c"])
    comparison = pd.Series(["a", "d", "d"])  # "d" nunca apareceu na baseline

    binning = fit_categorical_binning(baseline)
    labels = assign_bins(comparison, binning)
    assert (labels == "__unseen__").sum() == 2


def test_low_cardinality_numeric_does_not_crash_on_duplicate_quantiles():
    """Coluna 0/1 (flag): quantis colapsam, fit_numeric_binning deve deduplicar sem erro."""
    baseline = pd.Series([0] * 800 + [1] * 200)
    binning = fit_numeric_binning(baseline, name="flag")
    assert len(binning.edges) >= 2
    assert len(set(binning.edges)) == len(binning.edges)

    psi, _ = compute_psi(baseline, baseline, binning=binning)
    assert psi == pytest.approx(0.0, abs=1e-9)
