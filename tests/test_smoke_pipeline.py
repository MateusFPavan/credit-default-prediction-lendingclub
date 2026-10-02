"""
Fast smoke test of the whole pipeline (ML Test Score, Infra 3).

What was missing, and why the existing tests did not cover it: run_all.py and
verify_pipeline ARE end-to-end integration tests, but they are manual, sit outside
CI and need the parquets (gitignored). So integration was only exercised when
someone remembered, on a machine that had the data.

This file runs in SECONDS, without parquet, inside CI, and covers the two things
that break silently in a refactor:

  1. the IMPORT GRAPH of src/ -- a circular import or a renamed symbol takes down
     the whole pipeline and no unit test notices, because each one imports only
     its own module;
  2. the TRAINING FEATURE PATH end to end, with a real fit on synthetic data --
     clean_record -> build_features -> prepare_X -> fit -> predict.
     It does not check NUMBERS (that is verify_pipeline with the real data); it checks
     that the interfaces between the four steps still fit together.

The split of work is deliberate: this test answers "does the pipeline assemble?" in
seconds and on every push; verify_pipeline answers "does the pipeline reproduce the
number?" with the real data, when run manually.
"""
import importlib
import time

import numpy as np
import pandas as pd
import pytest

from src.cleaning import clean_record
from src.data import FEATURE_SET, CATEGORICAL_COLS
from src.features import build_features, prepare_X
from src.scoring import score_frame

MODULOS = ["src.data", "src.cleaning", "src.features", "src.scoring", "src.models",
           "src.economics", "src.psi", "src.api", "src.verify_pipeline", "src.run_facts"]

BASE = {
    "loan_amnt": 10000.0, "installment": 325.5, "term": 36,
    "annual_inc": 60000.0, "fico_range_low": 710.0, "dti": 15.2,
    "earliest_cr_line": "2001-08-01", "issue_d": "2015-06-01",
    "home_ownership": "rent", "purpose": "debt_consolidation",
    "verification_status": "verified", "initial_list_status": "w",
    "acc_open_past_24mths": 3.0, "open_acc": 8.0, "total_acc": 20.0,
    "revol_bal": 8500.0, "revol_util": 42.3, "inq_last_6mths": 1.0,
}


@pytest.mark.parametrize("mod", MODULOS)
def test_todo_modulo_de_src_importa(mod):
    """A circular import or a renamed symbol takes down the pipeline and no unit test
    notices -- each one imports only its own module."""
    importlib.import_module(mod)


def _amostra(n=200, seed=0):
    """Synthetic data with the raw columns the pipeline requires. The values need not
    be realistic: the test is about interfaces fitting together, not about numbers."""
    rng = np.random.RandomState(seed)
    linhas = []
    for i in range(n):
        r = dict(BASE)
        r["annual_inc"] = float(rng.randint(20_000, 200_000))
        r["loan_amnt"] = float(rng.randint(1_000, 35_000))
        r["installment"] = r["loan_amnt"] / 36 * 1.15
        r["fico_range_low"] = float(rng.randint(660, 830))
        r["dti"] = float(rng.uniform(0, 40))
        r["open_acc"] = float(rng.randint(1, 30))
        r["total_acc"] = float(rng.randint(5, 60))
        r["revol_bal"] = float(rng.randint(0, 50_000))
        r["home_ownership"] = ["rent", "own", "mortgage"][i % 3]
        r["purpose"] = ["debt_consolidation", "credit_card", "other"][i % 3]
        r["verification_status"] = ["verified", "not verified"][i % 2]
        r["initial_list_status"] = ["w", "f"][i % 2]
        linhas.append(r)
    df = pd.DataFrame(linhas)
    df["issue_d"] = pd.to_datetime(df["issue_d"])
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"])
    return df


def test_caminho_de_treino_monta_ponta_a_ponta():
    """clean_record -> build_features -> prepare_X -> fit -> predict, with a real
    fit. Does not measure numbers; checks that the four steps still fit together.

    Uses a small XGBClassifier on purpose, NOT build_xgb_final: the goal is speed and
    independence from the frozen configuration, not reproducing a result."""
    from xgboost import XGBClassifier

    df = _amostra()
    y = (np.arange(len(df)) % 5 == 0).astype(int)   # synthetic target, 20% positives

    X = prepare_X(build_features(clean_record(df)), FEATURE_SET, CATEGORICAL_COLS)
    assert len(X) == len(df)
    assert X.select_dtypes(include=["object"]).empty, "sobrou coluna nao numerica"

    m = XGBClassifier(n_estimators=8, max_depth=3, random_state=42, n_jobs=1)
    m.fit(X, y)
    p = m.predict_proba(X)[:, 1]
    assert p.shape == (len(df),)
    assert np.isfinite(p).all() and (0 <= p).all() and (p <= 1).all()


def test_caminho_de_serving_monta_ponta_a_ponta():
    """The other half: from the raw record to the decision, through the real artifact."""
    out = score_frame(_amostra(n=20))
    assert list(out.columns) == ["probability_default", "decision"]
    assert len(out) == 20
    assert np.isfinite(out["probability_default"]).all()
    assert set(out["decision"]) <= {"approve", "reject"}


def test_o_smoke_test_e_rapido_de_verdade():
    """A slow smoke test stops being run, and then it is not a smoke test.

    The limit is generous (10s) because CI machines vary; the point is to pin an order
    of magnitude, not to benchmark."""
    t0 = time.perf_counter()
    score_frame(_amostra(n=50))
    assert time.perf_counter() - t0 < 10.0
