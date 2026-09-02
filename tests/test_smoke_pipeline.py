"""
Smoke test rapido da pipeline inteira (P-010 / ML Test Score Infra 3).

O que faltava, e por que os testes existentes nao cobriam: run_all.py e
verify_pipeline SAO teste de integracao ponta a ponta, mas sao manuais, ficam fora
do CI e exigem os parquets (gitignored). Ou seja: a integracao so era exercitada
quando alguem lembrava, numa maquina que tivesse o dado.

Este arquivo roda em SEGUNDOS, sem parquet, dentro do CI, e cobre as duas coisas
que quebram calado numa refatoracao:

  1. o GRAFO DE IMPORTS de src/ -- um import circular ou um simbolo renomeado
     derruba a pipeline inteira e nenhum teste de unidade percebe, porque cada um
     importa so o proprio modulo;
  2. o CAMINHO DE FEATURE DE TREINO ponta a ponta, com um fit de verdade sobre dado
     sintetico -- clean_record -> build_features -> prepare_X -> fit -> predict.
     Nao verifica NUMERO (isso e o verify_pipeline com o dado real); verifica que as
     interfaces entre os quatro passos continuam se encaixando.

A divisao de trabalho e deliberada: este teste responde "a pipeline monta?" em
segundos e em toda push; o verify_pipeline responde "a pipeline reproduz o numero?"
com o dado real, quando o Pavan roda.
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
    """Import circular ou simbolo renomeado derruba a pipeline e nenhum teste de
    unidade percebe -- cada um importa so o proprio modulo."""
    importlib.import_module(mod)


def _amostra(n=200, seed=0):
    """Dado sintetico com as colunas cruas que a pipeline exige. Os valores nao
    precisam ser realistas: o teste e de encaixe de interface, nao de numero."""
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
    """clean_record -> build_features -> prepare_X -> fit -> predict, com fit de
    verdade. Nao afere numero; afere que os quatro passos continuam se encaixando.

    Usa um XGBClassifier pequeno de proposito, e NAO o build_xgb_final: o objetivo e
    velocidade e independencia da configuracao congelada, nao reproduzir resultado."""
    from xgboost import XGBClassifier

    df = _amostra()
    y = (np.arange(len(df)) % 5 == 0).astype(int)   # alvo sintetico, 20% de positivos

    X = prepare_X(build_features(clean_record(df)), FEATURE_SET, CATEGORICAL_COLS)
    assert len(X) == len(df)
    assert X.select_dtypes(include=["object"]).empty, "sobrou coluna nao numerica"

    m = XGBClassifier(n_estimators=8, max_depth=3, random_state=42, n_jobs=1)
    m.fit(X, y)
    p = m.predict_proba(X)[:, 1]
    assert p.shape == (len(df),)
    assert np.isfinite(p).all() and (0 <= p).all() and (p <= 1).all()


def test_caminho_de_serving_monta_ponta_a_ponta():
    """A outra metade: do registro cru ate a decisao, pelo artefato de verdade."""
    out = score_frame(_amostra(n=20))
    assert list(out.columns) == ["probability_default", "decision"]
    assert len(out) == 20
    assert np.isfinite(out["probability_default"]).all()
    assert set(out["decision"]) <= {"approve", "reject"}


def test_o_smoke_test_e_rapido_de_verdade():
    """Um smoke test lento deixa de ser rodado, e ai nao e smoke test.

    O limite e generoso (10s) porque maquina de CI varia; o ponto e travar uma ordem
    de grandeza, nao cronometrar."""
    t0 = time.perf_counter()
    score_frame(_amostra(n=50))
    assert time.perf_counter() - t0 < 10.0
