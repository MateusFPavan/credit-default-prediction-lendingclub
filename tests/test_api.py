"""
Testes de contrato da API de scoring (tarefa 4.3).

Usa fastapi.testclient.TestClient: exercita a API em processo, sem subir uvicorn
(httpx por baixo, ja em requirements). Cobre: /health, score valido coerente,
input fora de faixa -> 422, campo faltante -> 422, e term=60 recusado.
"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

# exemplo valido, com categorias reais lidas do contrato gerado
_contract = json.loads((Path("src/_api_contract.json")).read_text())
VALID = {
    "loan_amnt": 10000, "installment": 325.5, "term": 36,
    "annual_inc": 60000, "fico_range_low": 710, "dti": 15.2,
    "earliest_cr_line": "2001-08-01", "issue_d": "2015-06-01",
    "home_ownership": _contract["categories"]["home_ownership"][0],
    "purpose": _contract["categories"]["purpose"][0],
    "acc_open_past_24mths": 3, "open_acc": 8, "total_acc": 20,
    "revol_bal": 8500, "revol_util": 42.3,
    "initial_list_status": 'f',
    "inq_last_6mths": 1,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_score_valid_is_coherent():
    r = client.post("/score", json=VALID)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["probability_default"] <= 1.0
    assert body["decision"] in ("approve", "reject")
    # decisao concorda com prob e threshold
    assert (body["decision"] == "reject") == (
        body["probability_default"] >= body["threshold"])


def test_out_of_range_fico_returns_422():
    bad = dict(VALID, fico_range_low=9000)
    assert client.post("/score", json=bad).status_code == 422


def test_missing_required_field_returns_422():
    missing = {k: v for k, v in VALID.items() if k != "annual_inc"}
    assert client.post("/score", json=missing).status_code == 422


def test_term_60_rejected():
    """term=60 deve ser recusado: modelo nao-transferivel (Model Card §4)."""
    bad = dict(VALID, term=60)
    assert client.post("/score", json=bad).status_code == 422


def test_impossible_date_returns_422_or_handles_cleanly():
    """data impossivel nao pode virar 500 silencioso."""
    bad = dict(VALID, issue_d="2015-13-45")
    r = client.post("/score", json=bad)
    assert r.status_code != 500, r.text
