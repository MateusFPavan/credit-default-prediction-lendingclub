"""
Contract tests for the scoring API.

Uses fastapi.testclient.TestClient: exercises the API in-process, without starting
uvicorn (httpx underneath, already in requirements). Covers: /health, a coherent valid
score, out-of-range input -> 422, missing field -> 422, and term=60 rejected.
"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

# valid example, with real categories read from the generated contract
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
    # decision agrees with probability and threshold
    assert (body["decision"] == "reject") == (
        body["probability_default"] >= body["threshold"])


def test_out_of_range_fico_returns_422():
    bad = dict(VALID, fico_range_low=9000)
    assert client.post("/score", json=bad).status_code == 422


def test_missing_required_field_returns_422():
    missing = {k: v for k, v in VALID.items() if k != "annual_inc"}
    assert client.post("/score", json=missing).status_code == 422


def test_term_60_rejected():
    """term=60 must be rejected: the model is not transferable to it (Model Card §4)."""
    bad = dict(VALID, term=60)
    assert client.post("/score", json=bad).status_code == 422


def test_impossible_date_returns_422_or_handles_cleanly():
    """An impossible date must not turn into a silent 500."""
    bad = dict(VALID, issue_d="2015-13-45")
    r = client.post("/score", json=bad)
    assert r.status_code != 500, r.text
