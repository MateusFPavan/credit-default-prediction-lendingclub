"""
Inference API (FastAPI) for the Lending Club default model.

Task 4.3 of Project 4, Phase 2. Thin shell over src.scoring (pure logic).
- Input: raw origination fields. Required = the high-SHAP ones (Model Card
  §8); the rest of the bureau block is optional and, if omitted, falls into the SAME
  sentinel/missingness mechanism as training (FACTS §4) -- not a shortcut, it's the
  project's design. Omitting many fields moves the profile closer to the sentinel regime and
  should trigger a PSI check (see the drift monitor, task 4.4).
- term accepts ONLY 36. The model was trained only on 36 months and is declaredly
  not transferable to 60 (scope §9, Model Card §4, README). Accepting 60 would mean
  scoring something the model doesn't know how to score -- the API refuses with a message that
  points to the limitation, enforcing it rather than just describing it.
- Output: typed. probability_default in [0,1] + decision in {approve, reject}
  at the 0.31 operational cutoff.
- Errors: missing/out-of-range input -> Pydantic's own 422 (not a silent 500).

Inference is deterministic and cheap: loads the artifact and scores, never retrains.
The reproducibility gotchas (seed, n_jobs, row order) belong to TRAINING.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from src.scoring import score_frame, load_model, OPERATIONAL_THRESHOLD

# --- contract discovered at runtime (Block 2a) ---
_CONTRACT = json.loads((Path(__file__).parent / "_api_contract.json").read_text())
_HOME = _CONTRACT["categories"].get("home_ownership", ["mortgage", "rent", "own", "other"])
_PURP = _CONTRACT["categories"].get("purpose", ["debt_consolidation", "credit_card", "other"])
_ILS = _CONTRACT["categories"].get("initial_list_status", ['f', 'w'])


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()  # loads the .joblib once, at startup (not on the 1st request)
    yield


app = FastAPI(
    title="Lending Club — Credit Default Scoring API",
    version="1.0.0",
    description=(
        "Second decision layer over already-approved loans. Estimates "
        "P(default | approved) and decides approve/reject at the profit cutoff (0.31). "
        "Does NOT score rejected applicants (selection bias, see Model Card §9). "
        "Only accepts term=36 (model not transferable to 60 months). "
        "Score is optimistic (underestimates default); do not use as a lower bound."
    ),
    lifespan=lifespan,
)


class ScoreRequest(BaseModel):
    """Raw fields of a credit application. Required = high-SHAP; the rest optional."""
    loan_amnt: float = Field(..., gt=0, le=100_000, description="Requested amount (USD)")
    installment: float = Field(..., gt=0, le=2_000, description="Monthly installment (USD)")
    term: Literal[36] = Field(..., description="Term in months. Only 36 (model does not serve 60, see Model Card §4)")
    annual_inc: float = Field(..., ge=0, le=15_000_000, description="Self-reported annual income (USD)")
    fico_range_low: float = Field(..., ge=300, le=850, description="Lower FICO bound at origination")
    dti: float = Field(..., ge=0, le=100, description="Debt-to-income (%). >100 is impossible (scope §2)")
    earliest_cr_line: str = Field(..., description="Date of the oldest credit line (YYYY-MM-DD)")
    issue_d: str = Field(..., description="Loan issue month (YYYY-MM-DD)")
    home_ownership: str = Field(..., description=f"One of: {_HOME}")
    purpose: str = Field(..., description=f"One of: {_PURP}")
    acc_open_past_24mths: float = Field(..., ge=0, le=100)
    open_acc: float = Field(..., ge=0, le=200)
    total_acc: float = Field(..., ge=0, le=300)
    revol_bal: float = Field(..., ge=0, le=10_000_000)
    revol_util: float = Field(..., ge=0, le=250, description="Revolving utilization (%)")
    inq_last_6mths: float = Field(..., ge=0, le=100, description="Credit inquiries in the last 6 months. Required: informative feature, 0 in only 55% of cases; a default value would be wrong in ~1/5 of applications.")
    initial_list_status: str = Field(..., description="Initial listing status: w or f. Required: policy feature with no default value (scope §8), not inferable from a new application.")

    emp_length: Optional[str] = Field(None, description='Ex.: "10+ years", "< 1 year"')
    verification_status: Optional[str] = Field(None)
    application_type: Optional[str] = Field(None)
    delinq_2yrs: Optional[float] = Field(None, ge=0)
    pub_rec: Optional[float] = Field(None, ge=0)

    @field_validator("home_ownership")
    @classmethod
    def _check_home(cls, v):
        if v not in _HOME:
            raise ValueError(f"home_ownership must be one of {_HOME}, received {v!r}")
        return v

    @field_validator("initial_list_status")
    @classmethod
    def _check_ils(cls, v):
        if v not in _ILS:
            raise ValueError(f"initial_list_status must be one of {_ILS}, received {v!r}")
        return v

    @field_validator("purpose")
    @classmethod
    def _check_purpose(cls, v):
        if v not in _PURP:
            raise ValueError(f"purpose must be one of {_PURP}, received {v!r}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "loan_amnt": 10000, "installment": 325.5, "term": 36,
                "annual_inc": 60000, "fico_range_low": 710, "dti": 15.2,
                "earliest_cr_line": "2001-08-01", "issue_d": "2015-06-01",
                "home_ownership": _HOME[0] if _HOME else "rent",
                "purpose": _PURP[0] if _PURP else "debt_consolidation",
                "acc_open_past_24mths": 3, "open_acc": 8, "total_acc": 20,
                "revol_bal": 8500, "revol_util": 42.3,
                "initial_list_status": 'f',
                "inq_last_6mths": 1,
            }
        }
    }


class ScoreResponse(BaseModel):
    probability_default: float = Field(..., ge=0.0, le=1.0)
    decision: Literal["approve", "reject"]
    threshold: float = Field(..., description="Applied operational cutoff")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "threshold": OPERATIONAL_THRESHOLD}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    """Scores an application. Pydantic has already rejected invalid input with 422 before this point."""
    df = pd.DataFrame.from_records([req.model_dump()])
    out = score_frame(df)  # same encoding as training, via src.scoring
    p = float(out["probability_default"].iloc[0])
    d = str(out["decision"].iloc[0])
    return ScoreResponse(probability_default=p, decision=d, threshold=OPERATIONAL_THRESHOLD)
