"""
API de inferencia (FastAPI) para o modelo de default do Lending Club.

Tarefa 4.3 do Projeto 4, Fase 2. Casca fina sobre src.scoring (logica pura).
- Entrada: campos crus de origination. Obrigatorios = os de alto-SHAP (Model Card
  §8); o restante do bloco de bureau e opcional e, se omitido, cai no MESMO
  mecanismo de sentinela/missingness do treino (FACTS §4) -- nao e um atalho, e o
  design do projeto. Omitir muitos campos aproxima o perfil do regime sentinela e
  deve disparar checagem de PSI (ver monitor de drift, tarefa 4.4).
- term aceita SOMENTE 36. O modelo foi treinado so em 36 meses e e declaradamente
  nao-transferivel para 60 (scope §9, Model Card §4, README). Aceitar 60 seria
  pontuar algo que o modelo nao sabe pontuar -- a API recusa com mensagem que
  aponta a limitacao, aplicando-a em vez de so descreve-la.
- Saida: tipada. probability_default em [0,1] + decision in {approve, reject}
  no corte operacional 0.31.
- Erros: input faltante/fora de faixa -> 422 do proprio Pydantic (nao um 500 mudo).

Inferencia e deterministica e barata: carrega o artefato e pontua, nunca re-treina.
As pegadinhas de reprodutibilidade (seed, n_jobs, ordem de linha) sao do TREINO.
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

# --- contrato descoberto em runtime (Bloco 2a) ---
_CONTRACT = json.loads((Path(__file__).parent / "_api_contract.json").read_text())
_HOME = _CONTRACT["categories"].get("home_ownership", ["mortgage", "rent", "own", "other"])
_PURP = _CONTRACT["categories"].get("purpose", ["debt_consolidation", "credit_card", "other"])
_ILS = _CONTRACT["categories"].get("initial_list_status", ['f', 'w'])


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()  # carrega o .joblib uma vez, no startup (nao na 1a requisicao)
    yield


app = FastAPI(
    title="Lending Club — Credit Default Scoring API",
    version="1.0.0",
    description=(
        "Segunda camada de decisao sobre emprestimos ja aprovados. Estima "
        "P(default | approved) e decide approve/reject no corte de lucro (0.31). "
        "NAO pontua solicitantes rejeitados (vies de selecao, ver Model Card §9). "
        "So aceita term=36 (modelo nao-transferivel para 60 meses). "
        "Score e otimista (subestima default); nao usar como limite inferior."
    ),
    lifespan=lifespan,
)


class ScoreRequest(BaseModel):
    """Campos crus de um pedido de credito. Obrigatorios = alto-SHAP; resto opcional."""
    loan_amnt: float = Field(..., gt=0, le=100_000, description="Valor solicitado (USD)")
    installment: float = Field(..., gt=0, le=2_000, description="Parcela mensal (USD)")
    term: Literal[36] = Field(..., description="Prazo em meses. So 36 (modelo nao serve para 60, ver Model Card §4)")
    annual_inc: float = Field(..., ge=0, le=15_000_000, description="Renda anual auto-reportada (USD)")
    fico_range_low: float = Field(..., ge=300, le=850, description="Limite inferior FICO na origem")
    dti: float = Field(..., ge=0, le=100, description="Debt-to-income (%). >100 e impossivel (scope §2)")
    earliest_cr_line: str = Field(..., description="Data da linha de credito mais antiga (YYYY-MM-DD)")
    issue_d: str = Field(..., description="Mes de emissao do emprestimo (YYYY-MM-DD)")
    home_ownership: str = Field(..., description=f"Um de: {_HOME}")
    purpose: str = Field(..., description=f"Um de: {_PURP}")
    acc_open_past_24mths: float = Field(..., ge=0, le=100)
    open_acc: float = Field(..., ge=0, le=200)
    total_acc: float = Field(..., ge=0, le=300)
    revol_bal: float = Field(..., ge=0, le=10_000_000)
    revol_util: float = Field(..., ge=0, le=250, description="Utilizacao rotativa (%)")
    inq_last_6mths: float = Field(..., ge=0, le=100, description="Consultas de credito nos ultimos 6 meses. Obrigatorio: feature informativa, 0 em so 55% dos casos; default erraria em ~1/5 dos pedidos.")
    initial_list_status: str = Field(..., description="Status de listagem inicial: w ou f. Obrigatorio: feature de politica sem valor default (scope §8), nao inferivel de um pedido novo.")

    emp_length: Optional[str] = Field(None, description='Ex.: "10+ years", "< 1 year"')
    verification_status: Optional[str] = Field(None)
    application_type: Optional[str] = Field(None)
    delinq_2yrs: Optional[float] = Field(None, ge=0)
    pub_rec: Optional[float] = Field(None, ge=0)

    @field_validator("home_ownership")
    @classmethod
    def _check_home(cls, v):
        if v not in _HOME:
            raise ValueError(f"home_ownership deve ser um de {_HOME}, recebido {v!r}")
        return v

    @field_validator("initial_list_status")
    @classmethod
    def _check_ils(cls, v):
        if v not in _ILS:
            raise ValueError(f"initial_list_status deve ser um de {_ILS}, recebido {v!r}")
        return v

    @field_validator("purpose")
    @classmethod
    def _check_purpose(cls, v):
        if v not in _PURP:
            raise ValueError(f"purpose deve ser um de {_PURP}, recebido {v!r}")
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
    threshold: float = Field(..., description="Corte operacional aplicado")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "threshold": OPERATIONAL_THRESHOLD}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    """Pontua um pedido. Pydantic ja rejeitou input invalido com 422 antes daqui."""
    df = pd.DataFrame.from_records([req.model_dump()])
    out = score_frame(df)  # mesmo encoding do treino, via src.scoring
    p = float(out["probability_default"].iloc[0])
    d = str(out["decision"].iloc[0])
    return ScoreResponse(probability_default=p, decision=d, threshold=OPERATIONAL_THRESHOLD)
