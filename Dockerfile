# API de inferencia (tarefa 4.3) containerizada.
# Runtime enxuto: instala apenas requirements-api.txt, nao o ambiente de pesquisa.
FROM python:3.12-slim

# libgomp1: runtime do OpenMP, exigido pelo XGBoost em imagem slim (sem ele o
# `import xgboost` quebra mesmo com o pip install correto).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps primeiro (camada cacheavel): so o requirements enxuto
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# codigo: o pacote src/ inteiro (imports tardios exigem a cadeia completa:
# api -> scoring -> cleaning -> features -> data -> economics) e os artefatos
# json que a API le em runtime (ja vao junto com COPY src/).
COPY src/ ./src/
# so o modelo que a API usa (nao o baseline logistico, nao usado pela API)
COPY models/xgb_final.joblib ./models/xgb_final.joblib

# validacao em build-time: se o COPY esqueceu qualquer modulo da cadeia, o build
# FALHA aqui em vez de so quebrar em runtime.
RUN python -c "import src.api; print('import src.api OK no container')"

EXPOSE 8000
# 1 worker: inferencia deterministica, sem estado compartilhado; simples e suficiente.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
