# Inference API (task 4.3), containerized.
# Lean runtime: installs only requirements-api.txt, not the research environment.
FROM python:3.12-slim

# libgomp1: OpenMP runtime, required by XGBoost in a slim image (without it
# `import xgboost` breaks even with the correct pip install).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps first (cacheable layer): just the lean requirements
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# code: the entire src/ package (late imports require the full chain:
# api -> scoring -> cleaning -> features -> data -> economics) and the
# json artifacts the API reads at runtime (already included via COPY src/).
COPY src/ ./src/
# only the model the API uses (not the logistic baseline, unused by the API)
COPY models/xgb_final.joblib ./models/xgb_final.joblib

# build-time validation: if the COPY missed any module in the chain, the build
# FAILS here instead of only breaking at runtime.
RUN python -c "import src.api; print('import src.api OK no container')"

EXPOSE 8000
# 1 worker: deterministic inference, no shared state; simple and sufficient.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
