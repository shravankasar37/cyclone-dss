# Backend API image (FastAPI + models). Works on Hugging Face Spaces (Docker SDK, port 7860),
# Google Cloud Run, Railway or any Docker host. Secrets come from environment variables, never .env.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860 \
    WEIGHTS_DIR=/app/models/weights DB_PATH=/tmp/app.db \
    HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r /app/backend/requirements.txt

COPY ml /app/ml
COPY backend /app/backend
COPY models/weights /app/models/weights
COPY data/processed /app/data/processed
COPY data/raw/era5 /app/data/raw/era5

# Hugging Face Spaces runs containers as uid 1000; make the writable cache dirs available to it
RUN mkdir -p /app/data/processed/basin_cache && chmod -R a+rwX /app/data

WORKDIR /app/backend
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
