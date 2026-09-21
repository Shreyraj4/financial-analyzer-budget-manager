# One image serves the API and the dashboard (at /app). Trained model files are not in git,
# so the build trains them from the bundled synthetic dataset.
FROM python:3.12-slim

WORKDIR /srv
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY backend backend
COPY data data
COPY docs docs
COPY frontend frontend

WORKDIR /srv/backend
RUN python -m app.ml.categorization.train \
 && python -m app.ml.forecasting.train \
 && python -m app.ml.anomaly.train \
 && python -m app.ml.clustering.train \
 && python -m app.ml.recommendation.train

# Migrate, then serve. Railway provides $PORT.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
