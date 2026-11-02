# The API image.
#
# The core requirements are enough to serve: Qdrant, Postgres, Redis and
# MLflow are reached over the network, so the image does not need the heavy
# ML extras. Build the "full" image with:
#   docker build --build-arg INSTALL_ML=true .
#
# When the API is wired to the real MLflow registry and the real Postgres
# request log, it does need those two client libraries — but not the model
# weights. Use:
#   docker build --build-arg INSTALL_SERVING_BACKENDS=true .

FROM python:3.12-slim AS base

ARG INSTALL_ML=false
ARG INSTALL_SERVING_BACKENDS=false

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so the layer is cached across code changes.
COPY requirements.txt requirements-ml.txt requirements-serving-ml.txt ./
RUN if [ "$INSTALL_ML" = "true" ]; then \
        pip install --no-cache-dir -r requirements.txt -r requirements-ml.txt; \
    elif [ "$INSTALL_SERVING_BACKENDS" = "true" ]; then \
        pip install --no-cache-dir -r requirements.txt -r requirements-serving-ml.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .

# The corpus and the golden set travel with the image.
COPY data ./data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"

CMD ["python", "-m", "ragllmops.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
