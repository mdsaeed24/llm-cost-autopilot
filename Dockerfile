# One image for api, worker and dashboard (different commands in docker-compose.yml).
# Python 3.14 + pinned scikit-learn: the classifier is a pickle, which only loads reliably
# with the same scikit-learn version it was trained with.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, in their own layer, so code changes don't reinstall everything.
COPY pyproject.toml README.md ./
RUN mkdir -p src/autopilot && touch src/autopilot/__init__.py \
    && pip install -e ".[dashboard]"

# Application code and config. Models and data are mounted at runtime (see docker-compose.yml).
COPY src ./src
COPY config ./config
COPY scripts ./scripts

RUN useradd --create-home --uid 1000 autopilot \
    && mkdir -p /app/var /app/models /app/data \
    && chown -R autopilot:autopilot /app
USER autopilot

EXPOSE 8000 8501
CMD ["uvicorn", "autopilot.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
