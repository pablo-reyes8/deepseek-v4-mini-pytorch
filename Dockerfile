FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt README.md LICENSE ./
COPY src ./src
COPY training ./training
COPY data ./data
COPY tests ./tests
COPY config ./config

RUN pip install --upgrade pip \
    && pip install -e ".[dev,data]"

CMD ["pytest"]
