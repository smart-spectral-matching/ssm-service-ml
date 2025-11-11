FROM python:3.8-slim AS production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Production
RUN apt update \
    && apt install -y curl make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code
COPY . /code

RUN pip install .

# Development
FROM production AS development
RUN pip install .[dev]

