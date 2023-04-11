FROM python:3.8-slim as production

ARG POETRY_HTTP_BASIC_PYPI_USERNAME
ARG POETRY_HTTP_BASIC_PYPI_PASSWORD

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    POETRY_VERSION=1.2.0
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Production
RUN apt update \
    && apt install -y curl make \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.create false

WORKDIR /code
COPY . /code

RUN poetry install --only main

# Development
FROM production as development
RUN poetry install
