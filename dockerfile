# python image
FROM python:3.11.7-slim AS base

# environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.2

# install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl && \
    rm -rf /var/lib/apt/lists/*

# install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# copy only dependency file to leverage Docker cache
COPY pyproject.toml ./

# install Python dependencies globally
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-dev

# working directory for app code
WORKDIR /api

# copy rest of the application code into container
COPY . .

# run the app using Gunicorn with Uvicorn workers
CMD ["gunicorn", "api.main:api", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000"]
