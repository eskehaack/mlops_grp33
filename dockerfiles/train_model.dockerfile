# Base image
FROM python:3.10.12-slim


RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt --no-cache-dir


COPY pyproject.toml pyproject.toml
COPY MLops_project/ MLops_project/
COPY data/processed data/processed

WORKDIR /

RUN pip install -e . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "MLops_project/train_model.py"]
