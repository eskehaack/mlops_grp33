# Base image
FROM python:3.11.7-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY MLops_project/ MLops_project/

COPY data/processed/training_split_stats.json data/processed/training_split_stats.json
COPY data/processed/label_mapping.json data/processed/label_mapping.json

COPY test_images test_images
COPY outputs/2024-01-10/13-00-17/models/epoch=0-val_loss=0.00.ckpt outputs/2024-01-10/13-00-17/models/epoch=0-val_loss=0.00.ckpt


WORKDIR /
RUN --mount=type=cache,target=~/pip/.cache \
    pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "MLops_project/predict_model.py","outputs/2024-01-10/13-00-17/models/epoch=0-val_loss=0.00.ckpt","test_images"]
