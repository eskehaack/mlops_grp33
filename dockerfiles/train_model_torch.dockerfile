# Base image
FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN python --version

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements_container.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

WORKDIR /
COPY pyproject.toml pyproject.toml
COPY MLops_project/ MLops_project/



RUN pip install -e . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "MLops_project/train_model.py"]
