# Base image
FROM python:3.10.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

WORKDIR /
COPY pyproject.toml pyproject.toml
COPY MLops_project/ MLops_project/


RUN pip install -e . --no-deps --no-cache-dir

# ENTRYPOINT ["/bin/bash"]
ENTRYPOINT ["uvicorn", "--app-dir", "MLops_project/api", "fast_api_predict:app", "--host", "0.0.0.0", "--port", "8000"]
