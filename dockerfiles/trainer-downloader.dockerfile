FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir huggingface_hub aiohttp pydantic transformers

COPY scripts/ scripts/
COPY core/ core/

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "scripts/trainer_downloader.py"]
