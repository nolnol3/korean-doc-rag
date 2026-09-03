FROM python:3.11-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HOME=/app/.hf

# torch는 CPU 빌드만 (이미지 ~1GB 절감). 임베딩은 질의 시 1건씩이라 CPU로 충분
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
RUN pip install .

EXPOSE 8000
CMD ["uvicorn", "kdr.api:app", "--host", "0.0.0.0", "--port", "8000"]
