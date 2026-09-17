FROM python:3.11-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HOME=/app/.hf OCR_MODEL_DIR=/app/.hf/easyocr

# opencv-headless(easyocr 의존) 런타임 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# torch는 CPU 빌드만 (이미지 ~1GB 절감). 임베딩은 질의 시 1건씩이라 CPU로 충분
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
RUN pip install ".[ocr]"

EXPOSE 8000
CMD ["uvicorn", "kdr.api:app", "--host", "0.0.0.0", "--port", "8000"]
