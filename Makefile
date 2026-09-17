PY := .venv/bin/python

.PHONY: setup fixtures fetch index ask ablation eval serve test lint reproduce

setup:            ## venv + deps (OCR 포함. OCR 없이: uv pip install -e ".[dev]")
	uv venv -q --python 3.11 .venv && uv pip install -q -e ".[dev,ocr]"

fixtures:         ## 테스트용 PDF·HWPX·스캔 샘플 생성 (tests/fixtures)
	$(PY) tests/fixtures/make_sample_pdf.py && $(PY) tests/fixtures/make_sample_hwpx.py && $(PY) tests/fixtures/make_sample_scan.py

fetch:            ## KorQuAD 원본 내려받기 (data/raw, gitignore)
	$(PY) scripts/fetch_data.py

index:            ## 문단 청크 → 임베딩 → Chroma, BM25
	$(PY) -m kdr.ingest

ask:              ## make ask Q="질문"   (단순 RAG)
	$(PY) -m kdr.baseline "$(Q)"

ablation:         ## 검색 방식 4종 recall@k — LLM 호출 없음
	$(PY) -m kdr.eval --ablation

eval:             ## naive vs graph, 200문항
	$(PY) -m kdr.eval

serve:            ## FastAPI :8000
	.venv/bin/uvicorn kdr.api:app --host 0.0.0.0 --port 8000

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check src tests

reproduce: fetch index ablation eval   ## 처음부터 끝까지
