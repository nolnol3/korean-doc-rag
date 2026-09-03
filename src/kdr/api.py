"""FastAPI 서비스.

  POST /ask          {q, mode?, k?, collection?}  → 답변·출처·실행 경로
  POST /upload       multipart files[] (+collection)  → PDF/HWPX 파싱 후 인덱스에 추가
  GET  /collections  검색 가능한 컬렉션과 청크 수
  GET  /health       인덱스·모델·provider
  GET  /             UI (static/index.html)
  GET  /docs         Swagger (자동)

실행: uvicorn kdr.api:app --port 8000
"""
from __future__ import annotations

import re
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from kdr import baseline, graph
from kdr.config import settings
from kdr.llm import model_label

STATIC = Path(__file__).parent / "static"
_SAFE_NAME = re.compile(r"[^\w.\-가-힣 ]")

app = FastAPI(title="korean-doc-rag", version="0.1.0")


class AskRequest(BaseModel):
    q: str = Field(min_length=1, max_length=500)
    mode: Literal["graph", "naive", "none"] = "graph"
    k: int | None = Field(default=None, ge=1, le=20)
    collection: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_\-]{1,40}$")


class Citation(BaseModel):
    n: int
    id: str
    title: str
    text: str
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    path: list[str]
    grounded: bool | None
    attempts: int
    usage: dict
    latency_ms: int
    mode: str
    llm: str
    collection: str


def _collection_or_404(name: str | None) -> str:
    from kdr.retriever import list_collections

    name = name or settings.collection
    if name not in list_collections():
        raise HTTPException(404, f"collection '{name}' not found — 업로드하거나 make index")
    return name


@app.get("/collections")
def collections() -> list[dict]:
    from kdr.retriever import _client

    out = []
    for c in _client().list_collections():
        out.append({"name": c.name, "chunks": c.count(), "default": c.name == settings.collection})
    return sorted(out, key=lambda x: (not x["default"], x["name"]))


@app.get("/health")
def health() -> dict:
    from kdr.retriever import _chunks

    try:
        n = len(_chunks(settings.collection))
    except FileNotFoundError:
        raise HTTPException(503, "index not built — run: make index")
    return {"status": "ok", "chunks": n, "llm": model_label(), "provider": settings.llm_provider,
            "embed_model": settings.embed_model, "retrieval_mode": settings.retrieval_mode}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    col = _collection_or_404(req.collection) if req.mode != "none" else (req.collection or settings.collection)
    try:
        if req.mode == "graph":
            r = graph.ask(req.q, collection=col)
        elif req.mode == "naive":
            r = baseline.ask(req.q, k=req.k, collection=col)
        else:
            r = baseline.ask_no_retrieval(req.q)
    except FileNotFoundError:
        raise HTTPException(503, "index not built — run: make index")
    except RuntimeError as e:  # API 키 없음 등 설정 문제
        raise HTTPException(503, str(e))
    return AskResponse(**asdict(r), mode=req.mode, llm=model_label(), collection=col)


@app.post("/upload")
def upload(files: list[UploadFile] = File(...), collection: str = Form("docs")) -> dict:
    """PDF/HWPX를 받아 파싱하고 컬렉션에 추가한다. 같은 내용은 중복 추가되지 않는다."""
    from kdr.ingest_docs import PARSERS, add_files
    from kdr.retriever import invalidate

    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,40}", collection) or collection == "korquad":
        raise HTTPException(400, "collection 이름은 영문·숫자·_·- 만, korquad 는 예약")
    dest = settings.uploads_dir / collection
    dest.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for f in files:
        name = _SAFE_NAME.sub("_", Path(f.filename or "file").name)
        if Path(name).suffix.lower() not in PARSERS:
            raise HTTPException(415, f"{name}: PDF 또는 HWPX만 받습니다")
        path = dest / name
        with path.open("wb") as out:
            shutil.copyfileobj(f.file, out, length=1 << 20)
        if path.stat().st_size > 50 << 20:
            path.unlink()
            raise HTTPException(413, f"{name}: 50MB 초과")
        saved.append(path)
    try:
        result = add_files(saved, collection)
    except Exception as e:
        raise HTTPException(500, f"파싱 실패: {type(e).__name__}: {str(e)[:200]}")
    invalidate(collection)
    return result


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
