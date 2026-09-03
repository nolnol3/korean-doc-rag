"""FastAPI 서비스.

  POST /ask     {q, mode?: "graph"|"naive"|"none", k?}  → 답변·출처·실행 경로
  GET  /health  인덱스 크기·모델·provider
  GET  /        질문 입력창 하나짜리 HTML
  GET  /docs    Swagger (자동)

실행: uvicorn kdr.api:app --port 8000
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from kdr import baseline, graph
from kdr.config import settings
from kdr.llm import model_label

app = FastAPI(title="korean-doc-rag", version="0.1.0")


class AskRequest(BaseModel):
    q: str = Field(min_length=1, max_length=500)
    mode: Literal["graph", "naive", "none"] = "graph"
    k: int | None = Field(default=None, ge=1, le=20)


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


@app.get("/health")
def health() -> dict:
    from kdr.retriever import _chunks

    try:
        n = len(_chunks())
    except FileNotFoundError:
        raise HTTPException(503, "index not built — run: make index")
    return {"status": "ok", "chunks": n, "llm": model_label(), "provider": settings.llm_provider,
            "embed_model": settings.embed_model, "retrieval_mode": settings.retrieval_mode}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        if req.mode == "graph":
            r = graph.ask(req.q)
        elif req.mode == "naive":
            r = baseline.ask(req.q, k=req.k)
        else:
            r = baseline.ask_no_retrieval(req.q)
    except FileNotFoundError:
        raise HTTPException(503, "index not built — run: make index")
    except RuntimeError as e:  # API 키 없음 등 설정 문제
        raise HTTPException(503, str(e))
    return AskResponse(**asdict(r), mode=req.mode, llm=model_label())


_PAGE = """<!doctype html><meta charset="utf-8"><title>korean-doc-rag</title>
<style>
body{font:15px/1.5 -apple-system,system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 16px;color:#222}
input,select,button{font:inherit;padding:8px 10px}input{width:100%%;box-sizing:border-box}
.row{display:flex;gap:8px;margin:12px 0}.ans{font-size:18px;margin:16px 0;padding:12px;background:#f5f7fa;border-radius:6px}
.path span{display:inline-block;background:#e8eef7;border-radius:4px;padding:2px 8px;margin:2px 4px 2px 0;font-size:13px}
.cite{border-left:3px solid #cbd5e1;padding:6px 10px;margin:8px 0;font-size:14px;color:#444}.cite b{color:#222}
.meta{color:#777;font-size:13px}
</style>
<h2>korean-doc-rag</h2>
<p class=meta>KorQuAD 1.0 문단 10,639개 · 답변은 문서 근거와 함께 · 실행 경로 표시</p>
<input id=q placeholder="질문을 입력하세요" value="구룡폭포의 높이는?">
<div class=row><select id=mode><option value=graph>graph (LangGraph)</option><option value=naive>naive RAG</option><option value=none>LLM only</option></select>
<button onclick=go()>질문</button></div>
<div id=out></div>
<script>
async function go(){const o=document.getElementById('out');o.innerHTML='<p class=meta>생성 중…</p>';
const r=await fetch('/ask',{method:'POST',headers:{'content-type':'application/json'},
 body:JSON.stringify({q:document.getElementById('q').value,mode:document.getElementById('mode').value})});
if(!r.ok){o.innerHTML='<p style="color:#b00">'+(await r.text())+'</p>';return}
const d=await r.json();
o.innerHTML='<div class=ans>'+esc(d.answer)+'</div>'
 +'<div class=path>'+d.path.map(p=>'<span>'+esc(p)+'</span>').join('')+'</div>'
 +'<p class=meta>grounded='+d.grounded+' · attempts='+d.attempts+' · calls='+d.usage.calls+' · '+d.latency_ms+'ms · '+esc(d.llm)+'</p>'
 +d.citations.map(c=>'<div class=cite><b>['+c.n+'] '+esc(c.title)+'</b> <span class=meta>'+c.score+'</span><br>'+esc(c.text)+'</div>').join('');}
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
const u=new URLSearchParams(location.search);if(u.get('q')){document.getElementById('q').value=u.get('q');if(u.get('mode'))document.getElementById('mode').value=u.get('mode');go();}
</script>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE
