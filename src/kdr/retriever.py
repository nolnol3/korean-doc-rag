"""검색 계층. 단순 RAG와 그래프 RAG가 같은 함수를 쓴다.

mode:
  vector   bge-m3 코사인 top-k
  bm25     kiwi 형태소 BM25
  bm25_ws  공백 분리 BM25 (ablation)
  hybrid   vector + bm25 를 RRF로 합침 (기본)
  hybrid_rerank  hybrid 로 k×4 후보 → cross-encoder 재정렬 (LangChain ContextualCompressionRetriever, kdr/lc.py)
"""
from __future__ import annotations

import json
import pickle
import threading
from dataclasses import dataclass
from functools import lru_cache

import chromadb
import numpy as np

from kdr.config import settings
from kdr.tokenize import tokenize_kiwi, tokenize_ws


@dataclass
class Hit:
    id: str
    title: str
    text: str
    score: float
    rerank: float | None = None  # hybrid_rerank 일 때 cross-encoder 점수


@lru_cache(maxsize=8)
def _chunks(collection: str) -> dict[str, dict]:
    with settings.chunks_path_for(collection).open() as f:
        return {r["id"]: r for r in map(json.loads, f)}


@lru_cache(maxsize=8)
def _bm25(collection: str):
    with settings.bm25_path_for(collection).open("rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _client():
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


@lru_cache(maxsize=8)
def _collection(collection: str):
    return _client().get_collection(collection)


def invalidate(collection: str | None = None) -> None:
    """인제스트 뒤 캐시를 비운다 (서버가 새 청크를 보게)."""
    _chunks.cache_clear()
    _bm25.cache_clear()
    _collection.cache_clear()


def list_collections() -> list[str]:
    return sorted(c.name for c in _client().list_collections())


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embed_model)


def _hit(col: str, cid: str, score: float) -> Hit:
    c = _chunks(col)[cid]
    return Hit(cid, c["title"], c["text"], float(score))


# 프로세스 안의 모든 로컬 모델 forward(bge-m3, cross-encoder)가 공유하는 락.
# MPS 는 동시 forward 에 안전하지 않다 — 모델별로 락을 따로 두면 서로 다른 모델이 겹쳐 Metal 어설션으로 죽는다.
model_lock = threading.Lock()


def _vector(query: str, k: int, col: str) -> list[Hit]:
    n = _collection(col).count()
    if n == 0:
        return []
    with model_lock:
        q = _embedder().encode([query], normalize_embeddings=True)[0].tolist()
    res = _collection(col).query(query_embeddings=[q], n_results=min(k, n), include=["distances"])
    return [_hit(col, cid, 1.0 - d) for cid, d in zip(res["ids"][0], res["distances"][0])]


def _bm25_search(query: str, k: int, variant: str, col: str) -> list[Hit]:
    idx = _bm25(col)
    toks = tokenize_kiwi(query) if variant == "kiwi" else tokenize_ws(query)
    scores = idx[variant].get_scores(toks)
    top = np.argsort(scores)[::-1][:k]
    return [_hit(col, idx["ids"][i], scores[i]) for i in top if scores[i] > 0]


def _rrf(col: str, *ranked: list[Hit], k: int, c: int = 60) -> list[Hit]:
    fused: dict[str, float] = {}
    for lst in ranked:
        for rank, h in enumerate(lst):
            fused[h.id] = fused.get(h.id, 0.0) + 1.0 / (c + rank + 1)
    top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
    return [_hit(col, cid, s) for cid, s in top]


def retrieve(query: str, k: int | None = None, mode: str | None = None, collection: str | None = None) -> list[Hit]:
    k = k or settings.top_k
    mode = mode or settings.retrieval_mode
    col = collection or settings.collection
    if mode == "vector":
        return _vector(query, k, col)
    if mode == "bm25":
        return _bm25_search(query, k, "kiwi", col)
    if mode == "bm25_ws":
        return _bm25_search(query, k, "ws", col)
    if mode == "hybrid":
        # 각 계열에서 넉넉히 뽑아 합친다. RRF는 순위만 보므로 점수 스케일이 달라도 된다.
        return _rrf(col, _vector(query, k * 4, col), _bm25_search(query, k * 4, "kiwi", col), k=k)
    if mode == "hybrid_rerank":
        from kdr.lc import rerank_retriever

        docs = rerank_retriever(col, k).invoke(query)
        return [Hit(d.metadata["id"], d.metadata["title"], d.page_content, d.metadata["rerank_score"],
                    rerank=d.metadata["rerank_score"]) for d in docs]
    raise ValueError(f"unknown retrieval mode: {mode}")
