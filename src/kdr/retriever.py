"""검색 계층. 단순 RAG와 그래프 RAG가 같은 함수를 쓴다.

mode:
  vector   bge-m3 코사인 top-k
  bm25     kiwi 형태소 BM25
  bm25_ws  공백 분리 BM25 (ablation)
  hybrid   vector + bm25 를 RRF로 합침 (기본)
"""
from __future__ import annotations

import json
import pickle
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


@lru_cache(maxsize=1)
def _chunks() -> dict[str, dict]:
    with settings.chunks_path.open() as f:
        return {r["id"]: r for r in map(json.loads, f)}


@lru_cache(maxsize=1)
def _bm25():
    with settings.bm25_path.open("rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _collection():
    return chromadb.PersistentClient(path=str(settings.chroma_dir)).get_collection(settings.collection)


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embed_model)


def _hit(cid: str, score: float) -> Hit:
    c = _chunks()[cid]
    return Hit(cid, c["title"], c["text"], float(score))


def _vector(query: str, k: int) -> list[Hit]:
    q = _embedder().encode([query], normalize_embeddings=True)[0].tolist()
    res = _collection().query(query_embeddings=[q], n_results=k, include=["distances"])
    return [_hit(cid, 1.0 - d) for cid, d in zip(res["ids"][0], res["distances"][0])]


def _bm25_search(query: str, k: int, variant: str) -> list[Hit]:
    idx = _bm25()
    toks = tokenize_kiwi(query) if variant == "kiwi" else tokenize_ws(query)
    scores = idx[variant].get_scores(toks)
    top = np.argsort(scores)[::-1][:k]
    return [_hit(idx["ids"][i], scores[i]) for i in top if scores[i] > 0]


def _rrf(*ranked: list[Hit], k: int, c: int = 60) -> list[Hit]:
    fused: dict[str, float] = {}
    for lst in ranked:
        for rank, h in enumerate(lst):
            fused[h.id] = fused.get(h.id, 0.0) + 1.0 / (c + rank + 1)
    top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
    return [_hit(cid, s) for cid, s in top]


def retrieve(query: str, k: int | None = None, mode: str | None = None) -> list[Hit]:
    k = k or settings.top_k
    mode = mode or settings.retrieval_mode
    if mode == "vector":
        return _vector(query, k)
    if mode == "bm25":
        return _bm25_search(query, k, "kiwi")
    if mode == "bm25_ws":
        return _bm25_search(query, k, "ws")
    if mode == "hybrid":
        # 각 계열에서 넉넉히 뽑아 합친다. RRF는 순위만 보므로 점수 스케일이 달라도 된다.
        return _rrf(_vector(query, k * 4), _bm25_search(query, k * 4, "kiwi"), k=k)
    raise ValueError(f"unknown retrieval mode: {mode}")
