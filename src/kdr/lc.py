"""LangChain 연동 — 검색 계층을 LangChain 인터페이스로 노출하고 cross-encoder reranker를 붙인다.

  KdrRetriever         BaseRetriever            retriever.retrieve() 결과를 Document 로 감싼다
  CrossEncoderRerank   BaseDocumentCompressor   bge-reranker-v2-m3 로 재정렬. 점수는 metadata["rerank_score"] 에 남긴다
  rerank_retriever()   ContextualCompressionRetriever(KdrRetriever(k×4) → CrossEncoderRerank(top_n=k))

langchain-classic 의 CrossEncoderReranker 는 점수를 버리고(grade 임계값에 못 씀) 모델 래퍼는 sunset 된
langchain-community 에 있어, core 인터페이스(BaseRetriever / BaseDocumentCompressor)로 직접 구현했다.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun, Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.retrievers import BaseRetriever

from kdr.config import settings
from kdr.retriever import model_lock


def to_document(h) -> Document:  # h: retriever.Hit (순환 import 회피)
    return Document(page_content=h.text, metadata={"id": h.id, "title": h.title, "score": h.score, "kind": h.kind})


class KdrRetriever(BaseRetriever):
    """hybrid(bge-m3 + kiwi BM25, RRF) 검색을 LangChain Retriever 로. `.invoke(query)` → list[Document]."""

    collection: str
    k: int = 5
    mode: str = "hybrid"

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        from kdr.retriever import retrieve

        return [to_document(h) for h in retrieve(query, k=self.k, mode=self.mode, collection=self.collection)]


@lru_cache(maxsize=1)
def _cross_encoder():
    import torch
    from sentence_transformers import CrossEncoder

    # GPU(MPS/CUDA)면 fp16 — 점수 차이는 소수 셋째 자리, 속도 약 2배. CPU는 fp16 연산이 느려 fp32 유지
    half = torch.backends.mps.is_available() or torch.cuda.is_available()
    return CrossEncoder(settings.rerank_model, max_length=512,
                        model_kwargs={"torch_dtype": torch.float16} if half else {})


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    """(query, text) 쌍마다 0~1 관련도. 20쌍에 MPS 0.7초, CPU 2~3초."""
    if not texts:
        return []
    with model_lock:
        return [float(s) for s in _cross_encoder().predict([(query, t) for t in texts])]


class CrossEncoderRerank(BaseDocumentCompressor):
    """cross-encoder 점수로 재정렬해 top_n 만 남긴다. 점수는 metadata["rerank_score"]."""

    top_n: int = 5

    def compress_documents(self, documents: Any, query: str, callbacks: Callbacks | None = None) -> list[Document]:
        docs = list(documents)
        scored = sorted(zip(docs, rerank_scores(query, [d.page_content for d in docs])), key=lambda x: -x[1])
        return [Document(page_content=d.page_content, metadata={**d.metadata, "rerank_score": s})
                for d, s in scored[: self.top_n]]


def rerank_retriever(collection: str, k: int, candidates: int | None = None) -> ContextualCompressionRetriever:
    """hybrid 로 후보 k×4 개를 뽑아 reranker 로 k 개를 고른다."""
    return ContextualCompressionRetriever(
        base_compressor=CrossEncoderRerank(top_n=k),
        base_retriever=KdrRetriever(collection=collection, k=candidates or k * 4, mode="hybrid"),
    )
