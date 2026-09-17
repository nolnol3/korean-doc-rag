"""LangChain 연동(kdr/lc.py) — 모델·인덱스 없이 인터페이스 계약만 검증. 실제 점수는 eval 이 잰다."""
from langchain_core.documents import Document

from kdr import lc
from kdr.retriever import Hit


def _fake_retrieve(query, k=None, mode=None, collection=None):
    return [Hit(f"id{i}", f"t{i}", f"text {i} {query}", 1.0 / (i + 1)) for i in range(k)]


def test_kdr_retriever_wraps_retrieve_as_documents(monkeypatch):
    import kdr.retriever

    monkeypatch.setattr(kdr.retriever, "retrieve", _fake_retrieve)
    docs = lc.KdrRetriever(collection="c", k=3).invoke("q")
    assert [type(d) for d in docs] == [Document] * 3
    assert docs[0].metadata == {"id": "id0", "title": "t0", "score": 1.0, "kind": None}
    assert docs[0].page_content == "text 0 q"


def test_cross_encoder_rerank_sorts_and_keeps_scores(monkeypatch):
    monkeypatch.setattr(lc, "rerank_scores", lambda q, texts: [0.1, 0.9, 0.5][: len(texts)])
    docs = [Document(page_content=f"d{i}", metadata={"id": f"id{i}"}) for i in range(3)]
    out = lc.CrossEncoderRerank(top_n=2).compress_documents(docs, "q")
    assert [d.page_content for d in out] == ["d1", "d2"]
    assert out[0].metadata == {"id": "id1", "rerank_score": 0.9}


def test_rerank_retriever_pipeline(monkeypatch):
    import kdr.retriever

    monkeypatch.setattr(kdr.retriever, "retrieve", _fake_retrieve)
    monkeypatch.setattr(lc, "rerank_scores", lambda q, texts: [i * 0.01 for i in range(len(texts))])  # 뒤로 갈수록 높게
    out = lc.rerank_retriever("c", k=2).invoke("q")
    assert len(out) == 2
    assert [d.metadata["id"] for d in out] == ["id7", "id6"]  # 후보 k×4=8 중 점수 상위 2


def test_retrieve_hybrid_rerank_returns_hits_with_rerank(monkeypatch):
    import kdr.retriever as r

    class _R:
        def invoke(self, q):
            return [Document(page_content="x", metadata={"id": "a", "title": "T", "score": 0.3, "rerank_score": 0.95})]

    monkeypatch.setattr(lc, "rerank_retriever", lambda col, k: _R())
    hits = r.retrieve("q", k=1, mode="hybrid_rerank", collection="c")
    assert hits == [Hit("a", "T", "x", 0.95, rerank=0.95)]


def test_grade_rerank_mode_uses_threshold(monkeypatch):
    from kdr.config import settings
    from kdr.graph import n_grade

    monkeypatch.setattr(settings, "grade_mode", "rerank")
    monkeypatch.setattr(settings, "rerank_threshold", 0.5)
    monkeypatch.setattr(lc, "rerank_scores", lambda q, texts: [0.9, 0.1, 0.6])
    docs = [Hit(f"id{i}", "t", f"d{i}", 0.0) for i in range(3)]
    out = n_grade({"question": "q", "docs": docs, "path": [], "usage": None})
    assert [d.id for d in out["relevant"]] == ["id0", "id2"]
    assert out["path"] == ["grade:rerank:2/3"]
