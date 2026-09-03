"""그래프 RAG — LangGraph.

  retrieve → grade ─┬─ 관련 문서 있음 → generate → verify ─┬─ grounded → END
                    │                                      └─ 아니면 → rewrite → retrieve  (attempts < max)
                    └─ 없음 → rewrite → retrieve  (attempts < max)
                              └─ 소진 → abstain → END

단순 RAG(baseline.py)와 같은 retriever, 같은 generate 프롬프트. 차이는 grade·rewrite·verify뿐.
실행: python -m kdr.graph "질문"
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from kdr.baseline import Result, _citations
from kdr.config import settings
from kdr.llm import Usage, complete, complete_json, model_label
from kdr.prompts import (
    GENERATE_SYSTEM, GENERATE_USER, GRADE_SYSTEM, GRADE_USER,
    REWRITE_SYSTEM, REWRITE_USER, VERIFY_SYSTEM, VERIFY_USER, render_context,
)
from kdr.retriever import Hit, retrieve

ABSTAIN = "문서에서 근거를 찾지 못했습니다."


class State(TypedDict):
    question: str        # 원 질문 (불변)
    query: str           # 현재 검색 질의 (rewrite가 갱신)
    docs: list[Hit]
    relevant: list[Hit]
    answer: str
    grounded: bool | None
    attempts: int        # rewrite 횟수
    tried: list[str]     # 지금까지 쓴 검색 질의
    seen_relevant: list[str]  # 마지막 generate에 쓴 문서 id — 같은 근거면 재생성하지 않는다
    path: list[str]
    usage: Usage


# ── 노드 ──────────────────────────────────────────────────────────────────────

def n_retrieve(s: State) -> dict:
    return {"docs": retrieve(s["query"]), "path": s["path"] + ["retrieve"]}


def _grade_one(question: str, d: Hit, usage: Usage) -> bool:
    out = complete_json(GRADE_SYSTEM, GRADE_USER.format(question=question, title=d.title, text=d.text),
                        name="grade", usage=usage, max_tokens=32)
    v = out.get("relevant", False)
    return v if isinstance(v, bool) else str(v).lower() in ("true", "yes", "1")


def n_grade(s: State) -> dict:
    """문서 5개를 각각 yes/no로 — 병렬. 한 프롬프트에 5개를 넣는 것보다 짧고(토큰 1/4), 7B에 더 정확하다."""
    with ThreadPoolExecutor(max_workers=len(s["docs"]) or 1) as ex:
        flags = list(ex.map(lambda d: _grade_one(s["question"], d, s["usage"]), s["docs"]))
    relevant = [d for d, ok in zip(s["docs"], flags) if ok]
    return {"relevant": relevant, "path": s["path"] + [f"grade:{len(relevant)}/{len(s['docs'])}"]}


def n_rewrite(s: State) -> dict:
    tried = s["tried"] + [s["query"]]
    out = complete_json(
        REWRITE_SYSTEM,
        REWRITE_USER.format(question=s["question"], tried="\n".join(f"- {q}" for q in tried)),
        name="rewrite", usage=s["usage"],
    )
    query = (out.get("query") or "").strip() or s["question"]
    tag = "rewrite:dup" if query in tried else f"rewrite:{query[:30]}"
    return {"query": query, "tried": tried, "attempts": s["attempts"] + 1, "path": s["path"] + [tag]}


def n_generate(s: State) -> dict:
    ids = [d.id for d in s["relevant"]]
    answer = complete(
        GENERATE_SYSTEM,
        GENERATE_USER.format(context=render_context(s["relevant"]), question=s["question"]),
        name="generate", usage=s["usage"],
    )
    return {"answer": answer, "seen_relevant": ids, "path": s["path"] + ["generate"]}


def n_verify(s: State) -> dict:
    if s["answer"].startswith(ABSTAIN):
        return {"grounded": True, "path": s["path"] + ["verify:abstain"]}
    out = complete_json(
        VERIFY_SYSTEM,
        VERIFY_USER.format(context=render_context(s["relevant"]), question=s["question"], answer=s["answer"]),
        name="verify", usage=s["usage"], max_tokens=160,
    )
    grounded = bool(out.get("grounded", False))
    return {"grounded": grounded, "path": s["path"] + [f"verify:{'ok' if grounded else 'fail'}"]}


def n_stop(s: State) -> dict:
    return {"path": s["path"] + ["stop:no-new-evidence"]}


def n_abstain(s: State) -> dict:
    return {"answer": ABSTAIN, "grounded": True, "relevant": [], "path": s["path"] + ["abstain"]}


# ── 분기 ──────────────────────────────────────────────────────────────────────

def after_grade(s: State) -> str:
    if s["relevant"]:
        # verify:fail 뒤의 재검색인데 근거 문서가 그대로면 같은 답이 나온다 — 여기서 멈춘다
        if s["answer"] and [d.id for d in s["relevant"]] == s["seen_relevant"]:
            return "stop"
        return "generate"
    if s["answer"]:
        return "stop"  # 이미 답이 있는데 재검색이 빈손이면 그 답을 유지
    return "rewrite" if s["attempts"] < settings.max_attempts else "abstain"


def after_verify(s: State) -> str:
    if s["grounded"]:
        return END
    return "rewrite" if s["attempts"] < settings.max_attempts else END


# ── 그래프 ────────────────────────────────────────────────────────────────────

def build():
    g = StateGraph(State)
    for name, fn in [("retrieve", n_retrieve), ("grade", n_grade), ("rewrite", n_rewrite),
                     ("generate", n_generate), ("verify", n_verify), ("abstain", n_abstain), ("stop", n_stop)]:
        g.add_node(name, fn)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, {"generate": "generate", "rewrite": "rewrite", "abstain": "abstain", "stop": "stop"})
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", after_verify, {"rewrite": "rewrite", END: END})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("abstain", END)
    g.add_edge("stop", END)
    return g.compile()


_GRAPH = None


def ask(question: str) -> Result:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build()
    t0 = time.perf_counter()
    usage = Usage()
    s = _GRAPH.invoke({
        "question": question, "query": question, "docs": [], "relevant": [],
        "answer": "", "grounded": None, "attempts": 0, "tried": [], "seen_relevant": [], "path": [], "usage": usage,
    })
    return Result(
        question=question,
        answer=s["answer"],
        citations=_citations(s["relevant"]),
        path=s["path"],
        grounded=s["grounded"],
        attempts=s["attempts"] + 1,
        usage=usage.as_dict(),
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "구룡폭포의 높이는?"
    r = ask(q)
    print(f"Q: {r.question}\nA: {r.answer}\n")
    for c in r.citations:
        print(f"  [{c['n']}] {c['title']}  {c['text'][:60]}…")
    print(f"\npath: {' → '.join(r.path)}")
    print(f"grounded={r.grounded}  attempts={r.attempts}  calls={r.usage['calls']}  {r.latency_ms}ms  llm={model_label()}")
