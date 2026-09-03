"""단순 RAG: retrieve → generate. 그래프와 같은 retriever, 같은 generate 프롬프트를 쓴다.

실행: python -m kdr.baseline "질문"
"""
from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass

from kdr.config import settings
from kdr.llm import Usage, complete, model_label
from kdr.prompts import GENERATE_SYSTEM, GENERATE_USER, render_context
from kdr.retriever import Hit, retrieve


@dataclass
class Result:
    question: str
    answer: str
    citations: list[dict]
    path: list[str]
    grounded: bool | None
    attempts: int
    usage: dict
    latency_ms: int


def _citations(hits: list[Hit]) -> list[dict]:
    return [{"n": i + 1, "id": h.id, "title": h.title, "text": h.text, "score": round(h.score, 4)} for i, h in enumerate(hits)]


def ask(question: str, k: int | None = None, mode: str | None = None, collection: str | None = None) -> Result:
    t0 = time.perf_counter()
    usage = Usage()
    hits = retrieve(question, k=k, mode=mode, collection=collection)
    answer = complete(
        GENERATE_SYSTEM,
        GENERATE_USER.format(context=render_context(hits), question=question),
        name="generate",
        usage=usage,
    )
    return Result(
        question=question,
        answer=answer,
        citations=_citations(hits),
        path=["retrieve", "generate"],
        grounded=None,
        attempts=1,
        usage=usage.as_dict(),
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "대한민국의 수도는 어디인가요?"
    r = ask(q)
    print(f"Q: {r.question}\nA: {r.answer}\n")
    for c in r.citations:
        print(f"  [{c['n']}] {c['title']}  ({c['score']})  {c['text'][:60]}…")
    print(f"\n{r.path}  calls={r.usage['calls']}  {r.latency_ms}ms  mode={settings.retrieval_mode}  llm={model_label()}")


NONE_SYSTEM = """당신은 한국어 질문에 답하는 도우미입니다. 알고 있는 지식으로 답합니다.
**정답만 씁니다.** 명사구나 짧은 구로. 문장으로 풀어 쓰지 않고, 질문을 반복하지 않고, 설명을 붙이지 않습니다.
예) "74미터"  /  "1986년 1월 9일"  /  "올해의 가수상"
정말 모르는 경우에만 "모르겠습니다."라고 답합니다."""


def ask_no_retrieval(question: str) -> Result:
    """검색 없이 LLM만. '그냥 호출'과 RAG의 차이를 재는 대조군."""
    t0 = time.perf_counter()
    usage = Usage()
    answer = complete(NONE_SYSTEM, f"질문: {question}\n\n답변은 정답만 명사구로 씁니다. 문장으로 쓰지 않습니다.\n답변:", name="generate", usage=usage)
    return Result(
        question=question,
        answer=answer,
        citations=[],
        path=["generate"],
        grounded=None,
        attempts=1,
        usage=usage.as_dict(),
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )
