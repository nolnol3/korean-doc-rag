"""분기 함수만 검증 — LLM·인덱스 불필요."""
from langgraph.graph import END

from kdr.config import settings
from kdr.graph import after_grade, after_verify


def _state(**kw):
    base = {"question": "q", "query": "q", "collection": "korquad", "docs": [], "relevant": [], "answer": "",
            "grounded": None, "attempts": 0, "tried": [], "seen_relevant": [], "path": [], "usage": None}
    base.update(kw)
    return base


def test_grade_relevant_goes_to_generate():
    assert after_grade(_state(relevant=["doc"])) == "generate"


def test_grade_empty_rewrites_until_budget_then_abstains():
    assert after_grade(_state(relevant=[], attempts=0)) == "rewrite"
    assert after_grade(_state(relevant=[], attempts=settings.max_attempts)) == "abstain"


def test_verify_grounded_ends():
    assert after_verify(_state(grounded=True)) == END


def test_verify_not_grounded_retries_then_ends():
    assert after_verify(_state(grounded=False, attempts=0)) == "rewrite"
    assert after_verify(_state(grounded=False, attempts=settings.max_attempts)) == END


class _Doc:
    def __init__(self, id): self.id = id


def test_grade_after_verify_fail_stops_when_evidence_unchanged():
    # 이미 답이 있고(verify:fail 뒤) 재검색 결과가 같은 문서면 재생성하지 않는다
    s = _state(answer="이전 답", relevant=[_Doc("a"), _Doc("b")], seen_relevant=["a", "b"])
    assert after_grade(s) == "stop"


def test_grade_after_verify_fail_regenerates_on_new_evidence():
    s = _state(answer="이전 답", relevant=[_Doc("c")], seen_relevant=["a", "b"])
    assert after_grade(s) == "generate"


def test_grade_after_verify_fail_keeps_answer_when_rewrite_finds_nothing():
    s = _state(answer="이전 답", relevant=[], seen_relevant=["a"])
    assert after_grade(s) == "stop"
