"""문서 본문 청킹(_pack) — 문장 경계·상한·overlap·짧은 문단 합치기. LLM·인덱스 불필요."""
from kdr.config import settings
from kdr.ingest_docs import _pack

SENT = "가나다라마바사아자차카타파하 문장이다. "


def test_empty():
    assert _pack([]) == [] and _pack(["", "  "]) == []


def test_short_paragraphs_are_merged():
    out = _pack(["첫 문단이다."] * 5)
    assert len(out) == 1 and out[0].count("첫 문단이다.") == 5


def test_long_paragraph_splits_at_sentence_boundary_within_limit():
    out = _pack([(SENT * 100).strip()])  # ~2,100자
    assert len(out) >= 3
    for c in out:
        assert len(c) <= settings.chunk_size
        assert c.endswith("문장이다.")  # 단어 중간이 아니라 문장 끝에서 잘렸다
        assert not c.startswith(".")


def test_chunks_overlap():
    out = _pack([(SENT * 100).strip()])
    total = sum(map(len, out))
    original = len(SENT * 100)
    assert total > original  # 겹친 만큼 길어진다
    # 앞 청크의 마지막 문장이 다음 청크 앞에 다시 나온다
    tail = out[0][-len(SENT.strip()):]
    assert tail in out[1][: settings.chunk_overlap + len(SENT)]


def test_paragraph_breaks_preferred_over_sentence_breaks():
    a, b = "앞 문단 " * 110, "뒤 문단 " * 110  # 각 550자, 합치면 상한 초과
    out = _pack([a.strip(), b.strip()])
    assert len(out) == 2 and out[0].startswith("앞") and out[1].startswith("뒤")
