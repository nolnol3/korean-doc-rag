"""BM25용 토크나이저 두 개.

- tokenize_kiwi: 형태소 분석. 조사·어미·기호를 떼고 내용어만 남긴다 → "삼성전자의" → ["삼성전자"]
- tokenize_ws:   공백 분리. 형태소 분석이 없는 외산 플랫폼의 한국어 BM25가 하는 일. ablation용.
"""
from __future__ import annotations

import re
from functools import lru_cache

_KEEP_PREFIX = ("N", "V", "M", "X")  # 명사·동사/형용사·수식언·접사
_KEEP_EXACT = {"SL", "SN", "SH"}  # 영문·숫자·한자


@lru_cache(maxsize=1)
def _kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


def tokenize_kiwi(text: str) -> list[str]:
    return [
        t.form.lower()
        for t in _kiwi().tokenize(text)
        if t.tag.startswith(_KEEP_PREFIX) or t.tag in _KEEP_EXACT
    ]


def tokenize_ws(text: str) -> list[str]:
    return re.findall(r"\S+", text.lower())
