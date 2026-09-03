"""KorQuAD 1.0 원본을 data/raw/에 내려받는다.

라이선스: CC BY-ND 2.0 KR — 재배포·수정 금지. 그래서 레포에 넣지 않고 매번 원본에서 받는다.
출처: https://github.com/korquad/korquad.github.io/tree/master/dataset
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/korquad/korquad.github.io/master/dataset/"
FILES = {
    "KorQuAD_v1.0_train.json": 38_527_475,
    "KorQuAD_v1.0_dev.json": 3_881_058,
    "evaluate-v1.0.py": 4_305,  # 공식 EM/F1 스크립트 — 답변 정규화 규칙을 그대로 쓰기 위해
}
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch(name: str, expected_size: int) -> Path:
    dst = RAW / name
    if dst.exists() and dst.stat().st_size == expected_size:
        print(f"skip  {name} (already present)")
        return dst
    print(f"fetch {name} ...", end=" ", flush=True)
    urllib.request.urlretrieve(BASE + name, dst)
    size = dst.stat().st_size
    digest = hashlib.sha256(dst.read_bytes()).hexdigest()[:12]
    print(f"{size:,} bytes  sha256:{digest}")
    if size != expected_size:
        print(f"  warning: expected {expected_size:,} bytes", file=sys.stderr)
    return dst


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for name, size in FILES.items():
        fetch(name, size)
