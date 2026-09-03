"""평가.

  --ablation   검색 4종 × recall@k. LLM 호출 없음. 1,000문항
  (기본)       naive vs graph, 200문항. 답변 EM/F1은 KorQuAD 공식 정규화 규칙

샘플은 seed 고정. 200문항은 1,000문항의 앞 200개라 두 평가의 질문이 겹친다.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import random
import re
import string
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from kdr.config import settings
from kdr.llm import model_label
from kdr.retriever import retrieve

RESULTS = settings.data_dir.parent / "results"


def model_dir() -> Path:
    """모델별 결과 디렉토리 — results/qwen3-8b/, results/claude-haiku-4-5-20251001/ ..."""
    d = RESULTS / model_label().replace(":", "-")
    d.mkdir(parents=True, exist_ok=True)
    return d
SEED = 20260903
MODES = ["vector", "bm25_ws", "bm25", "hybrid"]
KS = (1, 3, 5, 10)


def load_questions(n: int) -> list[dict]:
    with settings.questions_path.open() as f:
        qs = [json.loads(line) for line in f]
    random.Random(SEED).shuffle(qs)
    return qs[:n]


# ── KorQuAD 공식 정규화 (evaluate-v1.0.py와 동일) ─────────────────────────────

def normalize_answer(s: str) -> str:
    s = re.sub(r"'", " ", s)
    s = re.sub(r'"', " ", s)
    s = re.sub(r"《", " ", s)
    s = re.sub(r"》", " ", s)
    s = re.sub(r"<", " ", s)
    s = re.sub(r">", " ", s)
    s = re.sub(r"〈", " ", s)
    s = re.sub(r"〉", " ", s)
    s = re.sub(r"\(", " ", s)
    s = re.sub(r"\)", " ", s)
    s = re.sub(r"'", " ", s)
    s = re.sub(r"'", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s.lower())
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p = list("".join(normalize_answer(pred).split()))  # 공식 스크립트: 문자 단위
    g = list("".join(normalize_answer(gold).split()))
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def em_score(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


_CITE = re.compile(r"\s*\[\d+\]")
_TRAIL_DIGIT = re.compile(r"\s+[1-9]$")  # "올해의 가수상 4" — 괄호를 빼먹은 인용 번호


def clean_answer(pred: str) -> str:
    """채점 전에 우리 포맷([1] 인용 표시)만 뗀다. 답변 내용은 건드리지 않는다."""
    pred = _CITE.sub("", pred).strip()
    if len(pred) > 2:
        pred = _TRAIL_DIGIT.sub("", pred)
    return pred.strip()


def best_over_golds(fn, pred: str, golds: list[str]) -> float:
    pred = clean_answer(pred)
    return max(fn(pred, g) for g in golds)


def rescore(d: Path) -> None:
    """저장된 답변으로 em/f1/abstain만 다시 계산 — LLM 호출 없음."""
    for p in sorted(d.glob("*.jsonl")):
        rows = [json.loads(l) for l in p.open()]
        for r in rows:
            r["em"] = best_over_golds(em_score, r["answer"], r["gold"])
            r["f1"] = best_over_golds(f1_score, r["answer"], r["gold"])
            r["abstain"] = r["answer"].startswith(("문서에서 근거를 찾지 못했습니다", "모르겠습니다"))
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"rescored {p.name}: {len(rows)} rows")


# ── ablation ──────────────────────────────────────────────────────────────────

def ablation(n: int = 1000) -> dict:
    qs = load_questions(n)
    recall = {m: {k: 0 for k in KS} for m in MODES}
    for q in tqdm(qs, desc="ablation"):
        for m in MODES:
            ids = [h.id for h in retrieve(q["question"], k=max(KS), mode=m)]
            for k in KS:
                recall[m][k] += q["gold_chunk"] in ids[:k]
    out = {m: {f"recall@{k}": round(v / n, 4) for k, v in d.items()} for m, d in recall.items()}
    out["_meta"] = {"n": n, "seed": SEED, "embed_model": settings.embed_model}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "retrieval_ablation.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n{'mode':10s}" + "".join(f"{'R@'+str(k):>8s}" for k in KS))
    for m in MODES:
        print(f"{m:10s}" + "".join(f"{out[m][f'recall@{k}']:8.3f}" for k in KS))
    return out


# ── naive vs graph ────────────────────────────────────────────────────────────

def run_arm(arm: str, n: int, workers: int = 1) -> list[dict]:
    if arm == "naive":
        from kdr.baseline import ask
    elif arm == "none":
        from kdr.baseline import ask_no_retrieval as ask
    else:
        from kdr.graph import ask
    qs = load_questions(n)
    rows = []
    path = model_dir() / f"{arm}.jsonl"

    def one(q: dict) -> dict:
        t0 = time.perf_counter()
        try:
            r = ask(q["question"])
        except Exception as e:  # 한 문항의 실패가 200문항을 죽이지 않게. 오류는 행에 남긴다
            from kdr.baseline import Result
            r = Result(q["question"], "", [], ["error"], None, 0, {"calls": 0, "input_tokens": 0, "output_tokens": 0}, 0)
            err = f"{type(e).__name__}: {str(e)[:160]}"
        else:
            err = None
        return {
            "error": err,
            "id": q["id"],
            "question": q["question"],
            "gold": q["answers"],
            "gold_chunk": q["gold_chunk"],
            "answer": r.answer,
            "em": best_over_golds(em_score, r.answer, q["answers"]),
            "f1": best_over_golds(f1_score, r.answer, q["answers"]),
            "abstain": r.answer.startswith(("문서에서 근거를 찾지 못했습니다", "모르겠습니다")),
            "gold_retrieved": any(c["id"] == q["gold_chunk"] for c in r.citations),
            "grounded": r.grounded,
            "attempts": r.attempts,
            "path": r.path,
            "usage": r.usage,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }

    retrieve("워밍업")  # 임베딩 모델·Chroma·BM25를 스레드 풀 전에 한 번 올린다 (lru_cache는 동시 첫 호출을 막지 못함)
    with path.open("w") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        for row in tqdm(ex.map(one, qs), total=len(qs), desc=f"{arm} x{workers}"):
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    return rows


def summarize(arms: list[str], d: Path | None = None) -> None:
    d = d or model_dir()
    lines = [f"모델: `{d.name}` · n=200 · seed {SEED}", "",
             "| arm | n | EM | F1 | abstain | gold in ctx | grounded | calls | p50 ms |", "|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        p = d / f"{arm}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.open()]
        n = len(rows)
        mean = lambda key: sum(r[key] for r in rows) / n
        grounded = [r["grounded"] for r in rows if r["grounded"] is not None]
        g = f"{sum(grounded)/len(grounded):.3f}" if grounded else "—"
        lat = sorted(r["latency_ms"] for r in rows)[n // 2]
        lines.append(
            f"| {arm} | {n} | {mean('em'):.3f} | {mean('f1'):.3f} | {mean('abstain'):.3f} | "
            f"{mean('gold_retrieved'):.3f} | {g} | {sum(r['usage']['calls'] for r in rows)/n:.1f} | {lat} |"
        )
    (d / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--arm", choices=["none", "naive", "graph"], action="append")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=1, help="동시 문항 수. API provider는 8 권장, Ollama는 OLLAMA_NUM_PARALLEL 이하")
    ap.add_argument("--summarize", metavar="DIR", help="결과 디렉토리만 다시 집계")
    ap.add_argument("--rescore", metavar="DIR", help="저장된 답변으로 점수만 재계산")
    a = ap.parse_args()
    if a.ablation:
        ablation(a.n if a.n != 200 else 1000)
    elif a.summarize:
        summarize(["none", "naive", "graph"], Path(a.summarize))
    elif a.rescore:
        rescore(Path(a.rescore))
        summarize(["none", "naive", "graph"], Path(a.rescore))
    else:
        for arm in a.arm or ["none", "naive", "graph"]:
            run_arm(arm, a.n, a.workers)
        summarize(["none", "naive", "graph"])
