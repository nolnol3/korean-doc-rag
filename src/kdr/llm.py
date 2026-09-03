"""LLM 호출 얇은 래퍼. provider 두 개, 인터페이스 하나.

  LLM_PROVIDER=ollama     로컬 (기본). 키 불필요, 비용 0. 본선 평가용
  LLM_PROVIDER=anthropic  Claude 직접. 키 필요. 비교용
  LLM_PROVIDER=openai     OpenAI 호환 엔드포인트 (게이트웨이·vLLM·LiteLLM 등). base_url + 키

모든 호출은 temperature 0. 그래프 노드는 provider를 모른다.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache

import httpx

from kdr.config import settings


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    log: list[str] = field(default_factory=list)
    timings: list[tuple[str, int, int, int]] = field(default_factory=list)  # (name, ms, in, out)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, name: str, in_tok: int, out_tok: int, ms: int = 0) -> None:
        with self._lock:  # grade가 문서별로 병렬 호출한다
            self.calls += 1
            self.input_tokens += in_tok
            self.output_tokens += out_tok
            self.log.append(name)
            self.timings.append((name, ms, in_tok, out_tok))

    def as_dict(self) -> dict:
        return {"calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


# ── ollama ────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _ollama() -> httpx.Client:
    return httpx.Client(base_url=settings.ollama_base_url, timeout=300)


def _ollama_chat(system: str, user: str, *, max_tokens: int, json_mode: bool) -> tuple[str, int, int]:
    body = {
        "model": settings.ollama_model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "think": False,  # qwen3 계열의 사고 모드 끔 — 판정·생성에 불필요하고 느림
        "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 8192},  # 문서 5개 프롬프트가 4096을 넘는다
    }
    if json_mode:
        body["format"] = "json"
    r = _ollama().post("/api/chat", json=body)
    r.raise_for_status()
    d = r.json()
    return d["message"]["content"], d.get("prompt_eval_count", 0), d.get("eval_count", 0)


# ── anthropic ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _anthropic():
    import anthropic

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example)")
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def _anthropic_chat(system: str, user: str, *, max_tokens: int, json_mode: bool) -> tuple[str, int, int]:
    msg = _anthropic().messages.create(
        model=settings.model,
        max_tokens=max_tokens,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    return text, msg.usage.input_tokens, msg.usage.output_tokens


# ── openai-compatible ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _openai() -> httpx.Client:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set (see .env.example)")
    return httpx.Client(
        base_url=settings.openai_base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {settings.openai_api_key}"},
        timeout=120,
    )


_OPENAI_SEM = threading.BoundedSemaphore(4)  # 게이트웨이는 동시 요청 수를 제한한다 (429). 4개까지만 동시에


def _openai_chat(system: str, user: str, *, max_tokens: int, json_mode: bool) -> tuple[str, int, int]:
    body = {
        "model": settings.openai_model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    # response_format은 게이트웨이마다 지원이 갈려서 안 보낸다. JSON은 프롬프트 + 관대한 파서로.
    with _OPENAI_SEM:
        d = _post_with_retry(_openai(), "/chat/completions", body)
    usage = d.get("usage") or {}
    return d["choices"][0]["message"]["content"], usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def _post_with_retry(client: httpx.Client, path: str, body: dict, tries: int = 6) -> dict:
    """429·5xx는 지수 백오프로 재시도. Retry-After 헤더가 있으면 그걸 따른다."""
    delay = 1.0
    for attempt in range(tries):
        r = client.post(path, json=body)
        if r.status_code < 400:
            return r.json()
        if r.status_code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
            r.raise_for_status()
        wait = float(r.headers.get("retry-after") or delay)
        time.sleep(min(wait, 30))
        delay = min(delay * 2, 30)
    raise RuntimeError("unreachable")


# ── 공용 ──────────────────────────────────────────────────────────────────────

def complete(system: str, user: str, *, name: str, usage: Usage | None = None,
             max_tokens: int = 1024, json_mode: bool = False) -> str:
    fn = {"ollama": _ollama_chat, "anthropic": _anthropic_chat, "openai": _openai_chat}[settings.llm_provider]
    t0 = time.perf_counter()
    text, in_tok, out_tok = fn(system, user, max_tokens=max_tokens, json_mode=json_mode)
    if usage is not None:
        usage.add(name, in_tok, out_tok, int((time.perf_counter() - t0) * 1000))
    return text.strip()


def complete_json(system: str, user: str, *, name: str, usage: Usage | None = None, max_tokens: int = 256) -> dict:
    """JSON 한 덩어리를 기대. 코드펜스·앞뒤 잡음은 걷어낸다. 7B 모델도 통과하도록 관대하게."""
    raw = complete(system, user, name=name, usage=usage, max_tokens=max_tokens, json_mode=True)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # 잘리거나 깨진 JSON — 필요한 키만 정규식으로 건진다 (모델이 reason을 길게 써서 잘리는 경우)
    out: dict = {}
    for key in ("grounded", "relevant"):
        b = re.search(rf'"{key}"\s*:\s*(true|false)', raw, re.I)
        if b:
            out[key] = b.group(1).lower() == "true"
    q = re.search(r'"query"\s*:\s*"([^"]+)"', raw)
    if q:
        out["query"] = q.group(1)
    if out:
        return out
    raise ValueError(f"{name}: no JSON in response: {raw[:200]!r}")


def model_label() -> str:
    return {"ollama": settings.ollama_model, "anthropic": settings.model,
            "openai": settings.openai_model.split("/")[-1]}[settings.llm_provider]
