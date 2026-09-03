# 구조

```
[오프라인]  문서 ─▶ 문단 청크 ─▶ bge-m3 임베딩 ─▶ Chroma
                          └─▶ kiwi 형태소 ─▶ BM25         두 결과를 RRF로 합침 (hybrid)

[온라인]    POST /ask {q}
              │
   ┌──────────┴──────────── graph.py (LangGraph) ───────────────────────────┐
   │  retrieve ─▶ grade ─┬─ 관련 있음 ─▶ generate ─▶ verify ─┬─ 근거 있음 ─▶ END
   │     ▲               │                                  │
   │     │               └─ 관련 없음 ─▶ rewrite ──┐         └─ 근거 없음 ─┐
   │     └─────────────────────────────────────────┴──────────────────────┘
   │                          재시도 ≤ 2 · 같은 근거면 정지 · 소진 시 "근거 못 찾음"
   └──────────────────────────────────────────────────────────────────────┘
              │
        {answer, citations[], path[], grounded, attempts}
```

## 노드

| 노드 | 하는 일 | LLM 호출 |
|---|---|---|
| `retrieve` | hybrid 검색 top-5 (벡터 + 형태소 BM25, RRF) | 0 |
| `grade` | 문서마다 "질문의 답이 이 안에 있나" yes/no — 병렬 | 5 |
| `rewrite` | 관련 문서가 없으면 질의를 검색용으로 재작성. 이전 질의 목록을 보여줘 반복을 막는다 | 1 |
| `generate` | 관련 문서만 넣고 답 생성. `[n]` 인용 강제, 정답만 명사구로 | 1 |
| `verify` | 답의 핵심 사실이 인용 문서로 뒷받침되는가 | 1 |

분기 두 곳:
- `grade` 뒤 — 관련 문서 있음 → `generate` / 없음 → `rewrite`(남은 재시도가 있으면) / 소진 → "문서에서 근거를 찾지 못했습니다"
- `verify` 뒤 — 근거 있음 → 끝 / 없음 → `rewrite`. 단 재검색 결과가 **직전과 같은 문서**면 같은 답이 나오므로 그 자리에서 멈춘다 (`stop:no-new-evidence`)

호출 수: 최선 7 (grade 5 + generate + verify), 최악 약 15.

## 대조군

| arm | 구성 | 재는 것 |
|---|---|---|
| `none` | 검색 없이 LLM만 | "그냥 호출"과의 차이 |
| `naive` | `retrieve → generate` | 단순 RAG |
| `graph` | 위 그래프 | grade·rewrite·verify의 몫 |

세 arm이 **같은 retriever, 같은 generate 프롬프트**를 쓴다. 차이는 그래프 노드의 유무뿐이라 비교가 깨끗하다.

## 검색 계층

| 모드 | 구현 |
|---|---|
| `vector` | bge-m3 (1024d, 정규화) → Chroma cosine |
| `bm25` | kiwipiepy 형태소 → 내용어만(명사·동사·수식언·숫자·영문) → rank_bm25 |
| `bm25_ws` | 공백 분리 BM25 — 형태소 분석이 없는 플랫폼의 한국어 상태를 재현하는 ablation용 |
| `hybrid` | vector·bm25 각 top-20을 RRF(k=60)로 합쳐 top-5 — 기본값 |

## LLM provider

| `LLM_PROVIDER` | 대상 | 비고 |
|---|---|---|
| `ollama` (기본) | 로컬. Qwen3 8B | 키 불필요. `num_ctx` 8192, think 끔 |
| `anthropic` | Claude 직접 | Anthropic SDK |
| `openai` | OpenAI 호환 엔드포인트 (vLLM · LiteLLM · 게이트웨이) | 동시 요청 4개 제한, 429·5xx 지수 백오프 |

모든 호출 temperature 0. grade·verify·rewrite는 JSON을 요구하되, 잘리거나 깨진 JSON에서도 필요한 키만 정규식으로 건지는 관대한 파서를 쓴다 (7B가 지시를 어기는 경우가 잦다).

## 설계 결정

| 결정 | 이유 |
|---|---|
| 청크 = KorQuAD 문단 그대로 | 정답 문단 id로 recall을 정확히 잰다. 고정 길이 청킹은 ablation 후보 |
| bge-m3 로컬 임베딩 | 한국어 상위권, 키 불필요, 결정적 |
| Chroma (embedded) | 서버 없이 파일로 재현. 운영이면 pgvector |
| grade를 문서별 병렬로 | 5개를 한 프롬프트에 넣는 것보다 토큰 1/4, 7B에 더 정확. 대신 요청 수 5배 — 요청 수 제한이 있는 게이트웨이에선 batch가 맞다 |
| verify 포함 | 사내문서 QA의 핵심 요구. 7B로는 무의미하다는 것도 결과 ([evaluation.md](evaluation.md)) |
| `langgraph` 라이브러리만, 서버는 자체 FastAPI | `langgraph`는 MIT지만 `langgraph-api` 서버 런타임은 Elastic License라 프로덕션에 상용 키가 필요 |
| Anthropic SDK / httpx 직접 | LangChain 래퍼 없이 — 프롬프트가 그대로 보인다 |
| temperature 0, seed 고정, 문항별 결과 jsonl | 재현과 사후 분석 |

## 프로덕션으로 가려면

| 지금 | 운영 |
|---|---|
| Chroma | pgvector (기존 Postgres) 또는 OpenSearch(BM25+kNN 내장) |
| rank_bm25 메모리 | OpenSearch nori |
| bge-m3 in-process | TEI 또는 vLLM `/v1/embeddings`로 분리 서빙 |
| 권한 없음 | 문서별 ACL을 청크 메타데이터에 넣고 **검색 시** 필터 |
| verify (7B) | 강한 모델 또는 규칙(인용 문단에 답 문자열 포함 여부) |
| path[] 로그 | Langfuse / OpenTelemetry로 노드별 지연·토큰 |
| 1회 평가 | 골든셋 + 지속 평가를 CI에 |
