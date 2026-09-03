# 구조

```
[색인]     문서 ─▶ 문단 청크 ─▶ bge-m3 임베딩 ─▶ Chroma
                          └─▶ kiwi 형태소 분석 ─▶ BM25        두 결과를 RRF로 합친다 (hybrid)

[질의]     POST /ask {q}
              │
   ┌──────────┴──────────── graph.py (LangGraph) ───────────────────────────┐
   │  retrieve ─▶ grade ─┬─ 관련 문서 있음 ─▶ generate ─▶ verify ─┬─ 근거 확인 ─▶ END
   │     ▲               │                                        │
   │     │               └─ 관련 문서 없음 ─▶ rewrite ──┐          └─ 근거 없음 ─┐
   │     └───────────────────────────────────────────────┴────────────────────┘
   │                 재시도는 2회까지. 재검색 결과가 이전과 같으면 멈추고, 재시도를 다 쓰면 "근거를 찾지 못했다"고 답한다
   └──────────────────────────────────────────────────────────────────────┘
              │
        {answer, citations[], path[], grounded, attempts}
```

## 노드

| 노드 | 역할 | LLM 호출 |
|---|---|---|
| `retrieve` | hybrid 검색으로 상위 5개 문단을 가져온다 | 0 |
| `grade` | 문단마다 "질문의 답이 이 안에 있는가"를 예/아니오로 판정한다. 5개를 병렬로 호출한다 | 5 |
| `rewrite` | 관련 문단이 하나도 없을 때 검색용 질의를 다시 쓴다. 이미 시도한 질의 목록을 함께 주어 같은 질의가 반복되지 않게 한다 | 1 |
| `generate` | 관련 문단만 넣고 답을 만든다. 답은 명사구로 짧게 쓰고 근거 문단 번호를 `[n]` 형식으로 붙이게 한다 | 1 |
| `verify` | 답의 핵심 사실이 인용한 문단으로 뒷받침되는지 확인한다 | 1 |

분기는 두 곳에 있다.

- `grade` 다음: 관련 문단이 있으면 `generate`로 간다. 없으면 남은 재시도가 있을 때 `rewrite`로 가고, 없으면 "문서에서 근거를 찾지 못했습니다"로 끝낸다.
- `verify` 다음: 근거가 확인되면 끝낸다. 확인되지 않으면 `rewrite`로 간다. 단, 재검색해서 가져온 문단이 직전과 같으면 같은 답이 다시 나올 것이므로 그 자리에서 멈춘다(`stop:no-new-evidence`).

LLM 호출 수는 가장 적을 때 7회(grade 5 + generate + verify), 가장 많을 때 15회 정도다.

## 비교 대상

| arm | 구성 | 확인하려는 것 |
|---|---|---|
| `none` | 검색 없이 LLM만 | 모델이 문서 없이 얼마나 답하는가 |
| `naive` | `retrieve → generate` | 검색만 붙였을 때의 효과 |
| `graph` | 위의 그래프 | grade·rewrite·verify가 더해 주는 효과 |

세 구성은 같은 검색기와 같은 답변 프롬프트를 쓴다. 차이는 그래프 노드가 있고 없고뿐이므로 결과 차이를 그 노드들의 효과로 볼 수 있다.

## 검색 계층

| 모드 | 구현 |
|---|---|
| `vector` | bge-m3(1024차원, 정규화)로 임베딩해 Chroma에서 코사인 유사도로 찾는다 |
| `bm25` | kiwipiepy로 형태소 분석을 하고 명사·동사·수식언·숫자·영문만 남겨 rank_bm25에 넣는다 |
| `bm25_ws` | 공백으로만 자른 BM25. 형태소 분석이 없는 검색 플랫폼에서 한국어가 어떻게 되는지 보기 위한 비교용이다 |
| `hybrid` | vector와 bm25에서 각각 상위 20개를 가져와 RRF(k=60)로 합치고 상위 5개를 쓴다. 기본값이다 |

## LLM 연결

| `LLM_PROVIDER` | 대상 | 비고 |
|---|---|---|
| `ollama` (기본) | 로컬 Ollama. Qwen3 8B | 키가 필요 없다. 컨텍스트 8192, 사고 모드는 끈다 |
| `anthropic` | Claude | Anthropic SDK를 직접 쓴다 |
| `openai` | OpenAI 호환 엔드포인트(vLLM, LiteLLM, 게이트웨이 등) | 동시 요청을 4개로 제한하고 429·5xx는 지수 백오프로 재시도한다 |

모든 호출은 temperature 0이다. grade·verify·rewrite는 JSON으로 답하게 하되, 7B 모델은 형식을 자주 어기므로 잘리거나 깨진 JSON에서도 필요한 키만 정규식으로 건지는 파서를 둔다.

## API

| 엔드포인트 | 설명 |
|---|---|
| `POST /ask` | `{q, mode?, k?, collection?}`를 받아 `{answer, citations[], path[], grounded, attempts, usage, latency_ms, mode, llm, collection}`을 돌려준다. `mode`는 `graph`, `naive`, `none` 중 하나다 |
| `POST /upload` | multipart로 PDF·HWPX 파일(`files[]`)과 `collection`(기본 `docs`)을 받아 파싱하고 색인에 추가한다. 이미 있는 청크는 다시 넣지 않는다 |
| `GET /collections` | 검색할 수 있는 컬렉션과 각각의 청크 수 |
| `GET /health` | 색인 크기, 모델, provider, 검색 모드 |
| `GET /` | 화면 한 장(`static/index.html`). `?q=&mode=&collection=`을 주면 열리면서 바로 질문한다 |

컬렉션은 색인의 단위다. `korquad`는 평가용이고 업로드한 문서는 `docs`나 새 이름의 컬렉션으로 들어간다. 서버는 컬렉션별로 청크·BM25·Chroma 핸들을 캐시하고 업로드 뒤에 비운다.

## 설계 결정

| 결정 | 이유 |
|---|---|
| 청크를 KorQuAD 문단 그대로 쓴다 | 정답 문단의 id가 있으므로 recall을 정확히 잴 수 있다. 고정 길이로 자르는 방식은 다음 실험 후보다 |
| 임베딩은 bge-m3를 로컬에서 돌린다 | 한국어 성능이 좋고 키가 필요 없으며 결과가 결정적이다 |
| Chroma(embedded) | 서버 없이 파일만으로 재현할 수 있다. 운영 환경이라면 pgvector가 맞다 |
| grade를 문단별로 나눠 병렬 호출한다 | 5개를 한 프롬프트에 넣는 것보다 토큰이 1/4이고 7B 모델의 판정이 더 정확하다. 대신 요청 수가 5배가 되므로 요청 수에 제한이 있는 게이트웨이에서는 한 번에 묶어 보내는 편이 낫다 |
| verify를 넣는다 | 사내 문서 QA에서는 답에 근거가 있는지가 핵심 요구사항이다. 7B 모델로는 이 판정이 되지 않는다는 점도 실험으로 확인했다([evaluation.md](evaluation.md)) |
| `langgraph` 라이브러리만 쓰고 서버는 FastAPI로 직접 만든다 | `langgraph`는 MIT지만 서버 런타임인 `langgraph-api`는 Elastic License여서 운영 배포에 상용 라이선스가 필요하다 |
| Anthropic SDK와 httpx를 직접 쓴다 | LangChain 래퍼를 거치지 않으면 실제로 보내는 프롬프트가 코드에 그대로 보인다 |
| temperature 0, seed 고정, 문항별 결과를 jsonl로 남긴다 | 재현과 사후 분석을 위해서다 |

## 운영 환경으로 가져갈 때 바꿀 것

| 지금 | 운영 |
|---|---|
| Chroma | pgvector(기존 Postgres를 쓸 수 있다) 또는 OpenSearch(BM25와 kNN을 함께 제공) |
| rank_bm25(메모리) | OpenSearch nori 분석기 |
| bge-m3를 프로세스 안에서 실행 | TEI나 vLLM의 `/v1/embeddings`로 분리해 서빙 |
| 권한 처리 없음 | 문서별 접근 권한을 청크 메타데이터에 넣고 검색 단계에서 필터링 |
| verify를 7B 모델이 수행 | 더 큰 모델에 맡기거나, 인용 문단에 답 문자열이 있는지 확인하는 규칙으로 대체 |
| `path[]` 로그 | Langfuse나 OpenTelemetry로 노드별 지연과 토큰을 추적 |
| 평가 1회 실행 | 골든셋을 두고 CI에서 지속적으로 평가 |
