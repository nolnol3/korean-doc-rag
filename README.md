# korean-doc-rag

한국어 문서를 대상으로 질문에 답하고, 답의 근거가 된 문단을 함께 보여주는 RAG 서비스입니다.

```
Q: 5년 이상 근속하면 연차가 며칠인가?
A: 17일 [2]
   [2] sample_policy.hwpx §1  [표] | 근속연수 | 연차일수 | … | 5년 이상 | 17일 |
   path: retrieve → grade:2/4 → generate → verify:ok
```

- 검색은 벡터 검색(bge-m3)과 형태소 기반 BM25(kiwi)를 함께 씁니다. 한국어는 조사가 붙기 때문에 형태소 분석 없이 BM25를 쓰면 recall이 16%p 떨어집니다.
- 답변 흐름은 LangGraph로 구성했습니다. 검색 → 관련성 판정 → (관련 문서가 없으면 질의를 고쳐 재검색) → 답변 생성 → 근거 검증 순서입니다.
- 로컬 모델(Ollama)로 동작하므로 API 키가 없어도 실행할 수 있습니다. Claude나 OpenAI 호환 엔드포인트도 설정으로 붙일 수 있습니다.
- PDF와 HWPX 파일을 올려서 질문할 수 있습니다. 문서 안의 표는 행과 열 구조를 유지한 채로 색인되므로 표 안의 값을 정확히 찾아 답합니다.

| KorQuAD 200문항, Qwen3 8B | EM | F1 |
|---|---|---|
| 검색 없이 LLM만 | 0.045 | 0.263 |
| 단순 RAG | 0.720 | 0.840 |
| 그래프 RAG | 0.725 | 0.860 |

어떤 구성이 얼마나 기여했는지, 무엇이 기대와 달랐는지는 [docs/evaluation.md](docs/evaluation.md)에 정리했습니다.

## 시작하기

```bash
git clone https://github.com/nolnol3/korean-doc-rag && cd korean-doc-rag
make setup                      # uv venv + 의존성 설치
ollama pull qwen3:8b            # 로컬 LLM
make fetch && make index        # KorQuAD 문단 10,639개 색인 (Apple Silicon 기준 약 10분)
make serve                      # http://localhost:8000
```

브라우저에서 `http://localhost:8000`을 열면 질문 화면이 나옵니다. API 문서는 `/docs`에 있습니다.

![korean-doc-rag UI](assets/ui.png)

## 사용법

**API**

```bash
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"q": "구룡폭포의 높이는?"}'
```

```json
{
  "answer": "74미터 [1]",
  "citations": [{"n": 1, "title": "금강산", "text": "내금강의 동쪽에 있으며, 동해안을 따라 펼쳐진 지역을 포괄한다. 크게 구…"}],
  "path": ["retrieve", "grade:1/5", "generate", "verify:ok"],
  "grounded": true
}
```

`mode`에 `naive`(단순 RAG)나 `none`(검색 없이 LLM만)을 주면 같은 질문을 다른 방식으로 처리한 결과를 비교해 볼 수 있습니다.

**문서 올리기 (PDF, HWPX)**

화면 오른쪽 "문서 넣기" 영역에 파일을 끌어다 놓으면 파싱과 임베딩이 끝난 뒤 바로 질문할 수 있습니다. API로는 다음과 같이 합니다.

```bash
curl -F files=@report.pdf -F files=@policy.hwpx -F collection=docs localhost:8000/upload
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"q": "5년 이상 근속하면 연차가 며칠인가?", "collection": "docs"}'
```

파일이 많으면 CLI가 편합니다: `COLLECTION=docs python -m kdr.ingest_docs ./my_docs/`. 자세한 내용은 [docs/documents.md](docs/documents.md)를 참고하세요.

**다른 LLM 사용**

`.env`에서 바꿉니다.

```
LLM_PROVIDER=ollama          # ollama | anthropic | openai (vLLM, LiteLLM 등 OpenAI 호환 엔드포인트)
OLLAMA_MODEL=qwen3:8b
RETRIEVAL_MODE=hybrid        # hybrid | vector | bm25 | bm25_ws
```

Docker로 띄우려면 `docker compose up`을 실행합니다. LLM은 호스트에서 실행 중인 Ollama를 사용합니다.

## 평가 재현

```bash
make ablation    # 검색 방식 4종의 recall@k 비교 (LLM 호출 없음)
make eval        # LLM만 / 단순 RAG / 그래프 RAG, 각 200문항
```

## 더 읽기

| 문서 | 내용 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 그래프의 노드와 분기, 검색 계층, API, 설계 결정, 운영 환경으로 가져갈 때 바꿀 것 |
| [docs/evaluation.md](docs/evaluation.md) | 결과표, 실험에서 확인한 것 네 가지, 실패 사례, 한계 |
| [docs/documents.md](docs/documents.md) | PDF·HWPX 처리 방식과 한계 |
| [results/](results/) | 집계표, 검색 ablation, 실패 분석. 문항별 jsonl은 KorQuAD 원문을 포함하므로 배포하지 않으며 `make eval`로 다시 만들 수 있습니다 |

## 라이선스

코드는 MIT입니다. KorQuAD 1.0은 CC BY-ND 2.0 KR이므로 저장소에 포함하지 않고 `make fetch`로 원본에서 내려받습니다.
