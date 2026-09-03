# PDF · HWPX 문서 넣기

세 가지 방법. 모두 KorQuAD와 분리된 컬렉션(기본 `docs`)에 들어간다.

| 방법 | 언제 |
|---|---|
| UI "문서 넣기"에 끌어다 놓기 | 몇 개 넣고 바로 물어볼 때 |
| `POST /upload` (multipart `files[]`, `collection`) | 다른 시스템에서 붙일 때. 기존 청크는 유지, 같은 내용은 중복 안 됨 |
| `COLLECTION=docs python -m kdr.ingest_docs ./dir/` | 대량. 그 컬렉션을 처음부터 다시 만든다(지운 문서가 남지 않게) |

질문할 때 `collection`을 지정한다 — UI의 컬렉션 선택, API의 `{"collection": "docs"}`.

## 무엇을 하나

| 형식 | 처리 |
|---|---|
| PDF | 페이지마다 본문 블록을 읽고 400~900자로 묶는다. 괘선 있는 표는 찾아서 **markdown 표 하나를 청크 하나로** 넣고 본문에서는 뺀다 |
| HWPX | `Contents/section*.xml`의 문단(`hp:p`)을 읽는다. 표(`hp:tbl`)는 markdown으로 |

청크마다 `source`(파일명) · `page`(PDF 쪽 / HWPX 절) · `kind`(text / table) 메타데이터가 붙고, 인용 제목은 `파일명 p.N`으로 API 응답에 그대로 나온다.

```
Q: 5년 이상 근속하면 연차가 며칠인가?
A: 17일 [2]
   [2] sample_policy.hwpx §1  [표] | 근속연수 | 연차일수 | ... | 5년 이상 | 17일 |
   path: retrieve → grade:2/4 → generate → verify:ok
```

## 확인한 범위

표 있는 PDF·HWPX 샘플(`tests/fixtures/`)과 위키백과 PDF 3건에서 11문항 중 10 정답, 답이 없는 1문항은 기권 (Qwen3 8B, 그래프).

## 한계

- **괘선 없는 표**(위키 인포박스 같은)는 표로 잡히지 않고 텍스트로 들어간다. 답은 나오지만 구조는 잃는다. → Docling / pymupdf_layout
- PDF 줄바꿈이 단어 중간에 남을 수 있다("증 권사이다"). 형태소 분석과 임베딩이 어느 정도 흡수한다.
- **HWP 5.0**(구형 바이너리)은 다루지 않는다. HWPX만.
- 스캔 PDF(이미지)는 OCR이 없어 빈 청크가 된다.
- 다단 레이아웃은 블록 순서가 섞일 수 있다.
