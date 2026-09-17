from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    llm_provider: str = "ollama"  # ollama | anthropic
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    anthropic_api_key: str = ""
    model: str = "claude-haiku-4-5"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    embed_model: str = "BAAI/bge-m3"
    retrieval_mode: str = "hybrid_rerank"  # vector | bm25 | bm25_ws | hybrid | hybrid_rerank
    top_k: int = 5
    max_attempts: int = 2

    # reranker (LangChain BaseDocumentCompressor, kdr/lc.py). hybrid_rerank 모드와 grade_mode=rerank 가 쓴다
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_threshold: float = 0.2  # grade_mode=rerank 에서 이 점수 이상이면 관련 문서로 본다
    grade_mode: str = "rerank"  # rerank: cross-encoder 점수 임계값 (LLM 호출 0) | llm: 문서마다 LLM yes/no

    # 문서 청크 (PDF·HWPX·OCR 본문). 글자 수 기준. KorQuAD는 문단 그대로라 해당 없음
    chunk_size: int = 900
    chunk_overlap: int = 100

    # OCR — 텍스트 레이어가 없는 PDF 페이지와 PNG/JPG 파일에만 적용 (easyocr, 선택 설치)
    ocr_enabled: bool = True
    ocr_langs: str = "ko,en"
    ocr_dpi: int = 200
    ocr_min_conf: float = 0.3  # 이 신뢰도 미만 상자는 버린다
    ocr_min_chars: int = 30  # 페이지 텍스트 레이어가 이보다 짧으면 스캔으로 보고 OCR
    ocr_gpu: bool = False  # CUDA 있을 때만 True. MPS는 easyocr 미지원

    data_dir: Path = ROOT / "data"
    chroma_dir: Path = ROOT / ".chroma"
    ocr_model_dir: Path = ROOT / ".easyocr"
    collection: str = "korquad"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    def _suffix(self, collection: str | None = None) -> str:
        c = collection or self.collection
        return "" if c == "korquad" else f".{c}"

    def chunks_path_for(self, collection: str | None = None) -> Path:
        return self.data_dir / f"chunks{self._suffix(collection)}.jsonl"

    def bm25_path_for(self, collection: str | None = None) -> Path:
        return self.data_dir / f"bm25{self._suffix(collection)}.pkl"

    @property
    def chunks_path(self) -> Path:
        return self.chunks_path_for()

    @property
    def questions_path(self) -> Path:
        return self.data_dir / "dev_questions.jsonl"

    @property
    def bm25_path(self) -> Path:
        return self.bm25_path_for()

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"


settings = Settings()
