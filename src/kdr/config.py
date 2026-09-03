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
    retrieval_mode: str = "hybrid"  # vector | bm25 | bm25_ws | hybrid
    top_k: int = 5
    max_attempts: int = 2

    data_dir: Path = ROOT / "data"
    chroma_dir: Path = ROOT / ".chroma"
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
