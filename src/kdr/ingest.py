"""KorQuAD JSON → 문단 청크 → (1) bge-m3 임베딩 → Chroma, (2) 형태소 BM25 / 공백 BM25.

청크 = KorQuAD 문단 그대로. 정답 문단 id로 recall@k를 정확히 재기 위해서다.
실행: python -m kdr.ingest
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from dataclasses import asdict, dataclass

import chromadb
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from kdr.config import settings
from kdr.tokenize import tokenize_kiwi, tokenize_ws


@dataclass(frozen=True)
class Chunk:
    id: str
    title: str
    text: str
    meta: dict | None = None  # 문서 인제스트용: source, page, kind


def _chunk_id(title: str, text: str) -> str:
    return hashlib.sha1(f"{title}\n{text}".encode()).hexdigest()[:16]


def load_korquad() -> tuple[list[Chunk], list[dict]]:
    """train+dev 문단을 코퍼스로, dev 질문만 평가용으로 뽑는다."""
    seen: dict[str, Chunk] = {}
    questions: list[dict] = []
    for split in ("train", "dev"):
        path = settings.raw_dir / f"KorQuAD_v1.0_{split}.json"
        if not path.exists():
            sys.exit(f"missing {path} — run: python scripts/fetch_data.py")
        for article in json.loads(path.read_text())["data"]:
            title = article["title"]
            for para in article["paragraphs"]:
                text = para["context"].strip()
                cid = _chunk_id(title, text)
                seen.setdefault(cid, Chunk(cid, title, text))
                if split == "dev":
                    for qa in para["qas"]:
                        questions.append(
                            {
                                "id": qa["id"],
                                "question": qa["question"],
                                "answers": [a["text"] for a in qa["answers"]],
                                "gold_chunk": cid,
                                "title": title,
                            }
                        )
    return list(seen.values()), questions


def write_jsonl(path, rows) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_bm25(chunks: list[Chunk], collection: str | None = None) -> None:
    """kiwi 형태소 BM25(본선)와 공백 분리 BM25(ablation)를 둘 다 만든다."""
    kiwi_corpus = [tokenize_kiwi(c.text) for c in tqdm(chunks, desc="kiwi tokenize", disable=len(chunks) < 50)]
    ws_corpus = [tokenize_ws(c.text) for c in chunks]
    payload = {
        "ids": [c.id for c in chunks],
        "kiwi": BM25Okapi(kiwi_corpus),
        "ws": BM25Okapi(ws_corpus),
    }
    with settings.bm25_path_for(collection).open("wb") as f:
        pickle.dump(payload, f)


def build_vector(chunks: list[Chunk], rebuild: bool = False, collection: str | None = None) -> None:
    from sentence_transformers import SentenceTransformer

    name = collection or settings.collection
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    if rebuild:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    # 이미 있는 id는 건너뛰고 없는 것만 임베딩한다 (업로드로 조금씩 추가하는 경우)
    existing = set(col.get(include=[])["ids"]) if col.count() else set()
    chunks = [c for c in chunks if c.id not in existing]
    if not chunks:
        print(f"chroma: {col.count()} chunks already indexed, skip")
        return
    model = SentenceTransformer(settings.embed_model)
    batch = 64
    for i in tqdm(range(0, len(chunks), batch), desc="embed", disable=len(chunks) < 64):
        part = chunks[i : i + batch]
        emb = model.encode([c.text for c in part], normalize_embeddings=True, batch_size=batch)
        col.upsert(
            ids=[c.id for c in part],
            embeddings=emb.tolist(),
            documents=[c.text for c in part],
            metadatas=[{"title": c.title, **(c.meta or {})} for c in part],
        )
    print(f"chroma: {col.count()} chunks")


def main() -> None:
    chunks, questions = load_korquad()
    print(f"chunks: {len(chunks):,}   dev questions: {len(questions):,}")
    settings.data_dir.mkdir(exist_ok=True)
    write_jsonl(settings.chunks_path, (asdict(c) for c in chunks))
    write_jsonl(settings.questions_path, questions)
    build_bm25(chunks)
    build_vector(chunks)


if __name__ == "__main__":
    main()
