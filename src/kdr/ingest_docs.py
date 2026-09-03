"""PDF · HWPX 문서 → 페이지·표 단위 청크 → 'docs' 컬렉션.

  python -m kdr.ingest_docs path/to/dir_or_file [...]
  COLLECTION=docs make serve        # 이 컬렉션으로 서비스

표는 markdown으로 보존해 청크 하나로 넣는다. 본문은 페이지 안에서 문단을 합쳐 400~900자.
인용 제목은 "파일명 p.N" — API 응답의 citations.title 에 그대로 나온다.
HWP 5.0(구형 바이너리)은 다루지 않는다. HWPX(zip+XML)만.
"""
from __future__ import annotations

import hashlib
import re
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path
from xml.etree import ElementTree as ET

from kdr.config import settings
from kdr.ingest import Chunk, build_bm25, build_vector, write_jsonl

MIN, MAX = 400, 900


def _cid(*parts: str) -> str:
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:16]  # 마지막 인자에 본문을 넣어 내용 변경이 id에 반영되게


def _pack(paras: list[str]) -> list[str]:
    """문단들을 MIN~MAX 자 청크로 합친다. 너무 긴 문단은 문장 경계에서 자른다."""
    out, buf = [], ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) + 1 <= MAX:
            buf = f"{buf}\n{p}" if buf else p
            continue
        if buf:
            out.append(buf)
        while len(p) > MAX:
            cut = max(p.rfind(". ", 0, MAX), p.rfind("다. ", 0, MAX), MAX)
            out.append(p[:cut + 1].strip())
            p = p[cut + 1:].strip()
        buf = p
    if buf:
        if out and len(buf) < MIN and len(out[-1]) + len(buf) <= MAX + MIN:
            out[-1] += "\n" + buf
        else:
            out.append(buf)
    return out


# ── PDF ───────────────────────────────────────────────────────────────────────

def parse_pdf(path: Path) -> list[Chunk]:
    import fitz  # pymupdf

    chunks: list[Chunk] = []
    name = path.name
    with fitz.open(path) as doc:
        for pno, page in enumerate(doc, start=1):
            title = f"{name} p.{pno}"
            # 표: 먼저 뽑고, 본문에서 그 영역은 뺀다
            table_rects = []
            try:
                tables = page.find_tables()
            except Exception:
                tables = None
            for ti, t in enumerate(getattr(tables, "tables", []) or []):
                md = t.to_markdown().strip()
                if md.count("|") < 4:
                    continue
                table_rects.append(t.bbox)
                chunks.append(Chunk(_cid(name, str(pno), "table", str(ti), md), title, f"[표] {md}",
                                    {"source": name, "page": pno, "kind": "table"}))
            # 본문: 블록 단위로 읽되 표 영역과 겹치는 블록은 제외
            paras = []
            for b in page.get_text("blocks"):
                x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
                if any(fitz.Rect(x0, y0, x1, y1).intersects(r) for r in table_rects):
                    continue
                # 블록 안의 줄바꿈은 PDF 줄바꿈(레이아웃)이지 문단 구분이 아니다 — 한 줄로 잇는다
                txt = re.sub(r"[ \t]+", " ", txt.replace("\n", " ")).strip()
                if len(txt) >= 2:
                    paras.append(txt)
            for i, text in enumerate(_pack(paras)):
                chunks.append(Chunk(_cid(name, str(pno), "text", str(i), text), title, text,
                                    {"source": name, "page": pno, "kind": "text"}))
    return chunks


# ── HWPX ──────────────────────────────────────────────────────────────────────

_NS = {"hp": "http://www.hancom.co.kr/hwpml/2011/paragraph"}


def _hwpx_text(el) -> str:
    return "".join(t.text or "" for t in el.iter(f"{{{_NS['hp']}}}t")).strip()


def parse_hwpx(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    name = path.name
    with zipfile.ZipFile(path) as z:
        sections = sorted(n for n in z.namelist() if re.match(r"Contents/section\d+\.xml", n))
        for sno, sec in enumerate(sections, start=1):
            root = ET.fromstring(z.read(sec))
            title = f"{name} §{sno}"
            paras, ti = [], 0
            for p in root.iter(f"{{{_NS['hp']}}}p"):
                tbl = p.find(f".//{{{_NS['hp']}}}tbl")
                if tbl is not None:
                    rows = []
                    for tr in tbl.iter(f"{{{_NS['hp']}}}tr"):
                        cells = [_hwpx_text(tc).replace("|", "/") for tc in tr.iter(f"{{{_NS['hp']}}}tc")]
                        rows.append("| " + " | ".join(cells) + " |")
                    if len(rows) >= 2:
                        rows.insert(1, "|" + "---|" * (rows[0].count("|") - 1))
                        chunks.append(Chunk(_cid(name, str(sno), "table", str(ti), "\n".join(rows)), title, "[표] " + "\n".join(rows),
                                            {"source": name, "page": sno, "kind": "table"}))
                        ti += 1
                    continue
                if p.find(f"..//{{{_NS['hp']}}}tbl") is not None:
                    continue  # 표 안의 문단은 위에서 처리됨
                t = _hwpx_text(p)
                if t:
                    paras.append(t)
            for i, text in enumerate(_pack(paras)):
                chunks.append(Chunk(_cid(name, str(sno), "text", str(i), text), title, text,
                                    {"source": name, "page": sno, "kind": "text"}))
    return chunks


# ── main ──────────────────────────────────────────────────────────────────────

PARSERS = {".pdf": parse_pdf, ".hwpx": parse_hwpx}


def collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        files += [f for f in (p.rglob("*") if p.is_dir() else [p]) if f.suffix.lower() in PARSERS]
    return sorted(set(files))


def add_files(files: list[Path], collection: str) -> dict:
    """파일들을 파싱해 컬렉션에 **추가**한다 (기존 청크 유지, 같은 id는 중복 없이). 업로드 API가 쓴다."""
    import json

    new: list[Chunk] = []
    per_file: dict[str, int] = {}
    for f in files:
        parser = PARSERS.get(f.suffix.lower())
        if not parser:
            continue
        got = parser(f)
        per_file[f.name] = len(got)
        new += got
    path = settings.chunks_path_for(collection)
    existing: dict[str, Chunk] = {}
    if path.exists():
        with path.open() as fh:
            for r in map(json.loads, fh):
                existing[r["id"]] = Chunk(r["id"], r["title"], r["text"], r.get("meta"))
    for c in new:
        existing.setdefault(c.id, c)
    chunks = list(existing.values())
    settings.data_dir.mkdir(exist_ok=True)
    write_jsonl(path, (asdict(c) for c in chunks))
    build_bm25(chunks, collection)
    build_vector(chunks, collection=collection)
    return {"files": per_file, "added": len(new), "total": len(chunks), "collection": collection}


def main(argv: list[str]) -> None:
    if settings.collection == "korquad":
        sys.exit("COLLECTION=korquad 는 KorQuAD 전용입니다. 예: COLLECTION=docs python -m kdr.ingest_docs ./my_docs")
    files = collect(argv)
    if not files:
        sys.exit("PDF/HWPX 파일이 없습니다")
    chunks: list[Chunk] = []
    for f in files:
        got = PARSERS[f.suffix.lower()](f)
        n_tbl = sum(1 for c in got if c.meta and c.meta["kind"] == "table")
        print(f"{f.name}: {len(got)} chunks ({n_tbl} tables)")
        chunks += got
    settings.data_dir.mkdir(exist_ok=True)
    write_jsonl(settings.chunks_path, (asdict(c) for c in chunks))
    build_bm25(chunks)
    build_vector(chunks, rebuild=True)  # 문서 컬렉션은 매번 처음부터 — 지운 문서가 남지 않게
    print(f"collection '{settings.collection}': {len(chunks)} chunks from {len(files)} files")


if __name__ == "__main__":
    main(sys.argv[1:])
