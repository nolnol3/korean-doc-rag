"""스캔 PDF 페이지·이미지 파일 → 텍스트 (EasyOCR, ko+en).

텍스트 레이어가 없는 PDF 페이지와 PNG/JPG 파일에만 쓴다. 텍스트 레이어가 있으면 pymupdf 경로가 더 정확하고 빠르다.
EasyOCR은 선택 의존성이다(`pip install -e ".[ocr]"`). 없으면 OCRUnavailable을 던지고 인제스트는 그 페이지를 건너뛴다.

첫 호출 때 인식 모델(~100MB)을 settings.ocr_model_dir 로 내려받는다. 이후는 캐시.
"""
from __future__ import annotations

import functools
import statistics
from pathlib import Path

from kdr.config import settings

Box = tuple[float, float, float, float, str]  # x0, y0, x1, y1, text


class OCRUnavailable(RuntimeError):
    pass


def available() -> bool:
    """easyocr 이 import 되는가 (모델은 로드하지 않는다). /health 와 업로드 경고가 쓴다."""
    try:
        import easyocr  # noqa: F401
    except ImportError:
        return False
    return True


@functools.lru_cache(maxsize=1)
def _reader():
    try:
        import easyocr
    except ImportError as e:
        raise OCRUnavailable("easyocr 가 없습니다 — pip install -e '.[ocr]'") from e
    settings.ocr_model_dir.mkdir(parents=True, exist_ok=True)
    langs = [s.strip() for s in settings.ocr_langs.split(",") if s.strip()]
    return easyocr.Reader(langs, gpu=settings.ocr_gpu, model_storage_directory=str(settings.ocr_model_dir),
                          verbose=False)


def group_lines(boxes: list[Box]) -> list[str]:
    """인식된 단어/구 상자들을 읽는 순서대로 문단 문자열로 묶는다.

    y 중심이 같은 줄 높이 안에 있으면 같은 줄, 줄 사이 세로 간격이 줄 높이의 1.5배를 넘으면 문단을 끊는다.
    한 줄 안에서는 x 순서. 순수 함수라 모델 없이 테스트한다.
    """
    if not boxes:
        return []
    h = statistics.median(y1 - y0 for _, y0, _, y1, _ in boxes) or 1.0
    lines: list[list[Box]] = []
    for b in sorted(boxes, key=lambda b: ((b[1] + b[3]) / 2, b[0])):
        cy = (b[1] + b[3]) / 2
        if lines and abs(cy - _cy(lines[-1])) < h * 0.6:
            lines[-1].append(b)
        else:
            lines.append([b])
    paras: list[str] = []
    prev_bottom: float | None = None
    for ln in lines:
        text = " ".join(b[4].strip() for b in sorted(ln, key=lambda b: b[0]) if b[4].strip())
        if not text:
            continue
        top = min(b[1] for b in ln)
        if paras and prev_bottom is not None and top - prev_bottom <= h * 1.5:
            paras[-1] += " " + text
        else:
            paras.append(text)
        prev_bottom = max(b[3] for b in ln)
    return paras


def _cy(line: list[Box]) -> float:
    return sum((b[1] + b[3]) / 2 for b in line) / len(line)


def _read(image) -> list[str]:
    """image: 파일 경로(str) 또는 HxWx3 uint8 배열."""
    raw = _reader().readtext(image, detail=1, paragraph=False)
    boxes: list[Box] = []
    for quad, text, conf in raw:
        if conf < settings.ocr_min_conf:
            continue
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        boxes.append((min(xs), min(ys), max(xs), max(ys), text))
    return group_lines(boxes)


def ocr_image(path: Path) -> list[str]:
    return _read(str(path))


def ocr_pdf_page(page) -> list[str]:
    """pymupdf Page 를 settings.ocr_dpi 로 렌더링해 인식한다."""
    import numpy as np
    import pymupdf

    pix = page.get_pixmap(dpi=settings.ocr_dpi, colorspace=pymupdf.csRGB, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    return _read(arr)
