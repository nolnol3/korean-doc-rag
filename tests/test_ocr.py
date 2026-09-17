"""OCR 줄·문단 묶기는 순수 함수라 항상 검증. 실제 인식은 easyocr 이 있을 때만 (첫 실행 시 모델 다운로드)."""
from pathlib import Path

import pytest

from kdr.ocr import group_lines

FIX = Path(__file__).parent / "fixtures"


def _box(x0, y0, x1, y1, t):
    return (x0, y0, x1, y1, t)


def test_group_lines_orders_by_row_then_x():
    boxes = [_box(200, 10, 300, 30, "보고서"), _box(10, 12, 150, 32, "사업 현황"),
             _box(10, 50, 150, 70, "매출은"), _box(160, 49, 250, 69, "1,240억")]
    assert group_lines(boxes) == ["사업 현황 보고서 매출은 1,240억"]


def test_group_lines_breaks_paragraph_on_large_gap():
    boxes = [_box(10, 10, 100, 30, "첫 문단"), _box(10, 34, 100, 54, "이어짐"),
             _box(10, 120, 100, 140, "둘째 문단")]
    assert group_lines(boxes) == ["첫 문단 이어짐", "둘째 문단"]


def test_group_lines_empty():
    assert group_lines([]) == []


easyocr = pytest.importorskip("easyocr")


@pytest.fixture(scope="module")
def fixtures():
    if not (FIX / "sample_scan.pdf").exists():
        import subprocess
        import sys

        subprocess.run([sys.executable, FIX / "make_sample_pdf.py"], check=True)
        subprocess.run([sys.executable, FIX / "make_sample_scan.py"], check=True)


def test_scanned_pdf_is_ocred(fixtures):
    from kdr.ingest_docs import parse_pdf

    chunks = parse_pdf(FIX / "sample_scan.pdf")
    assert chunks and all(c.meta["kind"] == "ocr" for c in chunks)
    text = "\n".join(c.text for c in chunks)
    # 고유명사(한빛테크)는 폰트에 따라 틀리기도 해서 숫자·일반어로 확인한다
    assert "사업 현황 보고서" in text and "1998년" in text and "420명" in text


def test_image_is_ocred(fixtures):
    from kdr.ingest_docs import parse_image

    chunks = parse_image(FIX / "sample_scan.png")
    assert chunks and chunks[0].title == "sample_scan.png"
    assert "1998년" in "\n".join(c.text for c in chunks)


def test_text_pdf_keeps_text_layer(fixtures):
    from kdr.ingest_docs import parse_pdf

    kinds = {c.meta["kind"] for c in parse_pdf(FIX / "sample_report.pdf")}
    assert "ocr" not in kinds and "text" in kinds
