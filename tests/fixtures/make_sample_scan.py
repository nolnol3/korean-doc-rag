"""스캔 문서 fixture: sample_report.pdf 를 이미지로 렌더링해 (1) 텍스트 레이어 없는 PDF, (2) PNG 로 저장.
python tests/fixtures/make_sample_scan.py   (sample_report.pdf 가 먼저 있어야 한다)"""
from pathlib import Path

import pymupdf

here = Path(__file__).parent
src = here / "sample_report.pdf"
if not src.exists():
    raise SystemExit("먼저 python tests/fixtures/make_sample_pdf.py")

with pymupdf.open(src) as doc:
    pix = doc[0].get_pixmap(dpi=300, alpha=False)
png = here / "sample_scan.png"
pix.save(png)

scan = pymupdf.open()
page = scan.new_page(width=pix.width * 72 / 300, height=pix.height * 72 / 300)
page.insert_image(page.rect, pixmap=pix)
scan.save(here / "sample_scan.pdf")
print("written", png, here / "sample_scan.pdf")
