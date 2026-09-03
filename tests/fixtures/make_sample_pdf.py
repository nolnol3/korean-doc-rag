"""테스트·데모용 한국어 PDF (표 1개 포함) 생성. python tests/fixtures/make_sample_pdf.py"""
import pymupdf as fitz
from pathlib import Path

out = Path(__file__).with_name("sample_report.pdf")
doc = fitz.open(); page = doc.new_page(width=595, height=842)
f = fitz.Font("cjk"); tw = fitz.TextWriter(page.rect)
tw.append((50, 60), "한빛테크 2025년 사업 현황 보고서", font=f, fontsize=16)
tw.append((50, 100), "한빛테크는 1998년 설립된 IT 서비스 기업이다. 본사는 대전에 있으며 임직원은 420명이다.", font=f, fontsize=10)
tw.append((50, 118), "2025년 매출은 1,240억 원, 영업이익은 149억 원으로 영업이익률은 12%이다.", font=f, fontsize=10)
tw.append((50, 150), "표 1. 사업 부문별 매출 비중", font=f, fontsize=11)
rows = [["사업 부문", "매출 비중", "주요 제품"], ["클라우드", "45%", "호스팅, 백업"],
        ["SI", "30%", "공공 시스템 구축"], ["커머스", "15%", "쇼핑몰 솔루션"], ["보안", "10%", "관제 서비스"]]
x0, y0, cw, rh = 50, 165, [120, 80, 200], 22
for r, row in enumerate(rows):
    x = x0
    for c, cell in enumerate(row):
        page.draw_rect(fitz.Rect(x, y0 + r * rh, x + cw[c], y0 + (r + 1) * rh), color=(0, 0, 0), width=0.7)
        tw.append((x + 4, y0 + r * rh + 15), cell, font=f, fontsize=9)
        x += cw[c]
tw.append((50, 320), "한빛 AI 비서는 2025년 3월 베타 출시 이후 2026년 1월 정식 출시되었으며, 부서 단위 구독 과금을 적용한다.", font=f, fontsize=10)
tw.write_text(page)
doc.save(out); print("written", out)
