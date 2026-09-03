"""테스트용 HWPX (zip + hwpml XML). 실제 한컴 파일의 최소 구조만 흉내낸다."""
import zipfile
from pathlib import Path

NS = 'xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph"'
def p(text): return f'<hp:p><hp:run><hp:t>{text}</hp:t></hp:run></hp:p>'
def tbl(rows):
    trs = "".join("<hp:tr>" + "".join(f"<hp:tc>{p(c)}</hp:tc>" for c in r) + "</hp:tr>" for r in rows)
    return f"<hp:p><hp:run><hp:tbl>{trs}</hp:tbl></hp:run></hp:p>"
section = f'<?xml version="1.0" encoding="UTF-8"?><hp:sec {NS}>' + p("연차휴가 운영 지침") + \
    p("입사 1년 미만 직원은 1개월 개근 시 1일의 연차가 발생한다. 1년 이상 근속한 직원에게는 15일의 연차가 부여된다.") + \
    p("표 1. 근속연수별 연차일수") + tbl([["근속연수", "연차일수"], ["1년 이상 3년 미만", "15일"], ["3년 이상 5년 미만", "16일"], ["5년 이상", "17일"]]) + \
    p("미사용 연차는 다음 해 3월 31일까지 이월할 수 있으며, 이월 한도는 5일이다.") + "</hp:sec>"
out = Path(__file__).with_name("sample_policy.hwpx")
with zipfile.ZipFile(out, "w") as z:
    z.writestr("mimetype", "application/hwp+zip"); z.writestr("Contents/section0.xml", section)
print("written", out)
