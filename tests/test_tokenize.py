from kdr.tokenize import tokenize_kiwi, tokenize_ws


def test_kiwi_strips_josa():
    # "삼성전자의" → "삼성전자" : 형태소 분석이 있어야 질의 "삼성전자"와 맞는다
    assert "삼성전자" in tokenize_kiwi("삼성전자의 본사는 수원이다.")
    assert "삼성전자의" not in tokenize_kiwi("삼성전자의 본사는 수원이다.")


def test_ws_keeps_josa():
    # 공백 분리는 어절 그대로 → 질의 "삼성전자"와 안 맞는다 (ablation이 재는 지점)
    toks = tokenize_ws("삼성전자의 본사는 수원이다.")
    assert "삼성전자의" in toks
    assert "삼성전자" not in toks


def test_numbers_kept():
    assert "1986" in tokenize_kiwi("1986년에 설립되었다.")
