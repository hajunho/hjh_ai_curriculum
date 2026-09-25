"""
level02 — 정규표현식

re 모듈로 가상의 사내 문서에서 전화번호·이메일·금액을 추출하고,
개인정보 마스킹과 탐욕(greedy) 매칭의 함정까지 실습합니다.
표준 라이브러리만 사용합니다.
"""

import re

# ---------------------------------------------------------------------------
# 실습용 사내 문서 (이 강의를 위해 창작한 가상의 텍스트)
# ---------------------------------------------------------------------------

DOCUMENTS = {
    "발주확인_메일.txt": (
        "안녕하세요, 총무팀 김민수입니다. 9월 발주 건 확인드립니다. "
        "사무용 의자 20개, 총액 1,200,000원이며 세금계산서는 "
        "kim.ms@example.co.kr 로 보내 주세요. 급하시면 02-312-4567 "
        "또는 010-2345-6789 로 연락 부탁드립니다."
    ),
    "거래처_연락망.txt": (
        "한빛물류 담당 박서연 010-8765-4321 / seoyeon@hanbit-logi.kr, "
        "누리인쇄 대표번호 031-777-0912, 견적 문의는 quote@nuri-print.co.kr. "
        "야간 긴급 배차는 01055551234 로 문자 남기시면 됩니다."
    ),
    "정산_공지.txt": (
        "3분기 행사비 정산 안내: 부스 임차료 350만 원, 판촉물 제작 9900원짜리 "
        "에코백 300개(총 2,970,000원)입니다. 증빙 누락 문의는 재무팀 "
        "fin.help@example.co.kr (내선 02-312-9999) 로 주세요."
    ),
}

# 이름 붙은 패턴 모음 — 유지보수를 위해 한곳에 모아 둡니다.
PATTERNS = {
    "전화번호": r"0\d{1,2}-?\d{3,4}-?\d{4}",
    "이메일": r"[\w.]+@[\w.-]+\.[a-z]{2,}",
    "금액": r"\d{1,3}(?:,\d{3})+원?|\d+만\s?원|\d+원",
}


def demo_basics() -> None:
    """[1] 기본 부품 워밍업 — 무엇을 / 몇 개나 / 어디서"""
    print("[1] 문법 워밍업: 패턴 부품이 각각 무엇을 잡는지")
    sample = "주문번호 A-2093, 수량 15개, 담당 김민수, 메모: 9월 26일 출고"
    drills = [
        (r"\d+", "숫자 덩어리"),
        (r"[가-힣]+", "한글 덩어리"),
        (r"[A-Z]-\d{4}", "대문자-숫자4개 (주문번호 형식)"),
        (r"\d+개", "숫자+'개' (수량 표현)"),
    ]
    print(f"    대상: {sample}")
    for pattern, meaning in drills:
        found = re.findall(pattern, sample)
        print(f"    {pattern:12s} ({meaning:22s}) -> {found}")
    print()


def demo_extraction() -> None:
    """[2] 문서 3건에서 전화번호·이메일·금액 일괄 추출"""
    print("[2] 정보 추출: 문서 더미에서 연락처와 금액만 걷어 오기")
    for name, text in DOCUMENTS.items():
        print(f"    -- {name}")
        for label, pattern in PATTERNS.items():
            found = re.findall(pattern, text)
            print(f"       {label:5s}: {found}")
    print("    -> 사람이 형광펜 들고 하던 일이 findall 세 번으로 끝납니다.\n")


def mask_phone(text: str) -> str:
    """전화번호 가운데 자리를 ****로 마스킹. 그룹 참조 \\1, \\3 사용."""
    return re.sub(r"(0\d{1,2}-?)(\d{3,4})(-?\d{4})", r"\1****\3", text)


def demo_masking() -> None:
    """[3] 개인정보 마스킹 — re.sub 와 그룹 참조"""
    print("[3] 개인정보 마스킹: 공개용 자료 만들 때의 필수 작업")
    original = DOCUMENTS["거래처_연락망.txt"]
    masked = mask_phone(original)
    print(f"    원본: {original[:60]}...")
    print(f"    마스킹: {masked[:60]}...")
    n = len(re.findall(r"\*{4}", masked))
    print(f"    -> 전화번호 {n}건의 가운데 자리가 **** 로 바뀌었습니다.\n")


def demo_greedy() -> None:
    """[4] 탐욕 매칭 사고 재현 — .* vs .*?"""
    print("[4] 탐욕 매칭의 함정: 별표는 기본적으로 '최대한 길게' 잡는다")
    text = "참석자: <김민수> <박서연> <이도현>"
    greedy = re.findall(r"<.*>", text)
    lazy = re.findall(r"<.*?>", text)
    print(f"    대상        : {text}")
    print(f"    <.*>  (탐욕): {greedy}   <- 통째로 한 덩어리!")
    print(f"    <.*?> (게으름): {lazy}")
    # 치환 사고: 태그만 지우려다 이름까지 전부 삭제되는 사례
    broken = re.sub(r"<.*>", "", text)
    fixed = re.sub(r"<.*?>", "", text)
    print(f"    태그 삭제(탐욕)  : {broken!r}  <- 데이터가 뭉텅이로 증발")
    print(f"    태그 삭제(게으름): {fixed!r}")
    print("    -> 치환 결과가 이상하게 사라졌다면 십중팔구 탐욕 매칭입니다.\n")


def demo_summary_table() -> None:
    """[5] 추출 결과를 CSV 스타일 요약표로 — 실무 최종 산출물 형태"""
    print("[5] 최종 정리: 문서별 추출 결과 요약표 (엑셀 붙여넣기용)")
    print("    문서,전화번호수,이메일수,금액수")
    for name, text in DOCUMENTS.items():
        counts = [len(re.findall(p, text)) for p in PATTERNS.values()]
        print(f"    {name},{counts[0]},{counts[1]},{counts[2]}")
    print("\n결론: 정규식은 '생김새로 찾는' 기술입니다. 뜻으로 찾는 법은 다음 레벨부터.")


if __name__ == "__main__":
    print("=" * 70)
    print("정규표현식 — 비즈니스 문서에서 정보 자동 추출")
    print("=" * 70 + "\n")
    demo_basics()
    demo_extraction()
    demo_masking()
    demo_greedy()
    demo_summary_table()
