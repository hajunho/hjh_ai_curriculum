"""
level00 — 컴퓨터가 언어를 다루는 어려움

단순 문자열 매칭(키워드 검색)으로 상담봇을 만들어 보고,
동의어 / 중의성 / 문맥(반어) / 한국어 교착어 특성 앞에서
어떻게 무너지는지 4가지 실패 사례로 시연합니다.
표준 라이브러리만 사용합니다.
"""


def keyword_match(text: str, keywords: list[str]) -> bool:
    """가장 단순한 접근: 키워드가 하나라도 '부분 문자열'로 들어 있으면 True."""
    return any(kw in text for kw in keywords)


# ---------------------------------------------------------------------------
# [1] 동의어 문제 — 같은 뜻, 다른 글자
# ---------------------------------------------------------------------------

REFUND_REQUESTS = [  # 전부 '환불을 원하는' 고객 문의입니다.
    "환불 처리 부탁드립니다",
    "돈 돌려주세요",
    "결제 취소하고 싶어요",
    "구매 취소 후 입금 요청합니다",
    "환불은 언제 되나요",
    "페이백 가능한가요",
]


def demo_synonym() -> None:
    print("[1] 동의어 문제 — '환불' 키워드 검색은 몇 건을 찾을까?")
    keywords = ["환불"]
    hit = 0
    for text in REFUND_REQUESTS:
        found = keyword_match(text, keywords)
        hit += found
        mark = "잡음  " if found else "놓침 ×"
        print(f"    {mark} | {text}")
    print(f"    -> 실제 환불 요청 {len(REFUND_REQUESTS)}건 중 {hit}건만 검출 "
          f"(적중률 {hit / len(REFUND_REQUESTS):.0%})")
    print("    -> '돈 돌려주세요'와 '환불'이 같은 뜻임을 문자열은 모릅니다.\n")


# ---------------------------------------------------------------------------
# [2] 중의성 문제 — '눈'의 두 의미
# ---------------------------------------------------------------------------

AMBIGUOUS_SENTENCES = [
    ("눈이 펑펑 내려서 길이 막혔다", "snow"),
    ("눈이 침침해서 안과에 갔다", "eye"),
    ("눈을 감고 심호흡을 했다", "eye"),
    ("눈을 뭉쳐서 눈사람을 만들었다", "snow"),
    ("모니터를 오래 봤더니 눈이 아프다", "eye"),
    ("첫눈이 내리면 만나기로 했다", "snow"),
]

# 주변 단어(문맥) 힌트: 뜻을 가르는 단서는 '눈' 자체가 아니라 이웃 단어입니다.
CONTEXT_HINTS = {
    "snow": ["내려", "내리", "뭉쳐", "눈사람", "첫눈", "쌓"],
    "eye": ["침침", "감고", "아프", "안과", "시력", "모니터"],
}


def guess_by_context(sentence: str) -> str:
    """이웃 단어 힌트로 뜻을 추정하는 초미니 '문맥' 분류기."""
    for sense, hints in CONTEXT_HINTS.items():
        if any(h in sentence for h in hints):
            return sense
    return "?"


def demo_ambiguity() -> None:
    print("[2] 중의성 문제 — 같은 글자 '눈', 컴퓨터에게는 완전히 동일")
    correct = 0
    for sentence, answer in AMBIGUOUS_SENTENCES:
        naive = "눈" in sentence          # 문자열 매칭: 전부 '눈 발견'으로 끝
        guess = guess_by_context(sentence)
        correct += guess == answer
        print(f"    문자열매칭={'눈 발견' if naive else '-'} | "
              f"문맥추정={guess:4s} | 정답={answer:4s} | {sentence}")
    print(f"    -> 문자열 매칭은 두 의미를 구분 못 하지만, '주변 단어'를 보면 "
          f"{correct}/{len(AMBIGUOUS_SENTENCES)} 구분 성공")
    print("    -> '단어의 뜻은 이웃이 결정한다' — level06 임베딩의 핵심 아이디어입니다.\n")


# ---------------------------------------------------------------------------
# [3] 문맥·반어 문제 — 긍정 키워드의 배신
# ---------------------------------------------------------------------------

REVIEWS = [  # (리뷰, 실제 감성 1=긍정 0=부정)
    ("품질이 정말 좋다", 1),
    ("배송도 빠르고 최고예요", 1),
    ("좋다고 해서 샀는데 완전 실망했어요", 0),
    ("최고라더니 하루 만에 고장", 0),
    ("가격 대비 안 좋다", 0),
    ("포장이 꼼꼼해서 만족합니다", 1),
]


def demo_sarcasm() -> None:
    print("[3] 문맥 문제 — '좋다/최고' 키워드로 긍정 판정하면?")
    keywords = ["좋다", "최고", "만족"]
    wrong = 0
    for text, label in REVIEWS:
        pred = 1 if keyword_match(text, keywords) else 0
        ok = pred == label
        wrong += not ok
        mark = "정답  " if ok else "오답 ×"
        print(f"    {mark} | 예측={'긍정' if pred else '부정'} "
              f"실제={'긍정' if label else '부정'} | {text}")
    print(f"    -> {len(REVIEWS)}건 중 {wrong}건 오분류. "
          f"'좋다'라는 글자와 '좋다는 뜻'은 다릅니다.\n")


# ---------------------------------------------------------------------------
# [4] 교착어 문제 — 조사가 붙으면 다른 문자열
# ---------------------------------------------------------------------------

DELIVERY_SENTENCES = [
    "배송이 빨라서 놀랐어요",
    "배송은 느렸지만 포장은 좋아요",
    "배송을 기다리는 중입니다",
    "배송도 친절도 모두 만족",
    "재배송 요청드립니다",      # '배송' 포함이지만 다른 개념(재배송)
    "무료배송이라 좋았어요",    # 이것도 복합어
]


def demo_agglutinative() -> None:
    print("[4] 교착어 문제 — '배송'을 찾고 싶은데 조사가 방해한다")
    exact = [s for s in DELIVERY_SENTENCES if "배송" in s.split()]     # 어절 정확 일치
    substr = [s for s in DELIVERY_SENTENCES if "배송" in s]            # 부분 포함
    print(f"    어절 정확 일치('배송' 단독)  : {len(exact)}건 검출 {exact}")
    print(f"    부분 문자열 포함('배송' in s): {len(substr)}건 검출")
    for s in substr:
        note = " <- 재배송/무료배송까지 딸려 옴" if ("재배송" in s or "무료배송" in s) else ""
        print(f"        - {s}{note}")
    print("    -> 정확 일치는 '배송이/배송은/배송을'을 전부 놓치고,")
    print("       부분 포함은 다른 단어까지 끌어들입니다. 형태소 분석(level03)이 필요한 이유.\n")


# ---------------------------------------------------------------------------
# [5] 로드맵
# ---------------------------------------------------------------------------

def print_roadmap() -> None:
    print("[5] 오늘 만난 실패는 이 강의에서 이렇게 해결됩니다")
    roadmap = [
        ("동의어·표기 흔들림", "level01 전처리, level06 임베딩(뜻이 비슷하면 좌표도 가깝게)"),
        ("중의성(문맥)", "level07 RNN, level08 어텐션(주변 단어를 보고 뜻을 정함)"),
        ("반어·부정 표현", "level05 분류 모델 + level08 이후 문맥 모델"),
        ("조사·어미(교착어)", "level03 토큰화·형태소 분석, 서브워드(BPE)"),
        ("신조어", "level03 서브워드(모르는 단어도 조각으로 처리)"),
    ]
    for problem, solution in roadmap:
        print(f"    {problem:12s} -> {solution}")
    print("\n결론: 언어는 '글자'가 아니라 '쓰임새'입니다. 다음 레벨부터 하나씩 정복합니다.")


if __name__ == "__main__":
    print("=" * 70)
    print("컴퓨터가 언어를 다루는 어려움 — 단순 문자열 매칭 붕괴 실험")
    print("=" * 70 + "\n")
    demo_synonym()
    demo_ambiguity()
    demo_sarcasm()
    demo_agglutinative()
    print_roadmap()
