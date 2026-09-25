"""
프롬프트 엔지니어링: 같은 모델, 다른 지시서가 만드는 품질 차이를 체험합니다.
- 나쁜 프롬프트 vs 좋은 프롬프트(역할·맥락·형식·예시 포함)를 나란히 실행
- 프롬프트 구성 요소 점검기(체크리스트)로 지시서를 채점
MockLLM 은 역할/형식/예시가 명시될수록 구조화된 답을 내도록 만들어져
있어, 실제 LLM 에서 관찰되는 품질 차이를 재현합니다.
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from mock_llm import MockLLM

MEETING_NOTE = (
    "이번 주 재고 회의에서 물류창고 자동화 투자를 내년 1분기로 연기하기로 했다. "
    "대신 성수기 대비 임시 인력 20명을 11월부터 채용한다. "
    "재고 관리 시스템의 오류 건수는 지난달 대비 40% 줄었다. "
    "다음 회의에서 각 지점별 안전재고 기준을 다시 정하기로 했다. "
    "회의실 에어컨 수리 요청은 총무팀에 전달했다."
)

COMPLAINT = "카드로 결제했는데 환불이 2주째 안 되고 고객센터는 연결도 안 돼요."

# 프롬프트 4대 구성 요소: 역할(Role), 맥락(Context), 형식(Format), 예시(Example)
CHECK_ITEMS = [
    ("역할", ["당신은", "역할"], "모델에게 어떤 전문가처럼 행동할지 지정"),
    ("맥락", ["상황", "대상 독자", "맥락"], "왜/누구를 위해 하는 작업인지 배경 제공"),
    ("형식", ["형식", "개조식", "JSON", "글머리"], "출력 모양을 명시 (표/목록/JSON...)"),
    ("예시", ["예시"], "원하는 입출력 샘플 제시 (few-shot)"),
]


def audit_prompt(prompt: str) -> tuple[int, list[str]]:
    """프롬프트에 4대 요소가 들어 있는지 채점합니다."""
    passed = []
    for name, keywords, _ in CHECK_ITEMS:
        if any(k in prompt for k in keywords):
            passed.append(name)
    return len(passed), passed


def show_case(title: str, prompt: str, llm: MockLLM) -> None:
    score, passed = audit_prompt(prompt)
    print(f"\n  ({title}) 포함 요소 {score}/4: {', '.join(passed) if passed else '없음'}")
    print("  --- 프롬프트 ---")
    for line in prompt.strip().splitlines():
        print(f"  > {line}")
    print("  --- MockLLM 응답 ---")
    for line in llm.complete(prompt).splitlines():
        print(f"  | {line}")


def main() -> None:
    llm = MockLLM()

    print("=" * 62)
    print("Level 02 | 프롬프트 엔지니어링 — 지시서 잘 쓰는 법")
    print("=" * 62)

    # [1] 프롬프트 4대 구성 요소
    print("\n[1] 좋은 지시서의 4대 요소")
    for name, _, desc in CHECK_ITEMS:
        print(f"    {name}: {desc}")

    # [2] 사례 A — 회의록 요약: 나쁜 프롬프트 vs 좋은 프롬프트
    print("\n[2] 사례 A: 회의록 요약")
    bad = "요약해줘.\n본문: " + MEETING_NOTE
    good = (
        "당신은 임원 보고용 회의록을 정리하는 비서입니다.\n"
        "상황: 바쁜 임원이 30초 안에 읽을 수 있어야 합니다.\n"
        "다음 회의록을 개조식 형식으로 핵심 결정사항 위주로 요약하세요.\n"
        "본문: " + MEETING_NOTE
    )
    show_case("나쁜 프롬프트", bad, llm)
    show_case("좋은 프롬프트", good, llm)
    print("\n    -> 나쁜 프롬프트는 문장 하나를 던져줄 뿐이지만,")
    print("       좋은 프롬프트는 '누가 왜 읽는지'까지 알려줘 구조화된 답을 얻습니다.")

    # [3] 사례 B — 민원 분류: few-shot 예시의 힘
    print("\n[3] 사례 B: 민원 분류 (few-shot)")
    bad2 = "이거 무슨 민원이야? 분류해봐.\n본문: " + COMPLAINT
    good2 = (
        "당신은 고객센터 민원 분류 담당자입니다.\n"
        "다음 민원을 분류하세요. 형식: '분류: <카테고리>' 한 줄 + 근거 키워드 한 줄.\n"
        "예시: '배송이 늦어요' -> 분류: 배송 / 근거 키워드: 배송\n"
        "본문: " + COMPLAINT
    )
    show_case("나쁜 프롬프트", bad2, llm)
    show_case("좋은 프롬프트", good2, llm)
    print("\n    -> 예시(few-shot)를 주면 출력 형식이 고정되어, 뒤 단계(자동 처리)에")
    print("       바로 연결할 수 있습니다. 형식이 들쭉날쭉하면 자동화가 깨집니다.")

    # [4] 요소별 효과 정리
    print("\n[4] 요소별 효과 요약")
    effects = [
        ("역할", "말투·전문성 수준이 잡힌다 ('비서처럼', '변호사처럼')"),
        ("맥락", "무엇을 남기고 버릴지 기준이 생긴다 (임원용 vs 실무용)"),
        ("형식", "출력이 예측 가능해져 후속 자동화가 쉬워진다"),
        ("예시", "말로 설명하기 힘든 스타일을 몇 개의 샘플로 전달한다"),
    ]
    for name, effect in effects:
        print(f"    {name}: {effect}")

    print("\n요약: 프롬프트는 '신입사원에게 주는 업무 지시서'입니다.")
    print("      역할·맥락·형식·예시 4가지를 채우면 결과 품질이 계단식으로 올라갑니다.")

    # ------------------------------------------------------------------
    # [참고] 실제 API 라면: 시스템 프롬프트에 역할을, 유저 메시지에 작업을
    # import anthropic
    # client = anthropic.Anthropic()
    # response = client.messages.create(
    #     model="claude-opus-5",
    #     max_tokens=1024,
    #     system="당신은 임원 보고용 회의록을 정리하는 비서입니다.",
    #     messages=[{"role": "user", "content": "다음 회의록을 개조식으로 요약..."}],
    # )
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
