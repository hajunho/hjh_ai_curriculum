"""
LLM을 회사 업무에 쓰는 대표 시나리오 3가지를 모의 LLM으로 체험합니다.
1) 이메일 초안 작성  2) 회의록 요약  3) 민원(문의) 자동 분류
마지막에 도입 시 반드시 점검할 리스크 체크리스트를 출력합니다.
API 키 없이 규칙 기반 MockLLM 으로 동작하며, 실제 API 호출 예시는
주석으로 제공합니다.
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from mock_llm import MockLLM

# 업무 활용 지도: 유형별 대표 업무와 위험도
USE_CASE_MAP = [
    ("요약", "회의록·보고서·계약서 요약", "낮음 (원문 대조 가능)"),
    ("초안", "이메일·공지·기획서 첫 버전", "낮음 (사람이 최종 수정)"),
    ("분류", "민원 라우팅·이메일 태깅", "중간 (오분류 모니터링 필요)"),
    ("추출", "문서에서 날짜·금액·조건 뽑기", "중간 (검증 로직 필수)"),
    ("번역", "해외 거래처 메일·매뉴얼", "중간 (계약 문서는 전문가 검수)"),
    ("코드", "엑셀 수식·SQL·스크립트 생성", "중간 (실행 전 검토)"),
    ("상담", "고객 응대 챗봇", "높음 (환각·책임 문제 -> RAG 필요)"),
]

RISK_CHECKLIST = [
    "이 업무에서 틀린 답이 나가면 얼마나 큰 손해인가? (금액·법적 책임)",
    "사람이 최종 검토하는 단계가 설계에 들어 있는가?",
    "회사 기밀·고객 개인정보를 외부 API 로 보내도 되는가? (보안 규정)",
    "답의 근거를 추적할 수 있는가? (RAG 라면 출처 인용)",
    "모델이 틀렸을 때 되돌릴 절차(롤백·정정 공지)가 있는가?",
    "비용: 호출량 x 토큰 단가가 사람이 하던 비용보다 싼가?",
]

MEETING_NOTE = (
    "3분기 매출은 전년 대비 12% 증가했으나 물류비 상승으로 영업이익은 소폭 감소했다. "
    "마케팅팀은 신규 캠페인 예산 5천만원 증액을 요청했다. "
    "개발팀은 주문 시스템 개선을 다음 달까지 완료하기로 했다. "
    "차기 회의는 10월 15일 오전 10시로 정했다. "
    "점심 메뉴는 구내식당 리뉴얼 이후로 논의를 미뤘다."
)

COMPLAINTS = [
    "택배가 일주일째 도착하지 않는데 운송장 조회도 안 됩니다.",
    "환불 신청했는데 카드 취소가 안 되고 이중 청구까지 됐어요.",
    "제품에서 이상한 소음이 나고 어제부터 작동이 안 됩니다.",
    "상담원 연결이 40분 걸렸고 응대도 불친절했습니다.",
]


def main() -> None:
    llm = MockLLM()

    print("=" * 62)
    print("Level 01 | LLM 업무 활용 지도 — 3종 업무 데모")
    print("=" * 62)

    # [1] 업무 활용 지도
    print("\n[1] LLM 업무 활용 지도 (유형 | 대표 업무 | 리스크)")
    for kind, desc, risk in USE_CASE_MAP:
        print(f"    {kind:<4}| {desc:<24}| {risk}")

    # [2] 업무 데모 1 — 이메일 초안
    print("\n[2] 데모 1: 이메일 초안 작성")
    prompt = (
        "당신은 비서입니다. 아래 정보로 정중한 업무 이메일 초안을 작성하세요. 형식: 인사-용건-마무리\n"
        "받는 사람: 김영업 팀장님\n"
        "목적: 3분기 판매 실적 자료 공유 요청\n"
        "기한: 이번 주 금요일"
    )
    print("    --- 지시(프롬프트) 요지: 받는 사람/목적/기한을 주고 초안 요청 ---")
    for line in llm.complete(prompt).splitlines():
        print(f"    | {line}")
    print("    -> 초안은 30초, 사람은 '검토와 서명'만. 이것이 초안(draft) 활용법입니다.")

    # [3] 업무 데모 2 — 회의록 요약
    print("\n[3] 데모 2: 회의록 요약 (5문장 -> 핵심 3줄)")
    summary = llm.complete("다음 회의록을 개조식 형식으로 요약하세요.\n본문: " + MEETING_NOTE)
    for line in summary.splitlines():
        print(f"    {line}")
    print("    -> 점심 메뉴처럼 중요도 낮은 문장은 빠졌는지 사람이 확인해야 합니다.")

    # [4] 업무 데모 3 — 민원 자동 분류
    print("\n[4] 데모 3: 민원 자동 분류 (담당 부서 라우팅)")
    for text in COMPLAINTS:
        result = llm.complete("다음 민원을 분류하세요. 형식: 분류/근거 키워드\n본문: " + text)
        cat = result.splitlines()[0].replace("분류: ", "")
        basis = result.splitlines()[1].replace("근거 키워드: ", "")
        print(f"    \"{text[:26]}...\"")
        print(f"      -> 분류: {cat:<8} (근거: {basis})")
    print("    -> 하루 500건 민원을 사람이 읽기 전에 부서별로 미리 나눠 줍니다.")

    # [5] 도입 리스크 체크리스트
    print("\n[5] 도입 전 리스크 체크리스트 — 하나라도 '아니오'면 설계를 다시")
    for i, item in enumerate(RISK_CHECKLIST, 1):
        print(f"    {i}. {item}")

    print("\n요약: LLM 은 '초안 생성기 + 분류기'로 시작하는 것이 안전합니다.")
    print("      사람이 최종 책임지는 검토 단계를 반드시 설계에 넣으세요.")

    # ------------------------------------------------------------------
    # [참고] 실제 API 키가 있다면 이렇게 (Claude API 예시)
    # import anthropic
    # client = anthropic.Anthropic()          # ANTHROPIC_API_KEY 환경변수
    # response = client.messages.create(
    #     model="claude-opus-5",
    #     max_tokens=1024,
    #     messages=[{"role": "user", "content": prompt}],
    # )
    # print(response.content[0].text)
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
