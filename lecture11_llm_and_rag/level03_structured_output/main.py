"""
LLM 출력을 프로그램이 받아 쓰는 법: 구조화된 출력(JSON)과 검증·재시도.
1) 계약서 텍스트에서 핵심 정보를 JSON 으로 추출하도록 요청
2) 모의 LLM 의 1차 응답은 실제로 자주 겪는 실패(잡담 섞임, 타입 오류)
3) 스키마 검증 -> 실패 원인을 담아 재요청 -> 성공하는 루프를 구현
이 '요청-검증-재시도' 패턴은 실제 API 를 쓸 때 그대로 재사용됩니다.
"""

import json
import pathlib
import re
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # mock_llm 경로 (관례 유지)

CONTRACT = (
    "용역 계약서. 주식회사 한빛물산(이하 '갑')과 청람솔루션 주식회사(이하 '을')는 "
    "재고 관리 시스템 구축 용역에 대해 다음과 같이 계약한다. "
    "계약 금액은 총 120,000,000원(부가세 별도)으로 한다. "
    "계약 기간은 2026년 11월 1일부터 2027년 4월 30일까지로 한다. "
    "을이 납기를 지연할 경우 지연 1일당 계약 금액의 0.1%를 위약금으로 지급한다."
)

# 프로그램이 기대하는 스키마: 필드 이름 -> (타입, 설명)
SCHEMA = {
    "party_a": (str, "갑 회사명"),
    "party_b": (str, "을 회사명"),
    "amount_krw": (int, "계약 금액(원, 숫자만)"),
    "start_date": (str, "시작일 YYYY-MM-DD"),
    "end_date": (str, "종료일 YYYY-MM-DD"),
    "penalty_rate_per_day": (float, "지연 1일당 위약금 비율(%)"),
}


class MockExtractorLLM:
    """추출 요청에 응답하는 모의 LLM.
    1차: 잡담 + 마크다운 펜스 + 금액이 문자열(흔한 실전 실패)
    2차: 오류 피드백이 포함된 강화 프롬프트를 받으면 올바른 JSON"""

    def complete(self, prompt: str) -> str:
        strict_retry = "오류" in prompt and "숫자" in prompt
        if not strict_retry:
            return (
                "네! 계약서에서 정보를 추출해 드릴게요.\n"
                "```json\n"
                "{\n"
                '  "party_a": "주식회사 한빛물산",\n'
                '  "party_b": "청람솔루션 주식회사",\n'
                '  "amount_krw": "1억 2천만원",\n'
                '  "start_date": "2026-11-01",\n'
                '  "end_date": "2027-04-30"\n'
                "}\n"
                "```\n"
                "도움이 되셨길 바랍니다!"
            )
        return (
            "{\n"
            '  "party_a": "주식회사 한빛물산",\n'
            '  "party_b": "청람솔루션 주식회사",\n'
            '  "amount_krw": 120000000,\n'
            '  "start_date": "2026-11-01",\n'
            '  "end_date": "2027-04-30",\n'
            '  "penalty_rate_per_day": 0.1\n'
            "}"
        )


def extract_json_block(text: str) -> str:
    """응답에서 JSON 부분만 방어적으로 잘라냅니다(잡담·펜스 제거)."""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    return brace.group(0) if brace else text.strip()


def validate(data: dict) -> list[str]:
    """스키마 검증: 누락 필드와 타입 오류를 모두 수집해 돌려줍니다."""
    errors = []
    for field, (ftype, desc) in SCHEMA.items():
        if field not in data:
            errors.append(f"필드 누락: {field} ({desc})")
        elif not isinstance(data[field], ftype):
            errors.append(f"타입 오류: {field}는 {ftype.__name__} 이어야 함 "
                          f"(현재 값: {data[field]!r})")
    for d in ("start_date", "end_date"):
        if isinstance(data.get(d), str) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", data[d]):
            errors.append(f"형식 오류: {d}는 YYYY-MM-DD 이어야 함")
    return errors


def extract_with_retry(llm: MockExtractorLLM, max_attempts: int = 3) -> dict | None:
    """요청 -> 파싱 -> 검증 -> (실패 시) 오류를 알려주며 재요청하는 루프."""
    base_prompt = (
        "다음 계약서에서 정보를 추출해 JSON 만 출력하세요.\n"
        f"필드: {', '.join(f'{k}({v[1]})' for k, v in SCHEMA.items())}\n"
        "본문: " + CONTRACT
    )
    feedback = ""
    for attempt in range(1, max_attempts + 1):
        prompt = base_prompt + feedback
        print(f"\n  --- 시도 {attempt} ---")
        raw = llm.complete(prompt)
        preview = raw.replace("\n", " ")[:76]
        print(f"  응답(원문 일부): {preview}...")

        try:
            data = json.loads(extract_json_block(raw))
        except json.JSONDecodeError as e:
            print(f"  [파싱 실패] JSON 해석 불가: {e}")
            feedback = "\n[오류] JSON 만 출력하세요. 설명 문장 금지."
            continue

        errors = validate(data)
        if not errors:
            print("  [검증 통과] 스키마와 일치합니다.")
            return data
        print(f"  [검증 실패] {len(errors)}건:")
        for err in errors:
            print(f"    - {err}")
        # 실패 원인을 다음 프롬프트에 그대로 실어 보냅니다 (자기 수정 유도)
        feedback = ("\n[오류] 이전 응답의 문제: " + " / ".join(errors) +
                    "\n금액은 숫자만, 모든 필드를 빠짐없이, JSON 만 출력하세요.")
    return None


def main() -> None:
    print("=" * 62)
    print("Level 03 | 구조화된 출력 — JSON 추출·검증·재시도")
    print("=" * 62)

    print("\n[1] 과제: 계약서 텍스트 -> 시스템에 입력할 구조화 데이터")
    print(f"    원문({len(CONTRACT)}자): {CONTRACT[:56]}...")
    print("    기대 스키마:")
    for field, (ftype, desc) in SCHEMA.items():
        print(f"      {field:<22}{ftype.__name__:<7}{desc}")

    print("\n[2] 요청-검증-재시도 루프 실행")
    llm = MockExtractorLLM()
    result = extract_with_retry(llm)

    print("\n[3] 최종 결과")
    if result is None:
        print("    3회 시도 모두 실패 -> 사람 검토 대기열로 보냅니다(실무 폴백).")
    else:
        for k, v in result.items():
            print(f"    {k:<22}= {v!r}")
        amount = result["amount_krw"]
        penalty_per_day = amount * result["penalty_rate_per_day"] / 100
        print(f"\n    활용 예: 지연 1일당 위약금 = {amount:,} x 0.1% = {penalty_per_day:,.0f}원")
        print("    -> 타입이 보장되니 곧바로 계산·DB 저장에 쓸 수 있습니다.")

    print("\n[4] 패턴 정리")
    print("    (1) JSON 만 출력하라고 형식을 못박는다 (스키마를 프롬프트에 명시)")
    print("    (2) 그래도 어긋날 수 있으니 파싱은 방어적으로 (펜스·잡담 제거)")
    print("    (3) 검증 실패 사유를 프롬프트에 실어 재요청한다 (보통 1~2회면 해결)")
    print("    (4) 최대 횟수 초과 시 사람에게 넘긴다 (조용한 실패 금지)")

    # ------------------------------------------------------------------
    # [참고] 실제 Claude API 라면: 구조화 출력 기능으로 스키마를 강제할 수
    # 있어 재시도 루프가 거의 필요 없어집니다.
    # import anthropic
    # client = anthropic.Anthropic()
    # response = client.messages.create(
    #     model="claude-opus-5", max_tokens=1024,
    #     messages=[{"role": "user", "content": "계약서에서 추출: " + CONTRACT}],
    #     output_config={"format": {"type": "json_schema", "schema": {...}}},
    # )
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
