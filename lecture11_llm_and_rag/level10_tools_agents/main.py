"""
도구 호출과 에이전트: LLM 에게 계산기와 문서검색을 쥐여주는 법.
ReAct 루프(생각 Thought -> 행동 Action -> 관찰 Observation 반복)를
규칙 기반 모의 LLM 으로 구현합니다.
- 계산 질문 -> calculator 도구
- 규정 질문 -> doc_search 도구
- 규정 + 계산이 섞인 질문 -> 두 도구를 순서대로 조합
"""

import pathlib
import re
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import hjh_data
from mock_llm import MockEmbedding, split_sentences


# ---------------------------------------------------------------------------
# 도구 1: 계산기 — LLM 은 산수를 자주 틀리므로 계산은 도구에 맡깁니다
# ---------------------------------------------------------------------------

def calculator(expression: str) -> str:
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expression):
        return "오류: 숫자와 사칙연산만 가능합니다"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # 화이트리스트 검증 후 계산
    except Exception as e:
        return f"오류: {e}"


# ---------------------------------------------------------------------------
# 도구 2: 문서 검색 — level05~07 에서 만든 의미 검색을 도구로 포장
# ---------------------------------------------------------------------------

class DocSearchTool:
    def __init__(self, docs: dict[str, str]):
        self.chunks = []
        for name, doc in docs.items():
            self.chunks += [(name, s) for s in split_sentences(doc)]
        texts = [t for _, t in self.chunks]
        self.emb = MockEmbedding(dim=512).fit(texts)
        self.vectors = self.emb.embed_batch(texts)

    def __call__(self, query: str) -> str:
        sims = self.vectors @ self.emb.embed(query)
        i = int(np.argmax(sims))
        src, text = self.chunks[i]
        return f"{text} (출처: {src})"


# ---------------------------------------------------------------------------
# 모의 에이전트 두뇌 — 실제로는 LLM 이 프롬프트로 이 판단을 합니다
# ---------------------------------------------------------------------------

class MockAgentLLM:
    """관찰 기록을 보고 '다음 행동'을 규칙으로 결정하는 모의 LLM.
    실제 에이전트는 도구 목록·사용 예시를 프롬프트로 주면
    모델이 스스로 이 판단(JSON 형식의 도구 호출)을 생성합니다."""

    def decide(self, question: str, notes: dict) -> dict:
        needs_docs = any(w in question for w in ("연차", "출장", "일비", "보증", "재택", "규정"))
        needs_math = any(w in question for w in ("얼마", "며칠 남", "총", "계산")) and \
            re.search(r"\d", question)

        # 1) 규정 관련인데 아직 문서를 안 찾았다면 -> 검색부터
        if needs_docs and "doc" not in notes:
            key = next(w for w in ("연차", "일비", "출장", "보증", "재택", "규정") if w in question)
            # 좋은 검색어 고르기도 에이전트의 몫입니다 (실제로는 LLM 이 생성)
            search_query = {"연차": "연차휴가 부여",
                            "일비": "출장 일비 기준 금액",
                            "출장": "출장 일비 기준 금액"}.get(key, key + " 규정")
            return {"thought": f"'{key}' 기준을 모른다. 문서부터 찾아야 한다.",
                    "action": "doc_search", "input": search_query}

        # 2) 계산이 필요한데 아직 안 했다면 -> 필요한 숫자를 모아 계산기 호출
        if needs_math and "calc" not in notes:
            q_nums = [int(n) for n in re.findall(r"\d+", question)]
            if "doc" in notes:
                doc_nums = [int(n) for n in re.findall(r"\d+", notes["doc"])]
                if "며칠 남" in question and doc_nums:          # 잔여 = 문서값 - 질문값
                    expr = f"{doc_nums[-1]} - {q_nums[-1]}"
                elif "만원" in notes["doc"]:                    # 총액 = 문서 단가 x 질문 일수
                    per_day = [n for n in doc_nums if "만원" in notes["doc"]][-1] * 10000
                    expr = f"{per_day} * {q_nums[-1]}"
                else:
                    expr = " + ".join(map(str, q_nums))
                return {"thought": "문서에서 기준 숫자를 얻었다. 계산은 계산기에 맡긴다.",
                        "action": "calculator", "input": expr}
            expr = re.sub(r"[^0-9+\-*/(). ]", "", question).strip()
            return {"thought": "순수 계산 문제다. 내 암산 대신 계산기를 쓴다.",
                    "action": "calculator", "input": expr}

        # 3) 더 쓸 도구가 없으면 -> 최종 답변
        if not notes:
            return {"thought": "도구로 해결할 수 없는 잡담형 질문이다.",
                    "action": "final",
                    "input": "이 질문은 제 도구(계산기·문서검색)로 답할 수 없습니다."}
        parts = []
        if "doc" in notes:
            parts.append(f"근거: {notes['doc']}")
        if "calc" in notes:
            parts.append(f"계산 결과: {notes['calc']}")
        return {"thought": "필요한 정보가 모두 모였다. 답을 조립한다.",
                "action": "final", "input": " / ".join(parts)}


# ---------------------------------------------------------------------------
# ReAct 루프 — 에이전트의 심장
# ---------------------------------------------------------------------------

def run_agent(question: str, brain: MockAgentLLM, tools: dict, max_steps: int = 4) -> None:
    print(f"\n  질문: {question}")
    notes: dict[str, str] = {}                 # 관찰(도구 결과) 기록장
    for step in range(1, max_steps + 1):
        decision = brain.decide(question, notes)
        print(f"    [Thought {step}] {decision['thought']}")
        if decision["action"] == "final":
            print(f"    [Answer   ] {decision['input']}")
            return
        result = tools[decision["action"]](decision["input"])
        print(f"    [Action  {step}] {decision['action']}({decision['input']!r})")
        print(f"    [Observe {step}] {result}")
        notes["doc" if decision["action"] == "doc_search" else "calc"] = result
    print("    [중단] 최대 단계 초과 — 무한 루프 방지 장치가 작동했습니다.")


def main() -> None:
    np.random.seed(42)

    print("=" * 62)
    print("Level 10 | 도구 호출과 에이전트 — ReAct 루프")
    print("=" * 62)

    print("\n[1] 에이전트에게 쥐여준 도구 2개")
    print("    calculator(식)  : 사칙연산 계산기 (LLM 암산 금지)")
    print("    doc_search(검색어): 사내 문서 의미 검색 (level05~07 재사용)")

    tools = {"calculator": calculator,
             "doc_search": DocSearchTool(hjh_data.SAMPLE_DOCS)}
    brain = MockAgentLLM()

    print("\n[2] 순수 계산 질문 — 도구 1개면 충분")
    run_agent("127 * 34 는 얼마인가요? 계산해줘", brain, tools)

    print("\n[3] 규정 질문 + 계산 — 도구 2개를 순서대로 조합")
    run_agent("연차가 며칠 남았는지 계산해줘. 올해 6일 썼어. 기본 부여 기준으로.", brain, tools)

    print("\n[4] 규정 조회 + 곱셈 — 문서의 단가 x 질문의 일수")
    run_agent("해외 출장을 4일 가면 일비는 총 얼마인가요?", brain, tools)

    print("\n[5] 도구 밖 질문 — 못 하는 일은 못 한다고 답하기")
    run_agent("오늘 점심 메뉴 좀 추천해줘", brain, tools)

    print("\n[6] 정리: 에이전트 = LLM(판단) + 도구(실행) + 루프(반복) + 안전장치")
    print("    - LLM 은 '무엇을 할지'만 정하고, 계산·검색은 도구가 정확히 수행")
    print("    - 매 단계 Thought/Action/Observation 로그가 남아 디버깅 가능")
    print("    - 최대 단계 수 제한: 무한 루프(비용 폭주) 방지의 필수 장치")

    # ------------------------------------------------------------------
    # [참고] 실제 Claude API 의 도구 호출은 이렇게 선언합니다:
    # tools = [{"name": "calculator",
    #           "description": "사칙연산 계산기. 수식 문자열을 받아 결과 반환",
    #           "input_schema": {"type": "object",
    #                            "properties": {"expression": {"type": "string"}},
    #                            "required": ["expression"]}}]
    # response = client.messages.create(model="claude-opus-5", max_tokens=1024,
    #                                   tools=tools, messages=[...])
    # # response.stop_reason == "tool_use" 이면 모델이 도구 호출을 요청한 것.
    # # 도구를 실행해 tool_result 를 붙여 다시 보내는 루프가 오늘 run_agent 와 동일.
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
