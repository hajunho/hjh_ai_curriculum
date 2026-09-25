"""
평가와 가드레일: RAG 챗봇을 '믿고 내놓을 수 있는' 상태로 만드는 기술.
1) 평가셋(질문+기대 근거)으로 RAG 응답을 자동 채점
   - 정답 키워드 포함 여부 + 답변-근거 일치율(groundedness)
2) 환각 시뮬레이션: 근거에 없는 문장이 섞이면 채점기가 잡아내는지 확인
3) 가드레일: 개인정보(전화·주민번호·이메일) 마스킹 + 금칙어/과잉약속 필터
"""

import pathlib
import re
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import hjh_data
from mock_llm import MockEmbedding, MockLLM, split_sentences

# 평가셋: 질문 / 답에 꼭 있어야 할 키워드 / 기대 출처 (없으면 거절이 정답)
EVAL_SET = [
    {"q": "연차는 며칠 전에 신청해야 하나요?", "keyword": "3영업일", "source": "사내규정_휴가.txt"},
    {"q": "해외 출장 일비는 얼마인가요?", "keyword": "8만원", "source": "사내규정_경비.txt"},
    {"q": "재택근무는 주 몇 회까지 가능한가요?", "keyword": "주 2회", "source": "사내규정_재택.txt"},
    {"q": "보증 기간은 얼마나 되나요?", "keyword": "2년", "source": "제품매뉴얼_보증.txt"},
    {"q": "내년도 최저임금은 얼마인가요?", "keyword": None, "source": None},  # 거절이 정답
]

BANNED_WORDS = ["무조건", "100% 보장", "법적 책임은 없", "경쟁사"]


# ---------------------------------------------------------------------------
# 미니 RAG (level08 축약판)
# ---------------------------------------------------------------------------

class MiniRAG:
    def __init__(self, docs: dict[str, str], min_score: float = 0.15):
        self.min_score = min_score
        self.llm = MockLLM()
        self.chunks = [(n, s) for n, d in docs.items() for s in split_sentences(d)]
        texts = [t for _, t in self.chunks]
        self.emb = MockEmbedding(dim=512).fit(texts)
        self.vectors = self.emb.embed_batch(texts)

    def answer(self, question: str) -> tuple[str, list[tuple[str, str]]]:
        sims = self.vectors @ self.emb.embed(question)
        order = np.argsort(-sims)[:3]
        kept = [(self.chunks[i][0], self.chunks[i][1])
                for i in order if sims[i] >= self.min_score]
        if not kept:
            return "죄송합니다. 문서에서 근거를 찾지 못해 답변드릴 수 없습니다.", []
        return self.llm.answer_with_context(question, kept), kept


# ---------------------------------------------------------------------------
# 채점기: 근거 일치율(groundedness) + 키워드 + 거절 판단
# ---------------------------------------------------------------------------

def groundedness(answer: str, evidence: list[tuple[str, str]]) -> tuple[float, list[str]]:
    """답변의 주장 문장 중 근거 문서에 실제로 있는 비율을 계산합니다.
    비율이 낮다 = 모델이 근거에 없는 말을 지어냈다(환각 신호)."""
    context = " ".join(text for _, text in evidence)
    claims, unsupported = [], []
    for line in answer.splitlines():
        line = line.strip()
        if not line.startswith("- "):          # 인사말 등 상투구는 제외
            continue
        claim = re.sub(r"\s*\(출처:.*?\)\s*$", "", line[2:]).strip()
        claims.append(claim)
        if claim not in context:               # 근거 원문에 없는 주장
            unsupported.append(claim)
    if not claims:
        return 1.0, []                         # 주장 없음(거절 등)은 위반 아님
    return 1 - len(unsupported) / len(claims), unsupported


def grade(item: dict, answer: str, evidence: list) -> dict:
    refused = "답변드릴 수 없습니다" in answer
    if item["keyword"] is None:                 # 문서 밖 질문: 거절해야 만점
        ok = refused
        return {"keyword_ok": ok, "ground": 1.0 if ok else 0.0, "refused": refused}
    g, _ = groundedness(answer, evidence)
    src_ok = any(item["source"] == s for s, _ in evidence)
    return {"keyword_ok": (item["keyword"] in answer) and src_ok,
            "ground": g, "refused": refused}


# ---------------------------------------------------------------------------
# 가드레일: 개인정보 마스킹 + 금칙어 필터
# ---------------------------------------------------------------------------

PII_PATTERNS = [
    (re.compile(r"\d{6}-[1-4]\d{6}"), "[주민번호]"),
    (re.compile(r"01[016789]-?\d{3,4}-?\d{4}"), "[전화번호]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[이메일]"),
]


def mask_pii(text: str) -> tuple[str, int]:
    """외부 API 로 보내기 전/로그에 남기기 전 개인정보를 가립니다."""
    count = 0
    for pattern, token in PII_PATTERNS:
        text, n = pattern.subn(token, text)
        count += n
    return text, count


def check_banned(text: str) -> list[str]:
    """출력에 금칙어·과잉 약속 표현이 있는지 검사합니다."""
    return [w for w in BANNED_WORDS if w in text]


def main() -> None:
    np.random.seed(42)

    print("=" * 62)
    print("Level 11 | 평가·가드레일 — 믿고 내놓을 수 있는 챗봇 만들기")
    print("=" * 62)

    rag = MiniRAG(hjh_data.SAMPLE_DOCS)

    # [1] 평가셋 자동 채점
    print("\n[1] 평가셋 5문항 자동 채점 (키워드/출처 + 근거 일치율)")
    total_kw, total_g = 0, 0.0
    for item in EVAL_SET:
        answer, evidence = rag.answer(item["q"])
        result = grade(item, answer, evidence)
        total_kw += result["keyword_ok"]
        total_g += result["ground"]
        expected = item["keyword"] or "(거절)"
        print(f"    Q: {item['q']}")
        print(f"       기대: {expected:<8} | 정답 {'O' if result['keyword_ok'] else 'X'}"
              f" | 근거일치율 {result['ground']:.0%}"
              f"{' | 거절함' if result['refused'] else ''}")
    print(f"    ---- 총점: 정답률 {total_kw}/{len(EVAL_SET)}, "
          f"평균 근거일치율 {total_g / len(EVAL_SET):.0%}")
    print("    -> 이 숫자를 배포 전/후, 프롬프트 수정 전/후에 비교하는 것이 '평가'입니다.")

    # [2] 환각 검출: 근거에 없는 문장을 답변에 섞어 보기
    print("\n[2] 환각 시뮬레이션 — 채점기가 지어낸 문장을 잡는가?")
    q = "연차는 며칠 전에 신청해야 하나요?"
    answer, evidence = rag.answer(q)
    fake = answer + "\n- 미사용 연차는 최대 30일까지 다음 해로 이월할 수 있다. (출처: 사내규정_휴가.txt)"
    for label, ans in (("정상 답변", answer), ("환각 섞인 답변", fake)):
        g, unsupported = groundedness(ans, evidence)
        print(f"    {label}: 근거일치율 {g:.0%}")
        for u in unsupported:
            print(f"      !! 근거 없는 주장 감지: \"{u[:40]}...\"")
    print("    -> 일치율이 기준(예: 100%) 미만이면 자동 차단하고 사람 검토로 보냅니다.")
    print("       출처 표기까지 그럴듯하게 지어내므로, 표기가 아니라 '원문 대조'로 잡아야 합니다.")

    # [3] 가드레일 1: 개인정보 마스킹 (입력 단계)
    print("\n[3] 가드레일 1 — 개인정보 마스킹 (외부 API 전송 전)")
    user_input = ("환불 문의드립니다. 제 연락처는 010-1234-5678 이고 "
                  "이메일은 hong@example.com, 주민번호는 900101-1234567 입니다.")
    masked, n = mask_pii(user_input)
    print(f"    원문  : {user_input}")
    print(f"    마스킹: {masked}")
    print(f"    -> 개인정보 {n}건을 가렸습니다. LLM 은 마스킹된 버전만 봅니다.")

    # [4] 가드레일 2: 출력 금칙어/과잉 약속 필터
    print("\n[4] 가드레일 2 — 출력 필터 (발송 직전 최종 관문)")
    outputs = [
        "규정에 따라 영수증은 출장 종료 후 7일 이내 제출하시면 됩니다.",
        "고객님, 환불은 무조건 가능하며 100% 보장해 드립니다!",
    ]
    for out in outputs:
        hits = check_banned(out)
        verdict = "통과" if not hits else f"차단 (금칙어: {', '.join(hits)})"
        print(f"    \"{out[:34]}...\" -> {verdict}")
    print("    -> 차단된 응답은 발송하지 않고 안전한 기본 문구로 대체하거나 사람에게 넘깁니다.")

    # [5] 운영 체계 정리
    print("\n[5] 운영 품질 체계 한 장 정리")
    print("    배포 전 : 평가셋 채점(정답률·근거일치율) 통과 기준 정하기")
    print("    입력 단계: 개인정보 마스킹, (필요시) 프롬프트 주입 필터")
    print("    출력 단계: 근거일치율 검사 + 금칙어/과잉약속 필터")
    print("    배포 후 : 실패 사례를 평가셋에 계속 추가 (평가셋은 자산입니다)")

    print("\n요약: '평가 없는 배포'는 감으로 하는 품질 관리입니다.")
    print("      숫자(정답률·근거일치율)와 관문(마스킹·필터)이 신뢰를 만듭니다.")


if __name__ == "__main__":
    main()
