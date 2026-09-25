"""
LLM API 호출의 운영 패턴을 가짜 API 서버로 안전하게 연습합니다.
- 요청/응답의 구조 (모델명, 메시지, 토큰 사용량)
- API 키를 환경변수로 관리하는 법 (코드에 하드코딩 금지)
- 재시도 + 지수 백오프 + 타임아웃 + 비용 추적 클라이언트 구현
가짜 서버는 실제 운영에서 겪는 429(속도 제한)·500(서버 오류)·지연을
시드 고정 난수로 재현합니다.
"""

import os
import random
import time

# 가격표: 백만 토큰당 요금 (교육용 가상 단가, 달러)
PRICE_PER_MTOK = {"input": 3.0, "output": 15.0}
USD_KRW = 1400


# ---------------------------------------------------------------------------
# 가짜 API 서버 — 네트워크 너머에 있다고 상상하세요
# ---------------------------------------------------------------------------

def fake_llm_api(request: dict, rng: random.Random) -> dict:
    """실제 LLM API 의 동작(성공/한도 초과/서버 오류/지연)을 흉내 냅니다."""
    if request.get("api_key") != "sk-demo-1234":
        return {"status": 401, "error": "authentication_error: 잘못된 API 키"}

    roll = rng.random()
    latency = rng.uniform(0.3, 1.2)          # 평상시 응답 지연(초, 가상)
    if roll < 0.25:
        return {"status": 429, "error": "rate_limit_error: 분당 요청 한도 초과",
                "retry_after": 1.0}
    if roll < 0.40:
        return {"status": 500, "error": "internal_server_error: 일시적 서버 오류"}
    if roll < 0.50:
        latency = 45.0                       # 가끔 극단적으로 느린 응답

    prompt = request["messages"][0]["content"]
    in_tok = max(1, len(prompt) // 2)        # 한국어는 대략 2자당 1토큰 가정
    out_text = f"(모의 응답) '{prompt[:14]}...' 요청을 처리했습니다."
    out_tok = max(1, len(out_text) // 2)
    return {"status": 200, "latency": latency,
            "content": out_text,
            "usage": {"input_tokens": in_tok, "output_tokens": out_tok}}


# ---------------------------------------------------------------------------
# 운영급 클라이언트 — 재시도·백오프·타임아웃·비용 추적
# ---------------------------------------------------------------------------

class RobustClient:
    def __init__(self, api_key: str, timeout: float = 10.0,
                 max_retries: int = 4, seed: int = 42, speedup: float = 100.0):
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.rng = random.Random(seed)       # 시드 고정: 데모 재현성
        self.speedup = speedup               # 데모용: 대기시간을 1/100로 축소
        self.total = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "retries": 0}

    def _wait(self, seconds: float, reason: str) -> None:
        print(f"      … {seconds:.1f}초 대기 ({reason})")
        time.sleep(seconds / self.speedup)   # 실제 코드에서는 그대로 sleep(seconds)

    def chat(self, prompt: str) -> str | None:
        request = {"model": "mock-llm-v1", "api_key": self.api_key,
                   "messages": [{"role": "user", "content": prompt}]}
        for attempt in range(self.max_retries + 1):
            resp = fake_llm_api(request, self.rng)
            status = resp["status"]

            if status == 200 and resp["latency"] > self.timeout:
                status = 408                 # 타임아웃: 늦게 온 성공은 실패로 간주
                print(f"    시도 {attempt + 1}: 응답 {resp['latency']:.0f}초 "
                      f"> 타임아웃 {self.timeout:.0f}초 -> 끊고 재시도")
            elif status != 200:
                print(f"    시도 {attempt + 1}: HTTP {status} — {resp['error']}")

            if status == 200:
                self.total["calls"] += 1
                self.total["input_tokens"] += resp["usage"]["input_tokens"]
                self.total["output_tokens"] += resp["usage"]["output_tokens"]
                print(f"    시도 {attempt + 1}: 성공 ({resp['latency']:.1f}초, "
                      f"입력 {resp['usage']['input_tokens']}tok / "
                      f"출력 {resp['usage']['output_tokens']}tok)")
                return resp["content"]
            if status == 401:
                print("    -> 키 오류는 재시도해도 소용없음: 즉시 중단(설정 문제)")
                return None

            if attempt < self.max_retries:
                self.total["retries"] += 1
                # 지수 백오프: 1, 2, 4, 8초... + 무작위 지터(동시 재시도 분산)
                backoff = min(2 ** attempt, 30) + self.rng.uniform(0, 0.5)
                if status == 429 and "retry_after" in resp:
                    backoff = max(backoff, resp["retry_after"])
                self._wait(backoff, "지수 백오프")
        print("    -> 재시도 한도 초과: 실패 처리(큐 적재/사람 알림)")
        return None

    def cost_report(self) -> str:
        cin = self.total["input_tokens"] / 1e6 * PRICE_PER_MTOK["input"]
        cout = self.total["output_tokens"] / 1e6 * PRICE_PER_MTOK["output"]
        return (f"성공 호출 {self.total['calls']}건, 재시도 {self.total['retries']}회 | "
                f"입력 {self.total['input_tokens']:,}tok + 출력 {self.total['output_tokens']:,}tok"
                f" = ${cin + cout:.6f} (약 {(cin + cout) * USD_KRW:.2f}원)")


def main() -> None:
    print("=" * 62)
    print("Level 04 | API 호출 패턴과 키 관리")
    print("=" * 62)

    # [1] API 키는 환경변수로 — 코드/저장소에 절대 남기지 않기
    print("\n[1] API 키 관리: 환경변수에서 읽기")
    key = os.environ.get("MOCK_LLM_API_KEY", "sk-demo-1234")
    masked = key[:7] + "*" * (len(key) - 7)
    src = "환경변수" if "MOCK_LLM_API_KEY" in os.environ else "기본값(데모용)"
    print(f"    MOCK_LLM_API_KEY -> {masked} ({src})")
    print("    실무: export ANTHROPIC_API_KEY=... 처럼 셸/비밀관리자에 저장,")
    print("          코드·깃 저장소에는 키가 한 글자도 들어가면 안 됩니다.")

    # [2] 요청/응답 구조 살펴보기
    print("\n[2] 요청/응답 구조 (한 번 호출해 관찰)")
    rng = random.Random(7)
    demo_req = {"model": "mock-llm-v1", "api_key": key,
                "messages": [{"role": "user", "content": "재택근무 규정을 요약해줘"}]}
    demo_resp = fake_llm_api(demo_req, rng)
    while demo_resp["status"] != 200:        # 관찰용: 성공 응답이 나올 때까지
        demo_resp = fake_llm_api(demo_req, rng)
    print(f"    요청  : model={demo_req['model']!r}, messages=[{{role, content}}]")
    print(f"    응답  : {demo_resp}")
    print("    -> usage 의 토큰 수가 곧 요금입니다. 응답을 받을 때마다 기록하세요.")

    # [3] 잘못된 키: 재시도해도 소용없는 오류 구분
    print("\n[3] 잘못된 키로 호출 (재시도 무의미한 4xx 오류)")
    bad = RobustClient(api_key="sk-wrong-key", seed=1)
    bad.chat("이 요청은 실패해야 정상입니다")

    # [4] 재시도·백오프·타임아웃이 있는 클라이언트로 5건 처리
    print("\n[4] 운영급 클라이언트로 5건 호출 (429/500/지연이 섞인 환경)")
    client = RobustClient(api_key=key, timeout=10.0, seed=7)
    prompts = ["휴가 규정 요약", "경비 정산 기한 안내문 초안", "민원 분류: 배송 지연",
               "회의록 3줄 요약", "보증 기간 안내 문구"]
    for i, p in enumerate(prompts, 1):
        print(f"  ({i}) 요청: {p}")
        answer = client.chat(p)
        if answer:
            print(f"      답변: {answer}")

    # [5] 비용 리포트
    print("\n[5] 비용 추적 리포트")
    print(f"    {client.cost_report()}")
    print("    -> 월 예산 = 예상 호출 수 x 평균 토큰 x 단가. 대시보드로 상시 감시하세요.")

    print("\n요약: 성공 경로만 만들면 데모, 실패 경로(429/500/타임아웃/키 오류)까지")
    print("      만들면 운영입니다. 백오프·상한·비용 추적은 LLM 호출의 기본기입니다.")

    # ------------------------------------------------------------------
    # [참고] 실제 Claude API 라면 (SDK 가 재시도·타임아웃을 대신해 줍니다)
    # import anthropic
    # client = anthropic.Anthropic(          # ANTHROPIC_API_KEY 환경변수 사용
    #     timeout=20.0, max_retries=3)
    # resp = client.messages.create(
    #     model="claude-opus-5", max_tokens=1024,
    #     messages=[{"role": "user", "content": "휴가 규정 요약"}])
    # print(resp.usage.input_tokens, resp.usage.output_tokens)  # 과금 확인
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
