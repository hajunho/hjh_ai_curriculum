"""
LLM 서비스 모니터링·거버넌스 실습.
60일치 운영 로그(요청 수, 오류율, 지연시간, 만족도, 토큰 사용량)를 시뮬레이션하고
이동평균과 z-score 로 품질 드리프트(drift)를 탐지하는 텍스트 대시보드를 만듭니다.
비용 리포트, 개인정보 마스킹, API 키 하드코딩 점검까지 —
모델을 '만드는' 일이 끝난 뒤 '지키는' 일이 무엇인지 한 화면에 담습니다.
"""

import re
import statistics

import numpy as np

rng = np.random.default_rng(42)
DAYS = 60
DRIFT_DAY = 45          # 이날부터 품질이 몰래 나빠지기 시작 (예: 상품 카탈로그 개편)
BASELINE_DAYS = 30      # 처음 30일을 '정상 기준선'으로 삼음
PRICE_PER_MTOK = 3.0    # 백만 토큰당 3달러라고 가정


def simulate_logs() -> dict:
    """일별 운영 지표를 만듭니다. DRIFT_DAY 이후 만족도↓·오류율↑ 드리프트 주입."""
    day = np.arange(DAYS)
    requests = (1200 + 8 * day + rng.normal(0, 60, DAYS)).astype(int)  # 완만한 성장
    error_rate = np.clip(rng.normal(0.010, 0.003, DAYS), 0, None)
    latency_p95 = rng.normal(820, 40, DAYS)                            # ms
    feedback = rng.normal(4.35, 0.08, DAYS)                            # 5점 만점
    tokens = requests * rng.normal(900, 40, DAYS)                      # 하루 토큰
    drifted = day >= DRIFT_DAY
    feedback[drifted] -= 0.012 * (day[drifted] - DRIFT_DAY + 1)        # 서서히 하락
    error_rate[drifted] += 0.0022 * (day[drifted] - DRIFT_DAY + 1)     # 서서히 상승
    return {"requests": requests, "error_rate": error_rate,
            "latency_p95": latency_p95, "feedback": feedback, "tokens": tokens}


def moving_avg(x: np.ndarray, w: int = 7) -> np.ndarray:
    """마지막 w 일 평균. 앞부분은 있는 만큼만 평균."""
    return np.array([x[max(0, i - w + 1):i + 1].mean() for i in range(len(x))])


def zscore_today(series: np.ndarray, baseline: np.ndarray) -> float:
    """기준선 분포 대비 오늘(7일 이동평균)의 z-score."""
    return (series[-1] - baseline.mean()) / (baseline.std() + 1e-12)


def bar(v: float, vmax: float, width: int = 24) -> str:
    return "#" * max(1, int(width * v / vmax))


def mask_pii(text: str) -> str:
    """로그를 남기기 전 이메일·전화번호를 가립니다."""
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "<EMAIL>", text)
    text = re.sub(r"01[016789]-?\d{3,4}-?\d{4}", "<PHONE>", text)
    return text


def scan_secrets(code: str) -> list[str]:
    """소스 코드에서 하드코딩된 API 키 흔적을 찾습니다."""
    patterns = [r"sk-[A-Za-z0-9]{16,}", r"(api_key|API_KEY)\s*=\s*['\"][^'\"]{8,}"]
    return [m.group(0) for p in patterns for m in re.finditer(p, code)]


if __name__ == "__main__":
    logs = simulate_logs()

    print("[1] 운영 로그 시뮬레이션 — 60일치 일별 지표")
    print(f"    지표: 요청 수, 오류율, p95 지연, 사용자 만족도(5점), 토큰 사용량")
    print(f"    (몰래 {DRIFT_DAY}일차부터 드리프트를 주입해 두었습니다. 탐지해 봅시다.)")

    print("\n[2] 대시보드 — 최근 10일 요약")
    print(f"    {'일차':>4s} {'요청':>6s} {'오류율':>7s} {'p95(ms)':>8s} {'만족도':>6s}  만족도 추이")
    fb_ma = moving_avg(logs["feedback"])
    for d in range(DAYS - 10, DAYS):
        print(f"    {d + 1:>4d} {logs['requests'][d]:>6d} "
              f"{logs['error_rate'][d]:>6.2%} {logs['latency_p95'][d]:>8.0f} "
              f"{logs['feedback'][d]:>6.2f}  {bar(fb_ma[d] - 3.5, 1.0)}")

    print("\n[3] 드리프트 탐지 — 기준선(첫 30일) 대비 7일 이동평균의 z-score")
    alerts = 0
    for name, series, direction in [("만족도", logs["feedback"], -1),
                                    ("오류율", logs["error_rate"], +1),
                                    ("p95 지연", logs["latency_p95"], +1)]:
        ma = moving_avg(series)
        z = zscore_today(ma, series[:BASELINE_DAYS])
        status = "정상"
        if direction * z > 2.0:  # 나쁜 방향으로 2 시그마 이상 벗어나면 경고
            status = "!! 경고: 드리프트 의심"
            alerts += 1
        print(f"    {name:8s} 기준선 {series[:BASELINE_DAYS].mean():8.3f} → "
              f"현재(7일평균) {ma[-1]:8.3f}  z={z:+5.1f}  {status}")
    print(f"    → 경고 {alerts}건. 모델은 그대로인데 세상(입력 분포)이 바뀌면 이렇게 드러납니다.")

    print("\n[4] 비용 리포트 — 토큰을 돈으로 환산")
    month_tokens = logs["tokens"][-30:].sum()
    month_cost = month_tokens / 1e6 * PRICE_PER_MTOK
    per_req = month_cost / logs["requests"][-30:].sum()
    print(f"    최근 30일 토큰: {month_tokens / 1e6:,.1f}M → 비용 ${month_cost:,.2f}"
          f" (백만 토큰당 ${PRICE_PER_MTOK:.2f} 가정)")
    print(f"    요청 1건당 평균 비용: ${per_req:.4f}")
    print(f"    → 이 숫자가 매출 기여보다 크면, 캐싱·작은 모델·프롬프트 다이어트를 검토합니다.")

    print("\n[5] 개인정보 보호 — 로그에 남기기 전 마스킹")
    raw = "문의: 환불해 주세요. 연락처 kim.cs@example.com / 010-1234-5678 입니다."
    print(f"    원본 로그 : {raw}")
    print(f"    저장 로그 : {mask_pii(raw)}")
    print("    → 원문 그대로 저장하면 로그 열람 권한자 전원에게 개인정보가 노출됩니다.")

    print("\n[6] 보안 점검 — 하드코딩된 API 키 탐지 (커밋 전 자동 검사 흉내)")
    bad_code = 'client = Client(api_key="sk-live-abcdef1234567890XYZ")  # TODO 지우기'
    found = scan_secrets(bad_code)
    print(f"    검사 대상 코드: {bad_code[:52]}...")
    print(f"    탐지된 비밀값 {len(found)}건: {[f[:14] + '...' for f in found]}")
    print("    → 키는 코드가 아니라 환경변수·비밀 관리 서비스에 둡니다.")

    print("\n[7] 마무리 — 운영 체크리스트")
    for item in ["품질 지표(만족도·오류율)를 매일 자동 수집하는가",
                 "드리프트 경고가 사람에게 전달되는 경로가 있는가",
                 "로그에서 개인정보를 마스킹하는가",
                 "API 키가 코드 밖(환경변수)에 있는가",
                 "모델·데이터 버전을 기록해 언제든 되돌릴 수 있는가"]:
        print(f"    [ ] {item}")
    print("    이 다섯 줄이 '모델을 만드는 사람'과 '서비스를 책임지는 사람'의 차이입니다.")
