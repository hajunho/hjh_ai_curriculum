"""
level00 — 숫자로 현실을 요약한다는 것

평균이 똑같이 100인 매출 데이터 4종(A~D)을 만들어,
요약값(평균)만으로는 전혀 다른 현실이 구분되지 않음을 확인합니다.
텍스트 히스토그램으로 각 데이터의 '모양'을 직접 그려 봅니다.
"""

import random
import statistics


def make_datasets() -> dict[str, list[float]]:
    """평균이 모두 100(만 원)이 되도록 설계한 일매출 데이터 4종."""
    rng = random.Random(500)  # 재현성을 위한 seed 고정
    n = 200

    # A. 안정형: 100 주변에 고르게 모임 (단골 위주 가게)
    a = [rng.gauss(100, 8) for _ in range(n)]

    # B. 두 무리형: 평일 손님(60대)과 주말 손님(140대)이 반반
    b = [rng.gauss(60, 7) for _ in range(n // 2)] + \
        [rng.gauss(140, 7) for _ in range(n // 2)]

    # C. 극단값형: 평소엔 80대, 대형 계약 몇 건이 평균을 끌어올림
    c = [rng.gauss(80, 6) for _ in range(n - 4)] + [1200, 1150, 900, 850]

    # D. 복불복형: 40~160 사이에 넓게 퍼짐
    d = [rng.uniform(40, 160) for _ in range(n)]

    data = {"A(안정형)": a, "B(두 무리형)": b, "C(극단값형)": c, "D(복불복형)": d}
    # 미세한 난수 오차를 보정해 평균을 정확히 100으로 맞춥니다.
    for name, values in data.items():
        shift = 100 - statistics.mean(values)
        data[name] = [v + shift for v in values]
    return data


def summarize(values: list[float]) -> dict[str, float]:
    """가장 기본적인 요약값들을 계산합니다."""
    return {
        "평균": statistics.mean(values),
        "중앙값": statistics.median(values),
        "최소": min(values),
        "최대": max(values),
    }


def ascii_hist(values: list[float], lo: float, hi: float,
               n_bins: int = 12, width: int = 40) -> None:
    """구간(bin)을 나눠 빈도를 세고 별표로 그리는 텍스트 히스토그램."""
    counts = [0] * n_bins
    step = (hi - lo) / n_bins
    for v in values:
        idx = int((v - lo) / step)
        idx = max(0, min(n_bins - 1, idx))  # 범위를 벗어난 값은 양 끝 칸에
        counts[idx] += 1
    peak = max(counts) or 1
    for i, cnt in enumerate(counts):
        left = lo + i * step
        bar = "*" * round(cnt / peak * width)
        print(f"  {left:7.1f} ~ {left + step:7.1f} | {bar} ({cnt})")


def main() -> None:
    data = make_datasets()

    print("[1] 요약(평균)만 보면 네 가게는 똑같습니다.")
    for name, values in data.items():
        print(f"    {name:<10} 평균 일매출 = {statistics.mean(values):7.1f} 만 원")

    print()
    print("[2] 다른 요약값을 곁들이면 차이가 드러나기 시작합니다.")
    header = f"    {'데이터':<10} {'평균':>8} {'중앙값':>8} {'최소':>8} {'최대':>8}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for name, values in data.items():
        s = summarize(values)
        print(f"    {name:<10} {s['평균']:>8.1f} {s['중앙값']:>8.1f} "
              f"{s['최소']:>8.1f} {s['최대']:>8.1f}")
    print("    -> C 는 평균(100)과 중앙값(약 80)이 크게 어긋납니다.")
    print("       극단값 몇 건이 평균을 끌어올렸다는 신호입니다.")

    print()
    print("[3] 모양을 직접 봅니다 — 텍스트 히스토그램 (0~200 구간)")
    for name, values in data.items():
        print(f"  <{name}>")
        ascii_hist(values, lo=0, hi=200)
        print()

    print("[4] 요약이 답하지 못하는 질문들")
    print("    Q. 주말 야간 인력을 늘려야 하는 가게는?  -> B (두 무리형)")
    print("    Q. 대형 계약 의존도를 걱정해야 하는 가게는? -> C (극단값형)")
    print("    Q. 현금 흐름 변동 대비가 필요한 가게는?   -> D (복불복형)")
    print("    네 가게의 '평균'은 모두 100으로 같지만, 올바른 결정은 전부 다릅니다.")
    print("    => 요약은 출발점일 뿐, 퍼짐·모양·극단값을 반드시 확인해야 합니다.")


if __name__ == "__main__":
    main()
