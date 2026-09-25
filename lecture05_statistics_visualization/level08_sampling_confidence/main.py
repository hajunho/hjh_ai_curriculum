"""
level08 — 표본과 신뢰구간

카페 매출 전체를 '모집단'으로 삼아,
1) n=50 표본으로 95% 신뢰구간을 100번 만들어 진짜 평균을 몇 번 포함하는지 세고
2) 표본 크기에 따라 구간 폭이 sqrt(n)에 반비례해 좁아지는 것을 확인하며
3) 편향 표본(주말만 조사)은 표본을 키워도 계속 틀린다는 것을 보입니다.
"""

import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def set_korean_font() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["AppleGothic", "Malgun Gothic", "NanumGothic", "NanumBarunGothic"]:
        if cand in names:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False


def confidence_interval(sample: np.ndarray, level: float = 0.95):
    """t-분포 기반 신뢰구간 (표본이 작아도 안전)."""
    n = len(sample)
    mean = sample.mean()
    se = sample.std(ddof=1) / np.sqrt(n)          # 표준오차
    t_crit = stats.t.ppf((1 + level) / 2, df=n - 1)
    return mean - t_crit * se, mean + t_crit * se


def ring_toss(pop: np.ndarray, mu: float) -> None:
    """[2][3] 고리 던지기: 신뢰구간 100개 중 몇 개가 mu 를 포함하나."""
    rng = np.random.default_rng(808)
    n_trials, n_sample = 100, 50
    intervals, hits = [], 0
    for _ in range(n_trials):
        sample = rng.choice(pop, size=n_sample, replace=False)
        lo, hi = confidence_interval(sample)
        contains = lo <= mu <= hi
        hits += contains
        intervals.append((lo, hi, contains))

    print()
    print(f"[2] 고리 던지기 — n={n_sample} 표본으로 95% 신뢰구간을 {n_trials}번 제작")
    print(f"    진짜 평균 μ 를 포함한 구간: {hits}/{n_trials} = {hits / n_trials:.0%}")
    print("    -> '95%' 는 구간 하나의 확률이 아니라 '이 제작 방법'의 장기 적중률입니다.")

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (lo, hi, ok) in enumerate(intervals):
        color = "#9aa7b5" if ok else "#d1495b"
        ax.plot([lo / 1e4, hi / 1e4], [i, i], color=color, lw=1.6)
    ax.axvline(mu / 1e4, color="#2e6f40", lw=2, label=f"진짜 평균 μ = {mu / 1e4:.1f}만 원")
    ax.set_title(f"신뢰구간 {n_trials}개 중 {n_trials - hits}개가 μ 를 놓침 (빨간 구간)")
    ax.set_xlabel("건당 매출 (만 원)")
    ax.set_ylabel("실험 번호")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "ci_rings.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[3] 고리 그림 저장: {path}")


def width_vs_n(pop: np.ndarray) -> None:
    """[4] 표본 크기별 구간 폭 — sqrt(n) 반비례 확인."""
    rng = np.random.default_rng(809)
    sizes = [20, 50, 200, 800]
    widths = []
    print()
    print("[4] 표본 크기와 구간 폭 (폭을 절반으로 줄이려면 표본 4배)")
    for n in sizes:
        # 같은 크기로 200번 반복해 평균 폭을 측정 (한 번의 우연을 제거)
        w = [np.subtract(*confidence_interval(rng.choice(pop, n, replace=False))[::-1])
             for _ in range(200)]
        widths.append(np.mean(w))
        print(f"    n = {n:>4}: 평균 구간 폭 {np.mean(w) / 1e4:8.2f} 만 원")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sizes, [w / 1e4 for w in widths], marker="o", color="#4878cf", label="측정된 폭")
    ref = widths[0] * np.sqrt(sizes[0]) / np.sqrt(np.array(sizes))
    ax.plot(sizes, ref / 1e4, ls="--", color="#d1495b", label="이론: 1/√n 곡선")
    ax.set_title("신뢰구간 폭은 √n 에 반비례한다")
    ax.set_xlabel("표본 크기 n")
    ax.set_ylabel("95% 구간 폭 (만 원)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "ci_width.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    폭 그림 저장: {path}")


def biased_sampling(df: pd.DataFrame, mu: float) -> None:
    """[5] 편향 표본: 주말만 조사하면 표본을 키워도 계속 틀립니다."""
    rng = np.random.default_rng(810)
    weekend = df[df["weekday"].isin(["토", "일"])]["revenue"].to_numpy()
    print()
    print("[5] 편향 실험 — '주말에만' 조사한 표본의 신뢰구간")
    for n in [50, 400, 2000]:
        # 주말 손님만 계속 조사하는 상황 (복원 추출로 조사 규모 확대를 흉내)
        sample = rng.choice(weekend, size=n, replace=True)
        lo, hi = confidence_interval(sample)
        verdict = "포함" if lo <= mu <= hi else "놓침!"
        print(f"    n = {n:>4}: [{lo / 1e4:7.1f}, {hi / 1e4:7.1f}] 만 원 -> μ {verdict}")
    print(f"    (진짜 μ = {mu / 1e4:.1f} 만 원)")
    print("    -> 구간은 점점 좁고 자신만만해지지만, 일관되게 틀립니다.")
    print("       편향은 표본 크기로 해결되지 않습니다. '어떻게 뽑았나'가 먼저입니다.")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    set_korean_font()

    rows = hjh_data.sales_table(n_days=180, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0].copy()
    pop = df["revenue"].to_numpy(dtype=float)
    mu = pop.mean()
    print(f"[1] 모집단 준비: 매출 {len(pop):,}건, 진짜 평균 μ = {mu / 1e4:.1f} 만 원")
    print("    (현실에서 μ 는 미지수지만, 오늘은 실험을 위해 알고 시작합니다)")

    ring_toss(pop, mu)
    width_vs_n(pop)
    biased_sampling(df, mu)


if __name__ == "__main__":
    main()
