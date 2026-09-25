"""
level07 — 정규분포와 중심극한정리(CLT)

1) 68-95-99.7 규칙을 난수 100만 개로 검증합니다.
2) 균등분포·지수분포에서 표본 크기 n=1/5/30 의 표본평균을 5,000번씩 뽑아,
   원본 모양과 무관하게 평균의 분포가 종 모양으로 수렴하는 과정을
   outputs/clt_grid.png 한 장(2x3 그리드)으로 저장합니다.
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SAMPLE_SIZES = [1, 5, 30]
N_REPEAT = 5_000  # 표본평균을 몇 번 뽑을지


def set_korean_font() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["AppleGothic", "Malgun Gothic", "NanumGothic", "NanumBarunGothic"]:
        if cand in names:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False


def rule_68_95_997() -> None:
    """정규분포 난수 100만 개로 68-95-99.7 규칙을 셉니다."""
    rng = np.random.default_rng(707)
    z = rng.normal(0, 1, size=1_000_000)
    print("[1] 68-95-99.7 규칙 검증 (표준정규 난수 1,000,000개)")
    for k, expect in [(1, 68.3), (2, 95.4), (3, 99.7)]:
        ratio = (np.abs(z) <= k).mean() * 100
        print(f"    ±{k}σ 안의 비율: {ratio:5.2f}%  (이론값 약 {expect}%)")


def normal_pdf(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    """정규분포 확률밀도함수 (scipy 없이 직접)."""
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def clt_experiment() -> None:
    """균등·지수분포의 표본평균 분포를 그리드로 그립니다."""
    rng = np.random.default_rng(708)

    # (이름, 표본 생성 함수, 모평균, 모표준편차)
    uniform_spec = ("균등분포 U(0,1)", lambda size: rng.uniform(0, 1, size),
                    0.5, 1 / np.sqrt(12))
    expo_spec = ("지수분포 (평균 1)", lambda size: rng.exponential(1.0, size),
                 1.0, 1.0)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    print()
    print("[2][3] 표본평균 실험 — 각 칸은 '표본 n개의 평균'을 5,000번 기록한 분포")
    print(f"    {'원본 분포':<14} {'n':>4} {'평균의 평균':>12} {'평균의 표준편차':>14} {'이론값 σ/√n':>12}")

    for row, (name, sampler, mu, sigma) in enumerate([uniform_spec, expo_spec]):
        for col, n in enumerate(SAMPLE_SIZES):
            # 핵심: (N_REPEAT, n) 표를 만들어 행마다 평균 -> 표본평균 N_REPEAT개
            means = sampler((N_REPEAT, n)).mean(axis=1)
            se_theory = sigma / np.sqrt(n)
            print(f"    {name:<14} {n:>4} {means.mean():>12.4f} "
                  f"{means.std():>14.4f} {se_theory:>12.4f}")

            ax = axes[row][col]
            ax.hist(means, bins=50, density=True, color="#9fbce8", edgecolor="white")
            xs = np.linspace(means.min(), means.max(), 200)
            ax.plot(xs, normal_pdf(xs, mu, se_theory), color="#d1495b", lw=2,
                    label="CLT 이론 곡선")
            ax.set_title(f"{name}, n={n}")
            if col == 0:
                ax.set_ylabel("밀도")
            if row == 0 and col == 2:
                ax.legend(fontsize=9)

    fig.suptitle("중심극한정리 — 원본이 무엇이든 '평균들'은 종 모양으로 모인다", fontsize=13)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "clt_grid.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print()
    print(f"[4] CLT 그리드 저장: {path}")
    print("    왼쪽 열(n=1)은 원본 모양 그대로, 오른쪽 열(n=30)은 종 모양에 밀착합니다.")
    print("[5] 위 표에서 '평균의 표준편차'가 이론값 σ/√n 과 일치함을 확인하세요.")
    print("    -> 표본을 4배 모아야 오차가 절반이 됩니다 (√n 의 경제학).")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    set_korean_font()
    rule_68_95_997()
    clt_experiment()


if __name__ == "__main__":
    main()
