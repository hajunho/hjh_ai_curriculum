"""
level10 — A/B 테스트 설계와 해석

전환율 A/B 테스트 시뮬레이터.
1) 효과가 있는 경우(3.0% vs 3.6%)와 없는 경우(A/A)의 z-검정
2) 효과 없는 테스트 2,000회 반복 -> 1종 오류율이 설계대로 5%인지 검증
3) peeking(중간마다 확인하고 유의하면 조기 종료) -> 거짓양성률 폭등 실험
"""

import os

import numpy as np
from scipy import stats
import matplotlib

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


def z_test_two_proportions(conv_a: int, n_a: int, conv_b: int, n_b: int) -> tuple[float, float]:
    """두 전환율 차이의 z-검정 (직접 구현). (z, p) 를 돌려줍니다."""
    p_a, p_b = conv_a / n_a, conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)          # H0: 두 비율이 같다
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 0.0, 1.0
    z = (p_b - p_a) / se
    p_value = 2 * stats.norm.sf(abs(z))               # 양측검정
    return z, p_value


def run_single_test(rate_a: float, rate_b: float, n: int, seed: int, label: str) -> None:
    """그룹당 n 명짜리 테스트 한 번을 돌리고 판정합니다."""
    rng = np.random.default_rng(seed)
    conv_a = rng.binomial(n, rate_a)
    conv_b = rng.binomial(n, rate_b)
    z, p = z_test_two_proportions(conv_a, n, conv_b, n)
    verdict = "B 승리 선언 (유의)" if p < 0.05 else "판단 유보 (유의하지 않음)"
    print(f"    {label}: A {conv_a / n:.3%} vs B {conv_b / n:.3%} "
          f"(차이 {(conv_b - conv_a) / n:+.3%})")
    print(f"      z = {z:+.2f}, p = {p:.4f} -> {verdict}")


def type1_error_and_peeking() -> None:
    """효과 없는(A/A) 테스트 2,000회 — 정직한 검정 vs peeking."""
    rng = np.random.default_rng(1010)
    n_sims = 2_000
    rate = 0.03
    total_n = 20_000          # 그룹당 최종 표본
    step = 1_000              # peeking 확인 주기
    checkpoints = np.arange(step, total_n + 1, step)

    honest_fp = 0            # 정직: 끝에서 한 번만 검정
    peeking_fp = 0           # 훔쳐보기: 중간에 유의하면 즉시 종료
    sample_trajectories = []  # 그림용 p-value 궤적 몇 개

    for sim in range(n_sims):
        # step 명씩 들어올 때마다의 누적 전환 수 (양 그룹 모두 효과 없음)
        inc_a = rng.binomial(step, rate, size=len(checkpoints))
        inc_b = rng.binomial(step, rate, size=len(checkpoints))
        cum_a, cum_b = np.cumsum(inc_a), np.cumsum(inc_b)

        p_traj = np.array([
            z_test_two_proportions(ca, n, cb, n)[1]
            for ca, cb, n in zip(cum_a, cum_b, checkpoints)
        ])
        if p_traj[-1] < 0.05:
            honest_fp += 1
        if (p_traj < 0.05).any():
            peeking_fp += 1
        if sim < 12:
            sample_trajectories.append(p_traj)

    print()
    print(f"[3] 1종 오류율 검증 — 효과 없는 테스트 {n_sims:,}회 반복")
    print(f"    정직한 검정(끝에서 1회):     거짓양성 {honest_fp:>4}회 "
          f"= {honest_fp / n_sims:.1%}  (설계값 5% 근처)")
    print()
    print(f"[4] peeking — {step:,}명마다 확인, 유의해지는 순간 '승리 선언 후 종료'")
    print(f"    훔쳐보기 정책:               거짓양성 {peeking_fp:>4}회 "
          f"= {peeking_fp / n_sims:.1%}  (약 {peeking_fp / max(honest_fp, 1):.1f}배 폭등!)")
    print("    -> 같은 데이터, 같은 유의수준인데 '언제 보느냐'만으로 오탐이 몇 배가")
    print("       됩니다. 종료 시점은 결과를 보기 전에 정해야 합니다.")

    # 그림: p-value 궤적 — 0.05 선을 잠깐 뚫었다 나오는 낚시의 순간들
    fig, ax = plt.subplots(figsize=(9, 5))
    for traj in sample_trajectories:
        dipped = (traj < 0.05).any()
        ax.plot(checkpoints, traj, lw=1.6 if dipped else 1.0,
                color="#d1495b" if dipped else "#9aa7b5", alpha=0.85)
    ax.axhline(0.05, color="black", ls="--", lw=1, label="유의수준 0.05")
    ax.set_ylim(0, 1)
    ax.set_title("효과가 '없는' 테스트 12건의 p-value 궤적 — 빨간 궤적이 peeking 의 먹잇감")
    ax.set_xlabel("그룹당 누적 표본 수")
    ax.set_ylabel("그 시점의 p-value")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "peeking.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[5] p-value 궤적 그림 저장: {path}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    set_korean_font()

    print("[1] 효과가 있는 경우 — 진짜 전환율 A 3.0% vs B 3.6%, 그룹당 20,000명")
    run_single_test(0.030, 0.036, 20_000, seed=101, label="본 테스트")
    print()
    print("[2] 효과가 없는 경우(A/A) — 양쪽 다 3.0%, 그룹당 20,000명")
    run_single_test(0.030, 0.030, 20_000, seed=102, label="A/A 테스트")

    type1_error_and_peeking()
    print()
    print("[6] 실무 요약: 지표·최소효과·표본크기·종료규칙을 실험 '전에' 문서로 정하고,")
    print("    표본이 찰 때까지 p-value 를 보지 않는 것이 가장 값싼 오탐 방지책입니다.")


if __name__ == "__main__":
    main()
