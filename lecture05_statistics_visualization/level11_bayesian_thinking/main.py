"""
level11 — 베이지안 사고와 불확실성 소통

1) 전환율에 대한 믿음(베타분포)이 데이터가 쌓이며 좁아지는 과정을
   beta_update.png 로 저장합니다 (사전 -> 증거 -> 사후).
2) A/B 두 안의 사후분포에서 몬테카를로로 P(B>A)·기대 개선·기대 손실을
   계산하고, 경영진용 보고 문장을 자동 생성합니다.
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


def belief_growth() -> None:
    """[1] 진짜 전환율 3.2% 서비스: 방문자가 쌓이며 믿음이 좁아지는 과정."""
    rng = np.random.default_rng(1111)
    true_rate = 0.032
    visitors = rng.random(10_000) < true_rate      # 방문자별 전환 여부
    stages = [0, 100, 1_000, 10_000]
    colors = ["#9aa7b5", "#e1a03c", "#4878cf", "#d1495b"]

    print("[1] 믿음의 성장 — 사전 Beta(1,1) 에서 출발해 관측을 더해 갑니다")
    print(f"    (진짜 전환율 {true_rate:.1%} — 현실에서는 모르는 값)")
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.linspace(0, 0.10, 800)
    for n, color in zip(stages, colors):
        s = int(visitors[:n].sum())               # 전환 수
        f = n - s                                 # 무전환 수
        alpha, beta = 1 + s, 1 + f                # 갱신 규칙: 더하기만!
        post = stats.beta(alpha, beta)
        lo, hi = post.ppf(0.025), post.ppf(0.975)  # 95% 신용구간
        label = (f"n={n:,} (전환 {s}) -> Beta({alpha},{beta})")
        ax.plot(xs, post.pdf(xs), color=color, lw=2, label=label)
        print(f"    n={n:>6,}: 전환 {s:>3}건 | 사후 Beta({alpha:>4},{beta:>5}) | "
              f"95% 신용구간 [{lo:.3%}, {hi:.3%}]")
    ax.axvline(true_rate, color="black", ls=":", lw=1.5, label="진짜 전환율 3.2%")
    ax.set_xlim(0, 0.10)
    ax.set_title("데이터가 쌓일수록 믿음의 분포가 좁아진다 (사전 → 사후)")
    ax.set_xlabel("전환율")
    ax.set_ylabel("믿음의 밀도")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "beta_update.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    분포 갱신 그림 저장: {path}")
    print("    -> 신용구간은 '진짜 값이 이 안에 있을 확률 95%'라고 말해도 되는 구간입니다.")


def ab_bayesian() -> tuple[float, float, float]:
    """[2] A/B 사후분포 몬테카를로: P(B>A), 기대 개선, 기대 손실."""
    rng = np.random.default_rng(1112)
    n_a, s_a = 20_000, 610      # A안: 방문 2만, 전환 610 (3.05%)
    n_b, s_b = 20_000, 668      # B안: 방문 2만, 전환 668 (3.34%)

    post_a = stats.beta(1 + s_a, 1 + n_a - s_a)
    post_b = stats.beta(1 + s_b, 1 + n_b - s_b)

    # 몬테카를로: 두 믿음의 분포에서 10만 쌍을 뽑아 '세어 보기'
    draws_a = post_a.rvs(100_000, random_state=rng)
    draws_b = post_b.rvs(100_000, random_state=rng)
    diff = draws_b - draws_a

    p_b_better = (diff > 0).mean()
    expected_lift = diff.mean()
    # 기대 손실: B 를 골랐는데 사실 A 가 나았을 경우의 평균 손해 (%p)
    expected_loss = np.maximum(-diff, 0).mean()

    print()
    print("[2] A/B 베이지안 판정 — A: 610/20,000 (3.05%) vs B: 668/20,000 (3.34%)")
    print(f"    P(B 가 A 보다 우월)     = {p_b_better:.1%}")
    print(f"    기대 개선 폭            = {expected_lift * 100:+.3f} %p")
    print(f"    B 선택 시 기대 손실     = {expected_loss * 100:.4f} %p (틀렸을 때 위험의 크기)")

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.linspace(0.025, 0.045, 800)
    ax.plot(xs, post_a.pdf(xs), color="#9aa7b5", lw=2, label="A안 사후분포 (3.05%)")
    ax.fill_between(xs, post_a.pdf(xs), color="#9aa7b5", alpha=0.25)
    ax.plot(xs, post_b.pdf(xs), color="#d1495b", lw=2, label="B안 사후분포 (3.34%)")
    ax.fill_between(xs, post_b.pdf(xs), color="#d1495b", alpha=0.25)
    ax.set_title(f"두 안의 믿음의 분포 — 겹친 면적이 '아직 남은 불확실성' (P(B>A)={p_b_better:.0%})")
    ax.set_xlabel("전환율")
    ax.set_ylabel("믿음의 밀도")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "ab_posterior.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    사후분포 비교 그림 저장: {path}")
    return p_b_better, expected_lift, expected_loss


def executive_summary(p_b: float, lift: float, loss: float) -> None:
    """[3] 3.4절 형식의 경영진 보고 문장을 자동 생성합니다."""
    print()
    print("[3] 경영진 보고 문장 (자동 생성)")
    print("    ------------------------------------------------------------")
    print(f"    1. 현재 데이터 기준, B안이 우월할 확률은 {p_b:.0%} 입니다.")
    print(f"    2. 기대 개선 폭은 전환율 {lift * 100:+.2f}%p 이며, 만에 하나 B안이")
    print(f"       열등할 경우의 기대 손실은 {loss * 100:.3f}%p 로 제한적입니다.")
    print("    3. 지금 전환 시 위험은 작고, 1주 더 관측하면 확신은 더 커집니다.")
    print("       전환 시점 결정을 요청드립니다. (사전믿음: 무정보 Beta(1,1))")
    print("    ------------------------------------------------------------")
    print("    -> p-value 없이도, 의사결정에 필요한 확률·크기·위험이 모두 담겼습니다.")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    set_korean_font()
    belief_growth()
    p_b, lift, loss = ab_bayesian()
    executive_summary(p_b, lift, loss)


if __name__ == "__main__":
    main()
