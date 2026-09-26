"""
level11 — ベイズ的思考と不確実性の伝え方

1) コンバージョン率に対する信念 (ベータ分布) がデータの蓄積とともに狭くなる過程を
   beta_update.png に保存します (事前 -> 証拠 -> 事後)。
2) A/B 2 案の事後分布からモンテカルロで P(B>A)・期待改善・期待損失を
   計算し、経営層向けの報告文を自動生成します。

※ 図中のテキストは、フォントの文字化けを避けるため英語にしています。
"""

import os

import numpy as np
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def setup_plot_style() -> None:
    plt.rcParams["axes.unicode_minus"] = False


def belief_growth() -> None:
    """[1] 本当のコンバージョン率 3.2% のサービス: 訪問者が貯まるほど信念が狭くなる過程。"""
    rng = np.random.default_rng(1111)
    true_rate = 0.032
    visitors = rng.random(10_000) < true_rate      # 訪問者ごとのコンバージョンの有無
    stages = [0, 100, 1_000, 10_000]
    colors = ["#9aa7b5", "#e1a03c", "#4878cf", "#d1495b"]

    print("[1] 信念の成長 — 事前 Beta(1,1) から出発して観測を足していきます")
    print(f"    (本当のコンバージョン率 {true_rate:.1%} — 現実では知り得ない値)")
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.linspace(0, 0.10, 800)
    for n, color in zip(stages, colors):
        s = int(visitors[:n].sum())               # コンバージョン数
        f = n - s                                 # 非コンバージョン数
        alpha, beta = 1 + s, 1 + f                # 更新ルール: 足すだけ!
        post = stats.beta(alpha, beta)
        lo, hi = post.ppf(0.025), post.ppf(0.975)  # 95% 信用区間
        label = (f"n={n:,} (conv {s}) -> Beta({alpha},{beta})")
        ax.plot(xs, post.pdf(xs), color=color, lw=2, label=label)
        print(f"    n={n:>6,}: コンバージョン {s:>3}件 | 事後 Beta({alpha:>4},{beta:>5}) | "
              f"95% 信用区間 [{lo:.3%}, {hi:.3%}]")
    ax.axvline(true_rate, color="black", ls=":", lw=1.5, label="True rate 3.2%")
    ax.set_xlim(0, 0.10)
    ax.set_title("More data, narrower belief (prior -> posterior)")
    ax.set_xlabel("Conversion rate")
    ax.set_ylabel("Belief density")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "beta_update.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    分布更新の図を保存: {path}")
    print("    -> 信用区間は「本当の値がこの中にある確率 95%」と言ってもよい区間です。")


def ab_bayesian() -> tuple[float, float, float]:
    """[2] A/B 事後分布のモンテカルロ: P(B>A)、期待改善、期待損失。"""
    rng = np.random.default_rng(1112)
    n_a, s_a = 20_000, 610      # A案: 訪問 2 万、コンバージョン 610 (3.05%)
    n_b, s_b = 20_000, 668      # B案: 訪問 2 万、コンバージョン 668 (3.34%)

    post_a = stats.beta(1 + s_a, 1 + n_a - s_a)
    post_b = stats.beta(1 + s_b, 1 + n_b - s_b)

    # モンテカルロ: 2 つの信念の分布から 10 万ペアを抜き取って「数えてみる」
    draws_a = post_a.rvs(100_000, random_state=rng)
    draws_b = post_b.rvs(100_000, random_state=rng)
    diff = draws_b - draws_a

    p_b_better = (diff > 0).mean()
    expected_lift = diff.mean()
    # 期待損失: B を選んだのに実は A が良かった場合の平均的な損 (%pt)
    expected_loss = np.maximum(-diff, 0).mean()

    print()
    print("[2] A/B のベイズ判定 — A: 610/20,000 (3.05%) vs B: 668/20,000 (3.34%)")
    print(f"    P(B が A より優れている) = {p_b_better:.1%}")
    print(f"    期待改善幅               = {expected_lift * 100:+.3f} %pt")
    print(f"    B 選択時の期待損失       = {expected_loss * 100:.4f} %pt (間違っていたときのリスクの大きさ)")

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.linspace(0.025, 0.045, 800)
    ax.plot(xs, post_a.pdf(xs), color="#9aa7b5", lw=2, label="Posterior of A (3.05%)")
    ax.fill_between(xs, post_a.pdf(xs), color="#9aa7b5", alpha=0.25)
    ax.plot(xs, post_b.pdf(xs), color="#d1495b", lw=2, label="Posterior of B (3.34%)")
    ax.fill_between(xs, post_b.pdf(xs), color="#d1495b", alpha=0.25)
    ax.set_title(f"Belief distributions — overlap is the remaining uncertainty (P(B>A)={p_b_better:.0%})")
    ax.set_xlabel("Conversion rate")
    ax.set_ylabel("Belief density")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "ab_posterior.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    事後分布の比較図を保存: {path}")
    return p_b_better, expected_lift, expected_loss


def executive_summary(p_b: float, lift: float, loss: float) -> None:
    """[3] 3.4 節の形式の経営層向け報告文を自動生成します。"""
    print()
    print("[3] 経営層向けの報告文 (自動生成)")
    print("    ------------------------------------------------------------")
    print(f"    1. 現在のデータに基づくと、B 案が優れている確率は {p_b:.0%} です。")
    print(f"    2. 期待改善幅はコンバージョン率 {lift * 100:+.2f}%pt であり、万一 B 案が")
    print(f"       劣っていた場合の期待損失は {loss * 100:.3f}%pt と限定的です。")
    print("    3. 今切り替えた場合のリスクは小さく、あと 1 週間観測すれば確信はさらに高まります。")
    print("       切り替え時期のご判断をお願いします。(事前の信念: 無情報 Beta(1,1))")
    print("    ------------------------------------------------------------")
    print("    -> p 値がなくても、意思決定に必要な確率・大きさ・リスクがすべて含まれています。")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_plot_style()
    belief_growth()
    p_b, lift, loss = ab_bayesian()
    executive_summary(p_b, lift, loss)


if __name__ == "__main__":
    main()
