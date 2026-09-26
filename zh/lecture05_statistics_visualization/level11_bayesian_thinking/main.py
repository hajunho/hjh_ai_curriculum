"""
level11 — 贝叶斯思维与不确定性沟通

1) 把对转化率的信念(Beta 分布)随数据积累而收窄的过程
   保存为 beta_update.png (先验 -> 证据 -> 后验)。
2) 从 A/B 两个方案的后验分布中用蒙特卡洛计算 P(B>A)、预期改善、
   预期损失，并自动生成管理层汇报文案。

※ 图内文字统一用英文，避免因环境缺字体而乱码。
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
    """[1] 真实转化率 3.2% 的服务: 访客累积、信念收窄的过程。"""
    rng = np.random.default_rng(1111)
    true_rate = 0.032
    visitors = rng.random(10_000) < true_rate      # 每名访客是否转化
    stages = [0, 100, 1_000, 10_000]
    colors = ["#9aa7b5", "#e1a03c", "#4878cf", "#d1495b"]

    print("[1] 信念的成长 — 从先验 Beta(1,1) 出发，不断加入观测")
    print(f"    (真实转化率 {true_rate:.1%} — 现实中不可知的值)")
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.linspace(0, 0.10, 800)
    for n, color in zip(stages, colors):
        s = int(visitors[:n].sum())               # 转化数
        f = n - s                                 # 未转化数
        alpha, beta = 1 + s, 1 + f                # 更新规则: 只做加法!
        post = stats.beta(alpha, beta)
        lo, hi = post.ppf(0.025), post.ppf(0.975)  # 95% 可信区间
        label = (f"n={n:,} (conv {s}) -> Beta({alpha},{beta})")
        ax.plot(xs, post.pdf(xs), color=color, lw=2, label=label)
        print(f"    n={n:>6,}: 转化 {s:>3}件 | 后验 Beta({alpha:>4},{beta:>5}) | "
              f"95% 可信区间 [{lo:.3%}, {hi:.3%}]")
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
    print(f"    分布更新图已保存: {path}")
    print("    -> 可信区间是'真实值落在里面的概率 95%'这句话可以说出口的区间。")


def ab_bayesian() -> tuple[float, float, float]:
    """[2] A/B 后验分布蒙特卡洛: P(B>A)、预期改善、预期损失。"""
    rng = np.random.default_rng(1112)
    n_a, s_a = 20_000, 610      # A 方案: 访问 2 万, 转化 610 (3.05%)
    n_b, s_b = 20_000, 668      # B 方案: 访问 2 万, 转化 668 (3.34%)

    post_a = stats.beta(1 + s_a, 1 + n_a - s_a)
    post_b = stats.beta(1 + s_b, 1 + n_b - s_b)

    # 蒙特卡洛: 从两个信念分布抽 10 万对、'数一数'
    draws_a = post_a.rvs(100_000, random_state=rng)
    draws_b = post_b.rvs(100_000, random_state=rng)
    diff = draws_b - draws_a

    p_b_better = (diff > 0).mean()
    expected_lift = diff.mean()
    # 预期损失: 选了 B 但其实 A 更好时的平均亏损 (百分点)
    expected_loss = np.maximum(-diff, 0).mean()

    print()
    print("[2] A/B 贝叶斯判定 — A: 610/20,000 (3.05%) vs B: 668/20,000 (3.34%)")
    print(f"    P(B 优于 A)             = {p_b_better:.1%}")
    print(f"    预期改善幅度            = {expected_lift * 100:+.3f} %p")
    print(f"    选 B 的预期损失         = {expected_loss * 100:.4f} %p (押错时风险的大小)")

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.linspace(0.025, 0.045, 800)
    ax.plot(xs, post_a.pdf(xs), color="#9aa7b5", lw=2, label="Posterior of A (3.05%)")
    ax.fill_between(xs, post_a.pdf(xs), color="#9aa7b5", alpha=0.25)
    ax.plot(xs, post_b.pdf(xs), color="#d1495b", lw=2, label="Posterior of B (3.34%)")
    ax.fill_between(xs, post_b.pdf(xs), color="#d1495b", alpha=0.25)
    ax.set_title(f"Beliefs about A and B — overlap = remaining uncertainty (P(B>A)={p_b_better:.0%})")
    ax.set_xlabel("Conversion rate")
    ax.set_ylabel("Belief density")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "ab_posterior.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    后验分布对比图已保存: {path}")
    return p_b_better, expected_lift, expected_loss


def executive_summary(p_b: float, lift: float, loss: float) -> None:
    """[3] 按 3.4 节格式自动生成管理层汇报文案。"""
    print()
    print("[3] 管理层汇报文案 (自动生成)")
    print("    ------------------------------------------------------------")
    print(f"    1. 以当前数据为准, B 方案更优的概率为 {p_b:.0%}。")
    print(f"    2. 预期改善幅度为转化率 {lift * 100:+.2f}%p; 万一 B 方案")
    print(f"       实为劣势, 预期损失仅 {loss * 100:.3f}%p, 相当有限。")
    print("    3. 现在切换风险很小, 再观察一周把握会更大。")
    print("       请决策切换时点。(先验: 无信息 Beta(1,1))")
    print("    ------------------------------------------------------------")
    print("    -> 不用 p 值, 决策所需的概率、幅度、风险也全都在里面了。")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_plot_style()
    belief_growth()
    p_b, lift, loss = ab_bayesian()
    executive_summary(p_b, lift, loss)


if __name__ == "__main__":
    main()
