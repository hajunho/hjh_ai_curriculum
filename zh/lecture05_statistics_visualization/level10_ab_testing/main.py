"""
level10 — A/B 测试的设计与解读

转化率 A/B 测试模拟器。
1) 有效应(3.0% vs 3.6%)与无效应(A/A)情形的 z 检验
2) 无效应测试重复 2,000 次 -> 验证第一类错误率是否如设计的 5%
3) peeking(中途反复确认、一显著就提前收工) -> 假阳性率暴涨实验

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


def z_test_two_proportions(conv_a: int, n_a: int, conv_b: int, n_b: int) -> tuple[float, float]:
    """两个转化率之差的 z 检验 (亲手实现)。返回 (z, p)。"""
    p_a, p_b = conv_a / n_a, conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)          # H0: 两个比例相同
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 0.0, 1.0
    z = (p_b - p_a) / se
    p_value = 2 * stats.norm.sf(abs(z))               # 双侧检验
    return z, p_value


def run_single_test(rate_a: float, rate_b: float, n: int, seed: int, label: str) -> None:
    """跑一次每组 n 人的测试并给出判定。"""
    rng = np.random.default_rng(seed)
    conv_a = rng.binomial(n, rate_a)
    conv_b = rng.binomial(n, rate_b)
    z, p = z_test_two_proportions(conv_a, n, conv_b, n)
    verdict = "宣布 B 获胜 (显著)" if p < 0.05 else "保留判断 (不显著)"
    print(f"    {label}: A {conv_a / n:.3%} vs B {conv_b / n:.3%} "
          f"(差异 {(conv_b - conv_a) / n:+.3%})")
    print(f"      z = {z:+.2f}, p = {p:.4f} -> {verdict}")


def type1_error_and_peeking() -> None:
    """无效应(A/A)测试 2,000 次 — 诚实检验 vs peeking。"""
    rng = np.random.default_rng(1010)
    n_sims = 2_000
    rate = 0.03
    total_n = 20_000          # 每组最终样本
    step = 1_000              # peeking 确认周期
    checkpoints = np.arange(step, total_n + 1, step)

    honest_fp = 0            # 诚实: 只在最后检验一次
    peeking_fp = 0           # 偷看: 中途一显著就立即收工
    sample_trajectories = []  # 画图用的几条 p 值轨迹

    for sim in range(n_sims):
        # 每进来 step 人时的累积转化数 (两组都没有效应)
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
    print(f"[3] 第一类错误率验证 — 无效应测试重复 {n_sims:,} 次")
    print(f"    诚实检验(最后 1 次):         假阳性 {honest_fp:>4}次 "
          f"= {honest_fp / n_sims:.1%}  (设计值 5% 附近)")
    print()
    print(f"[4] peeking — 每 {step:,} 人确认一次，一显著就'宣布胜利、立即收工'")
    print(f"    偷看政策:                    假阳性 {peeking_fp:>4}次 "
          f"= {peeking_fp / n_sims:.1%}  (暴涨约 {peeking_fp / max(honest_fp, 1):.1f}倍!)")
    print("    -> 同样的数据、同样的显著性水平，仅仅'什么时候看'就让误报翻了几倍。")
    print("       收盘时点必须在看到结果之前定下来。")

    # 图: p 值轨迹 — 短暂击穿 0.05 线又弹回的上钩瞬间
    fig, ax = plt.subplots(figsize=(9, 5))
    for traj in sample_trajectories:
        dipped = (traj < 0.05).any()
        ax.plot(checkpoints, traj, lw=1.6 if dipped else 1.0,
                color="#d1495b" if dipped else "#9aa7b5", alpha=0.85)
    ax.axhline(0.05, color="black", ls="--", lw=1, label="Significance level 0.05")
    ax.set_ylim(0, 1)
    ax.set_title("p-value paths of 12 tests with NO effect — red paths are peeking bait")
    ax.set_xlabel("Cumulative sample size per group")
    ax.set_ylabel("p-value at that moment")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "peeking.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[5] p 值轨迹图已保存: {path}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_plot_style()

    print("[1] 有效应的情形 — 真实转化率 A 3.0% vs B 3.6%, 每组 20,000 人")
    run_single_test(0.030, 0.036, 20_000, seed=101, label="正式测试")
    print()
    print("[2] 无效应的情形(A/A) — 两边都是 3.0%, 每组 20,000 人")
    run_single_test(0.030, 0.030, 20_000, seed=102, label="A/A 测试")

    type1_error_and_peeking()
    print()
    print("[6] 实务总结: 指标、最小效应、样本量、终止规则要在实验'之前'写进文档，")
    print("    在样本攒满之前不看 p 值，就是成本最低的防误报手段。")


if __name__ == "__main__":
    main()
