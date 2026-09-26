"""
level09 — 仮説検定と p 値

2 つの店舗 (大阪店 vs 名古屋店) の 1 件あたり売上の差が偶然かどうかを裁判にかけます。
1) 並べ替え検定: 店舗ラベルを 10,000 回混ぜ、「偶然の差」の分布から p 値を直接数える
2) scipy の Welch t 検定と結果を比較
3) 本当の差がない標本での対照実験 (第一種の過誤の意味)

※ 図中のテキストは、フォントの文字化けを避けるため英語にしています。
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

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
N_PERM = 10_000


def setup_plot_style() -> None:
    plt.rcParams["axes.unicode_minus"] = False


def load_two_stores() -> tuple[np.ndarray, np.ndarray]:
    rows = hjh_data.sales_table(n_days=60, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0]
    rng = np.random.default_rng(909)
    a = rng.choice(df[df["store"] == "大阪店"]["revenue"].to_numpy(), 150, replace=False)
    b = rng.choice(df[df["store"] == "名古屋店"]["revenue"].to_numpy(), 150, replace=False)
    return a.astype(float), b.astype(float)


def permutation_test(a: np.ndarray, b: np.ndarray, seed: int = 910) -> tuple[float, np.ndarray, float]:
    """ラベル混ぜ検定: 「偶然だけで生じる差」の分布を自分の手で作ります。"""
    rng = np.random.default_rng(seed)
    observed = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    n_a = len(a)
    fake_diffs = np.empty(N_PERM)
    for i in range(N_PERM):
        rng.shuffle(pooled)                      # H0 の世界: ラベルは無意味
        fake_diffs[i] = pooled[:n_a].mean() - pooled[n_a:].mean()
    # 両側検定: 観測された差の絶対値以上が偶然出た比率
    p = (np.abs(fake_diffs) >= abs(observed)).mean()
    return p, fake_diffs, observed


def trial_of_two_stores() -> None:
    a, b = load_two_stores()
    print("[1] 被告: 「大阪店と名古屋店の売上の差は偶然だ」 (帰無仮説 H0)")
    print(f"    大阪店   標本平均 {a.mean() / 1e4:8.1f} 万ウォン (n={len(a)})")
    print(f"    名古屋店 標本平均 {b.mean() / 1e4:8.1f} 万ウォン (n={len(b)})")
    print(f"    観測された差     {(a.mean() - b.mean()) / 1e4:+8.1f} 万ウォン")

    p_perm, fake_diffs, observed = permutation_test(a, b)
    print()
    print(f"[2] 並べ替え検定 — ラベルを {N_PERM:,}回混ぜて作った「偶然の差」の分布と比較")
    print(f"    偶然の差の分布: 平均 {fake_diffs.mean() / 1e4:+.2f}、"
          f"標準偏差 {fake_diffs.std() / 1e4:.2f} (万ウォン)")
    print(f"    p 値 (並べ替え) = {p_perm:.4f}")

    t_stat, p_scipy = stats.ttest_ind(a, b, equal_var=False)   # Welch の t 検定
    print()
    print("[3] scipy の 1 行検定 — stats.ttest_ind(a, b, equal_var=False)")
    print(f"    t = {t_stat:.3f}, p 値 (t 検定) = {p_scipy:.4f}")
    print(f"    -> シミュレーション p({p_perm:.4f}) ≈ 公式 p({p_scipy:.4f})。同じものを計算しています。")
    verdict = "棄却 — 偶然とは考えにくい (差を認定)" if p_scipy < 0.05 else "棄却の失敗 — 偶然でも説明可能"
    print(f"    判決 (有意水準 5%): H0 を{verdict}")

    # 図: 偶然の分布 + 観測された差の位置
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(fake_diffs / 1e4, bins=60, color="#9fbce8", edgecolor="white",
            label="Differences by chance alone (10,000 shuffles)")
    ax.axvline(observed / 1e4, color="#d1495b", lw=2,
               label=f"Observed diff {observed / 1e4:+.1f} (10k KRW)")
    ax.axvline(-observed / 1e4, color="#d1495b", lw=1, ls="--")
    ax.set_title(f"p-value = share of tail beyond red lines = {p_perm:.4f}")
    ax.set_xlabel("Mean difference between stores (10k KRW)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "permutation.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[4] 並べ替え分布の図を保存: {path}")


def control_experiment() -> None:
    """本当の差が「ない」2 つの標本 — 同じ店舗を半分に割って検定。"""
    rows = hjh_data.sales_table(n_days=60, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0]
    rng = np.random.default_rng(911)
    hongdae = df[df["store"] == "新宿店"]["revenue"].to_numpy(dtype=float).copy()
    rng.shuffle(hongdae)
    half1, half2 = hongdae[:100], hongdae[100:200]

    t_stat, p = stats.ttest_ind(half1, half2, equal_var=False)
    print()
    print("[5] 対照実験 — 同じ新宿店の売上を無作為に半分ずつ割って検定")
    print(f"    (定義上、本当の差はなし) t = {t_stat:.3f}, p = {p:.4f}")
    print("    -> p が大きい = 「偶然で十分に説明できる」。ただし、これが「差がないことの")
    print("       証明」ではありません。無罪判決と潔白の証明は別物です。")
    print("    -> こうした「差のない検定」でも、100 回中約 5 回は p<0.05 が出ます。")
    print("       それが有意水準 5% = 第一種の過誤 (誤検知) を受け入れる契約の意味です。")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_plot_style()
    trial_of_two_stores()
    control_experiment()


if __name__ == "__main__":
    main()
