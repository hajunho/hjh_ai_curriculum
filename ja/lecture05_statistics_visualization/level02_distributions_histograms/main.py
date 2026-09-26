"""
level02 — 分布とヒストグラム

カフェチェーンの売上データの分布を、ヒストグラム PNG 3 種類として保存します。
1) bin の数が物語 (形) を変える実験
2) 右の裾 (歪度) と平均・中央値の食い違い
3) 平日/週末が混ざってできた 2 つの山の分布

※ グラフ内のテキストは、日本語フォントの文字化けを避けるため英語にしています。
   (ターミナル出力は日本語です。)
"""

import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # 画面を使わずファイルにのみ保存
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def setup_plot_style() -> None:
    """グラフの共通設定。図中のラベルは英語なので追加フォントは不要です。"""
    plt.rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け防止


def load_sales() -> pd.DataFrame:
    rows = hjh_data.sales_table(n_days=180, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0].copy()
    df["revenue_man"] = df["revenue"] / 10_000  # 万ウォン単位で読みやすく
    return df


def describe(df: pd.DataFrame) -> None:
    rev = df["revenue_man"]
    mean, med = rev.mean(), rev.median()
    p95 = np.percentile(rev, 95)
    skew = rev.skew()
    print("[1] 売上 (1 件あたり、万ウォン) の分布の要約")
    print(f"    平均 {mean:.1f} | 中央値 {med:.1f} | p95 {p95:.1f} | 歪度 {skew:.2f}")
    print(f"    -> 平均 > 中央値、歪度 > 0 : 右の裾が長い分布です。")


def plot_bins_experiment(df: pd.DataFrame) -> None:
    """同じデータを bin 5 / 30 / 200 で描いて比較します。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, bins in zip(axes, [5, 30, 200]):
        ax.hist(df["revenue_man"], bins=bins, color="#4878cf", edgecolor="white")
        ax.set_title(f"bins = {bins}")
        ax.set_xlabel("Revenue per sale (10k KRW)")
    axes[0].set_ylabel("Frequency")
    fig.suptitle("Same data, different bins — the scale changes the story")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "hist_bins.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[2] bin 実験を保存: {path}")
    print("    bins=5 は山を潰し、bins=200 は乱数の凹凸まで描いてしまいます。")


def plot_skew(df: pd.DataFrame) -> None:
    """右裾の分布の上に、平均/中央値の縦線を重ねます。"""
    rev = df["revenue_man"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(rev, bins=40, color="#9fbce8", edgecolor="white")
    ax.axvline(rev.mean(), color="#d1495b", lw=2, label=f"Mean {rev.mean():.0f}")
    ax.axvline(rev.median(), color="#2e6f40", lw=2, ls="--",
               label=f"Median {rev.median():.0f}")
    ax.set_title("The right tail drags the mean")
    ax.set_xlabel("Revenue per sale (10k KRW)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "hist_skew.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[3] 歪度の図を保存: {path}")


def plot_bimodal(df: pd.DataFrame) -> None:
    """平日/週末に分けて重ね描きすると、隠れた 2 つの集団が見えます。"""
    weekend = df[df["weekday"].isin(["土", "日"])]["revenue_man"]
    weekday = df[~df["weekday"].isin(["土", "日"])]["revenue_man"]
    bins = np.linspace(df["revenue_man"].min(), df["revenue_man"].max(), 40)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # density=True: 標本数が違っても比率で公平に比較
    ax.hist(weekday, bins=bins, density=True, alpha=0.6, label="Weekday", color="#4878cf")
    ax.hist(weekend, bins=bins, density=True, alpha=0.6, label="Weekend", color="#e1a03c")
    ax.set_title("One lump was actually the sum of two groups")
    ax.set_xlabel("Revenue per sale (10k KRW)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "hist_bimodal.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[4] 平日/週末の分離図を保存: {path}")
    print(f"    平日の中央値 {weekday.median():.0f} vs 週末の中央値 {weekend.median():.0f} (万ウォン)")
    print("    -> 集団が混ざった分布を丸ごと解釈すると、2 つの物語を 1 つに潰してしまいます。")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_plot_style()
    df = load_sales()
    describe(df)
    plot_bins_experiment(df)
    plot_skew(df)
    plot_bimodal(df)


if __name__ == "__main__":
    main()
