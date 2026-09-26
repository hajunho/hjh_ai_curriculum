"""
level04 — 良いグラフ vs 悪いグラフ

完全に同じデータで「歪曲チャート」と「誠実なチャート」を並べて描きます。
1) 軸の切り詰めで小さな差を圧倒的な格差のように見せる
2) 3D 風の円グラフ vs 並べ替えた横棒
3) y 軸の範囲操作で微細な上下動をジェットコースターにする

※ 図中のテキストは、フォントの文字化けを避けるため英語にしています。
"""

import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# 図中で使う店舗名の英語表記
STORE_EN = {"渋谷店": "Shibuya", "新宿店": "Shinjuku", "大阪店": "Osaka",
            "名古屋店": "Nagoya", "福岡店": "Fukuoka"}


def setup_plot_style() -> None:
    """図中のラベルは英語なので、マイナス記号の設定だけ整えます。"""
    plt.rcParams["axes.unicode_minus"] = False


def load_data():
    rows = hjh_data.sales_table(n_days=90, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0]
    store_rev = (df.groupby("store")["revenue"].sum() / 1e8)  # 億ウォン
    return df, store_rev


def crime1_truncated_axis(store_rev: pd.Series) -> None:
    """罪状 1: 棒グラフの y 軸切り詰め。"""
    two = store_rev.sort_values(ascending=False).iloc[[1, 2]]  # 差の小さい 2 店舗
    a, b = two.index
    gap_pct = (two[a] - two[b]) / two[b] * 100

    fig, (bad, good) = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["#d1495b", "#9aa7b5"]
    labels = [STORE_EN.get(s, s) for s in two.index]

    bad.bar(labels, two.values, color=colors)
    bad.set_ylim(two.min() * 0.985, two.max() * 1.005)   # 軸の切り詰め!
    bad.set_title(f"[Distorted] Truncated axis — {gap_pct:.1f}% gap looks like a landslide")
    bad.set_ylabel("Total revenue (100M KRW)")

    good.bar(labels, two.values, color=colors)
    good.set_ylim(0, two.max() * 1.15)                   # 0 から始める
    good.set_title("[Honest] Bars start at zero — the real gap")
    good.set_ylabel("Total revenue (100M KRW)")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "truncated_axis.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[2] 軸切り詰めの比較を保存: {path}")
    print(f"    実際の差はわずか {gap_pct:.1f}% です。左の図の第一印象と比べてみてください。")


def crime2_pie_vs_bar(store_rev: pd.Series) -> None:
    """罪状 2: 似た値どうしの円グラフ vs 並べ替えた横棒。"""
    share = store_rev / store_rev.sum() * 100

    fig, (bad, good) = plt.subplots(1, 2, figsize=(11, 4.8))
    # 影 + 飛び出した一切れ = 会議室の 3D 円グラフの再現
    bad.pie(share.values, labels=[STORE_EN.get(s, s) for s in share.index],
            shadow=True,
            explode=[0.08 if i == 0 else 0 for i in range(len(share))],
            startangle=90)
    bad.set_title("[Distorted] Pie — can you tell which slice is 2nd?")

    ordered = share.sort_values()
    good.barh([STORE_EN.get(s, s) for s in ordered.index], ordered.values,
              color="#4878cf")
    for y, v in enumerate(ordered.values):
        good.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=9)
    good.set_title("[Honest] Sorted bars — rank and gap at a glance")
    good.set_xlabel("Revenue share (%)")
    good.set_xlim(0, ordered.max() * 1.25)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "pie_vs_bar.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[3] 円グラフ vs 棒グラフを保存: {path}")


def crime3_inflated_line(df: pd.DataFrame) -> None:
    """罪状 3: y 軸の範囲を狭めて微細な上下動をジェットコースターに。"""
    daily = df.groupby("day_index")["revenue"].sum() / 1e8
    weekly = daily.rolling(7).mean().dropna()            # 7 日移動平均でなだらかに

    fig, (bad, good) = plt.subplots(1, 2, figsize=(11, 4.2))
    bad.plot(weekly.index, weekly.values, color="#d1495b", lw=2)
    bad.set_ylim(weekly.min() * 0.998, weekly.max() * 1.002)  # 範囲の操作!
    bad.set_title("[Distorted] Zoomed axis — crash-and-rebound drama?")
    bad.set_xlabel("Day")
    bad.set_ylabel("7-day avg daily revenue (100M KRW)")

    good.plot(weekly.index, weekly.values, color="#4878cf", lw=2)
    good.set_ylim(0, weekly.max() * 1.2)
    good.set_title("[Honest] Contextual range — essentially stable")
    good.set_xlabel("Day")
    good.set_ylabel("7-day avg daily revenue (100M KRW)")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "inflated_line.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    swing = (weekly.max() - weekly.min()) / weekly.mean() * 100
    print(f"[4] 折れ線グラフの範囲操作を保存: {path}")
    print(f"    実際の振れ幅は平均に対して ±{swing / 2:.1f}% 程度です。")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_plot_style()
    df, store_rev = load_data()

    print("[1] 材料になる実際の数字 (店舗別総売上、億ウォン)")
    for store, v in store_rev.sort_values(ascending=False).items():
        print(f"    {store:<6} {v:8.2f}")
    print("    -> 下の 3 つの図の「歪曲」と「誠実」は、すべてこの同じ数字から生まれます。")

    crime1_truncated_axis(store_rev)
    crime2_pie_vs_bar(store_rev)
    crime3_inflated_line(df)
    print("[5] 結論: グラフの第一印象が数字を精読した結論と違うなら、そのグラフは悪いグラフです。")


if __name__ == "__main__":
    main()
