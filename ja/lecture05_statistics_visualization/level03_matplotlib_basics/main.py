"""
level03 — Matplotlib の基本グラフ

figure/axes 構造に沿って、カフェチェーンの売上から
折れ線グラフ (月別推移) · 棒グラフ (店舗比較) · 散布図 (広告費-売上) を
outputs/ 以下に PNG として保存します。日本語フォントの自動設定関数も含みます。

※ 図の中のテキストは、環境によるフォントの文字化けを避けるため英語で統一しています。
   (ターミナル出力は日本語です。)
"""

import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # 必ず pyplot の import「前」に — 画面の代わりにファイルへ保存
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# 図中で使う店舗名の英語表記 (日本語フォントがない環境でも崩れないように)
STORE_EN = {"渋谷店": "Shibuya", "新宿店": "Shinjuku", "大阪店": "Osaka",
            "名古屋店": "Nagoya", "福岡店": "Fukuoka"}


def set_japanese_font() -> None:
    """macOS/Windows/Linux の代表的な日本語フォントを順に探して登録します。

    この実習の図は英語ラベルなので必須ではありませんが、
    図中に日本語を書きたいプロジェクトへそのままコピーして使えます。
    """
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["Hiragino Sans", "Yu Gothic", "Noto Sans CJK JP", "IPAexGothic"]:
        if cand in names:
            plt.rcParams["font.family"] = cand
            print(f"[0] 日本語フォントを設定: {cand}")
            break
    else:
        print("[0] 日本語フォントが見つからないため既定フォントを使います (図は英語ラベルなので問題ありません)")
    plt.rcParams["axes.unicode_minus"] = False


def load_sales() -> pd.DataFrame:
    """売上データのクレンジング + 月カラムの追加。"""
    rows = hjh_data.sales_table(n_days=180, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0].copy()
    df["month"] = df["date"].str[:7]            # '2025-03' の形
    print(f"[1] データ準備完了: {len(df):,}行、期間 {df['date'].min()} ~ {df['date'].max()}")
    return df


def plot_line_monthly(df: pd.DataFrame) -> None:
    """折れ線グラフ: 時間による変化は線で。"""
    monthly = df.groupby("month")["revenue"].sum() / 1e8  # 億ウォン単位

    fig, ax = plt.subplots(figsize=(8, 4.5))              # 用紙 + 段落
    ax.plot(monthly.index, monthly.values,                # 文章 (線を引く)
            marker="o", color="#4878cf", lw=2)
    for x, y in monthly.items():                          # 値ラベル
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_title("Monthly Total Revenue Trend")           # 小見出し
    ax.set_xlabel("Month")
    ax.set_ylabel("Total revenue (100M KRW)")
    ax.set_ylim(0, monthly.max() * 1.2)                   # 0 から始める (level04 の予告)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "line_monthly.png")
    fig.savefig(path, dpi=120)                            # 提出 (保存)
    plt.close(fig)                                        # 用紙を片付ける
    print(f"[2] 折れ線グラフを保存: {path}")


def plot_bar_stores(df: pd.DataFrame) -> None:
    """棒グラフ: カテゴリ比較は棒で。トップだけ色で強調。"""
    stores = (df.groupby("store")["revenue"].sum() / 1e8).sort_values(ascending=False)
    colors = ["#d1495b" if i == 0 else "#9aa7b5" for i in range(len(stores))]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [STORE_EN.get(s, s) for s in stores.index]
    ax.bar(labels, stores.values, color=colors)
    ax.set_title("Total Revenue by Store (highlight = top store)")
    ax.set_xlabel("Store")
    ax.set_ylabel("Total revenue (100M KRW)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "bar_stores.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[3] 棒グラフを保存: {path} (1位: {stores.index[0]})")


def plot_scatter_ad(df: pd.DataFrame) -> None:
    """散布図: 2 つの数値の関係は点で。(2 つ先のレベルの予告編)"""
    daily = df.groupby("day_index").agg(
        ad=("ad_cost", "sum"), rev=("revenue", "sum"))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(daily["ad"] / 1e6, daily["rev"] / 1e8,
               s=18, alpha=0.5, color="#2e6f40")
    ax.set_title("Daily Ad Spend vs Daily Revenue")
    ax.set_xlabel("Ad spend (million KRW)")
    ax.set_ylabel("Revenue (100M KRW)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "scatter_ad.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[4] 散布図を保存: {path}")
    print("    点の雲が右上がりに見えますか? この「関係」は level05 で掘り下げます。")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    set_japanese_font()
    df = load_sales()
    plot_line_monthly(df)
    plot_bar_stores(df)
    plot_scatter_ad(df)
    print("[5] すべてのグラフは「用紙->段落->文章->小見出し->保存」の 5 段階で作られました。")


if __name__ == "__main__":
    main()
