"""
level10 — クラスタリングと次元削減: 正解のない顧客セグメンテーション

合成顧客データ (隠れた 4 つのタイプ、モデルには秘密) で
  [2] k-means + エルボー/シルエットでクラスタ数 k を選ぶ
  [3] クラスタ別プロフィール表 + 名前付け (解釈は人間の役目)
  [4] PCA で 4 変数 -> 2 次元、説明分散と軸の意味 (loading) を読む
  [5] PCA 2D 地図にクラスタを色分けした PNG を保存
"""

import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
COLS = ["annual_spend", "visits_per_month", "discount_rate", "tenure_months"]
COLS_JA = {"annual_spend": "年間購入額(万ウォン)", "visits_per_month": "月間訪問数",
           "discount_rate": "割引利用率", "tenure_months": "会員歴(月)"}


def make_customers(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """隠れた 4 タイプを持つ顧客データ。タイプ情報は捨てます (教師なし状況の再現)。"""
    X, _hidden = make_blobs(n_samples=n, centers=4, n_features=4,
                            cluster_std=1.1, random_state=seed)
    # 標準正規スケールの blob を現実的な尺度に変換
    df = pd.DataFrame({
        "annual_spend": np.clip(300 + 60 * X[:, 0], 30, None).round(0),      # 万ウォン
        "visits_per_month": np.clip(6 + 1.6 * X[:, 1], 0.2, None).round(1),
        "discount_rate": np.clip(0.35 + 0.09 * X[:, 2], 0.0, 0.95).round(2),
        "tenure_months": np.clip(24 + 7 * X[:, 3], 1, None).round(0),
    })
    return df


def suggest_name(profile: pd.Series, overall: pd.Series) -> str:
    """クラスタの平均プロフィールを全体平均と比べて名前の候補を自動提案。
    (実務ではこの段階が人間の議論のタネになります)"""
    tags = []
    if profile["annual_spend"] > overall["annual_spend"] * 1.15:
        tags.append("高消費")
    elif profile["annual_spend"] < overall["annual_spend"] * 0.85:
        tags.append("低消費")
    if profile["visits_per_month"] > overall["visits_per_month"] * 1.15:
        tags.append("常連")
    elif profile["visits_per_month"] < overall["visits_per_month"] * 0.85:
        tags.append("訪問まばら")
    if profile["discount_rate"] > overall["discount_rate"] * 1.15:
        tags.append("割引に敏感")
    if profile["tenure_months"] < overall["tenure_months"] * 0.7:
        tags.append("新規")
    return "・".join(tags) if tags else "平均型"


if __name__ == "__main__":
    np.random.seed(42)

    # [1] データ -------------------------------------------------------------
    df = make_customers()
    print(f"[1] 合成顧客 {len(df)}人 — 隠れたタイプは 4 つ (モデルには秘密)")
    print(df.head(3).to_string(index=False))
    print()

    # [2] k を選ぶ: エルボー + シルエット ---------------------------------------------
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df[COLS])       # 距離ベースなので標準化は必須!
    print("[2] クラスタ数 k を選ぶ (inertia=総移動距離, silhouette=分離の品質)")
    print("     k    inertia    silhouette")
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
        sil = silhouette_score(Xs, km.labels_)
        print(f"     {k}   {km.inertia_:8.1f}      {sil:.3f}")
    print("    -> inertia の減少が折れ (エルボー)、シルエットの高い k=4 を選択。")
    print("       最終決定は数学ではなく『その数でアクションが分けられるか』という業務判断。\n")

    # [3] k=4 のクラスタリング + プロフィール解釈 --------------------------------------------
    km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(Xs)
    df["cluster"] = km.labels_
    overall = df[COLS].mean()
    profile = df.groupby("cluster")[COLS].mean()
    sizes = df["cluster"].value_counts().sort_index()
    print("[3] クラスタ別プロフィール (平均) — 機械は番号だけをくれる。名前は人間が付けます")
    header = "    クラスタ  人数   " + "  ".join(f"{COLS_JA[c]:>10s}" for c in COLS) + "   提案する名前"
    print(header)
    for c in profile.index:
        row = profile.loc[c]
        vals = "  ".join(f"{row[col]:10.1f}" for col in COLS)
        print(f"     {c}   {sizes[c]:>4}   {vals}   {suggest_name(row, overall)}")
    print("    -> 名前が付いて初めてアクションが生まれます: 割引に敏感型->クーポン、高消費の常連->専用特典 など\n")

    # [4] PCA: 4 次元 -> 2 次元の地図 --------------------------------------------
    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(Xs)
    evr = pca.explained_variance_ratio_
    print("[4] PCA — 影の角度を選ぶ")
    print(f"    PC1 の説明分散 {evr[0]:.1%}, PC2 の説明分散 {evr[1]:.1%} "
          f"(2 次元の地図に元の情報の {evr.sum():.1%} を保存)")
    print("    軸の成分 (loading): 各主成分は元の変数の加重和")
    print("    変数               PC1      PC2")
    for i, c in enumerate(COLS):
        print(f"    {COLS_JA[c]:12s} {pca.components_[0][i]:+7.2f}  {pca.components_[1][i]:+7.2f}")
    print("    -> 重みの符号/大きさを見て軸に名前を付けて読みます (慎重に)。\n")

    # [5] 2D 地図にクラスタを色分け --------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in sorted(df["cluster"].unique()):
        pts = XY[df["cluster"] == c]
        ax.scatter(pts[:, 0], pts[:, 1], s=14, alpha=0.65, label=f"cluster {c}")
    cent = pca.transform(km.cluster_centers_)
    ax.scatter(cent[:, 0], cent[:, 1], marker="X", s=180, c="black", label="centers")
    ax.set_xlabel(f"PC1 ({evr[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({evr[1]:.0%} var)")
    ax.set_title("Customer segments on PCA 2D map")
    ax.legend()
    fig.tight_layout()
    png = OUT_DIR / "segments_pca.png"
    fig.savefig(png, dpi=120)
    print(f"[5] 図を保存: {png}")
    print("    地図の上でクラスタが実際に分かれて見えれば、セグメンテーションが構造を捉えた視覚的証拠。")
