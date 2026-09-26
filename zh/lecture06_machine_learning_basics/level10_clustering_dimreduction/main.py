"""
level10 — 聚类与降维: 没有答案的客户细分

用合成客户数据(暗藏 4 种类型, 对模型保密)
  [2] k-means + 肘部/轮廓系数挑选簇数 k
  [3] 各簇画像表 + 起名字(解读是人的活儿)
  [4] 用 PCA 把 4 个变量 -> 2 维, 读解释方差与轴的含义(loading)
  [5] 在 PCA 2D 地图上给簇着色并保存 PNG
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
COLS_ZH = {"annual_spend": "年消费额(万韩元)", "visits_per_month": "月到店数",
           "discount_rate": "折扣使用率", "tenure_months": "入网月数"}


def make_customers(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """暗藏 4 种类型的客户数据。类型信息扔掉(复现无监督处境)。"""
    X, _hidden = make_blobs(n_samples=n, centers=4, n_features=4,
                            cluster_std=1.1, random_state=seed)
    # 把标准正态尺度的 blob 换算成现实量纲
    df = pd.DataFrame({
        "annual_spend": np.clip(300 + 60 * X[:, 0], 30, None).round(0),      # 万韩元
        "visits_per_month": np.clip(6 + 1.6 * X[:, 1], 0.2, None).round(1),
        "discount_rate": np.clip(0.35 + 0.09 * X[:, 2], 0.0, 0.95).round(2),
        "tenure_months": np.clip(24 + 7 * X[:, 3], 1, None).round(0),
    })
    return df


def suggest_name(profile: pd.Series, overall: pd.Series) -> str:
    """把簇平均画像和整体平均对比，自动提议名字候选。
    (实务中这一步是人们讨论的话题)"""
    tags = []
    if profile["annual_spend"] > overall["annual_spend"] * 1.15:
        tags.append("高消费")
    elif profile["annual_spend"] < overall["annual_spend"] * 0.85:
        tags.append("低消费")
    if profile["visits_per_month"] > overall["visits_per_month"] * 1.15:
        tags.append("常客")
    elif profile["visits_per_month"] < overall["visits_per_month"] * 0.85:
        tags.append("到店稀少")
    if profile["discount_rate"] > overall["discount_rate"] * 1.15:
        tags.append("折扣敏感")
    if profile["tenure_months"] < overall["tenure_months"] * 0.7:
        tags.append("新客")
    return "·".join(tags) if tags else "平均型"


if __name__ == "__main__":
    np.random.seed(42)

    # [1] 数据 -------------------------------------------------------------
    df = make_customers()
    print(f"[1] 合成客户 {len(df)} 名 — 暗藏 4 种类型 (对模型保密)")
    print(df.head(3).to_string(index=False))
    print()

    # [2] 挑 k: 肘部 + 轮廓系数 ---------------------------------------------
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df[COLS])       # 基于距离，所以必须标准化!
    print("[2] 挑选簇数 k (inertia=总距离, silhouette=分离质量)")
    print("     k    inertia    silhouette")
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
        sil = silhouette_score(Xs, km.labels_)
        print(f"     {k}   {km.inertia_:8.1f}      {sil:.3f}")
    print("    -> 选 inertia 下降变缓(肘部)且轮廓系数高的 k=4。")
    print("       最终决定不靠数学，靠'这个数能不能拆出不同行动'的业务判断。\n")

    # [3] k=4 聚类 + 画像解读 --------------------------------------------
    km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(Xs)
    df["cluster"] = km.labels_
    overall = df[COLS].mean()
    profile = df.groupby("cluster")[COLS].mean()
    sizes = df["cluster"].value_counts().sort_index()
    print("[3] 各簇画像 (平均) — 机器只给编号, 名字由人来起")
    header = "    簇  人数   " + "  ".join(f"{COLS_ZH[c]:>10s}" for c in COLS) + "   提议名字"
    print(header)
    for c in profile.index:
        row = profile.loc[c]
        vals = "  ".join(f"{row[col]:10.1f}" for col in COLS)
        print(f"     {c}   {sizes[c]:>4}   {vals}   {suggest_name(row, overall)}")
    print("    -> 名字起出来了行动才跟得上: 折扣敏感型->优惠券, 高消费常客->专属权益 等\n")

    # [4] PCA: 4 维 -> 2 维地图 --------------------------------------------
    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(Xs)
    evr = pca.explained_variance_ratio_
    print("[4] PCA — 挑影子的角度")
    print(f"    PC1 解释方差 {evr[0]:.1%}, PC2 解释方差 {evr[1]:.1%} "
          f"(2 维地图保留了原始信息的 {evr.sum():.1%})")
    print("    轴的成分(loading): 每个主成分是原始变量的加权和")
    print("    变量               PC1      PC2")
    for i, c in enumerate(COLS):
        print(f"    {COLS_ZH[c]:12s} {pca.components_[0][i]:+7.2f}  {pca.components_[1][i]:+7.2f}")
    print("    -> 看权重的正负/大小给轴起名字来读 (要谨慎)。\n")

    # [5] 在 2D 地图上给簇着色 --------------------------------------------------
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
    print(f"[5] 图片已保存: {png}")
    print("    地图上簇真的分开了，就是细分抓住了结构的视觉证据。")
