"""
level10 — Clustering and dimensionality reduction: customer segmentation with no answers

On synthetic customer data (4 hidden types, kept secret from the model):
  [2] k-means + elbow/silhouette to choose the cluster count k
  [3] per-cluster profile table + naming (interpretation is the human's job)
  [4] PCA: 4 variables -> 2-D; reading explained variance and axis meaning (loadings)
  [5] save a PNG of the clusters colored on the PCA 2-D map
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
COLS_KO = {"annual_spend": "annual spend (10k KRW)", "visits_per_month": "visits/month",
           "discount_rate": "discount rate", "tenure_months": "tenure (months)"}


def make_customers(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """Customer data with 4 hidden types. We throw the type info away
    (recreating the unsupervised situation)."""
    X, _hidden = make_blobs(n_samples=n, centers=4, n_features=4,
                            cluster_std=1.1, random_state=seed)
    # Convert the standard-normal-scale blobs to realistic scales
    df = pd.DataFrame({
        "annual_spend": np.clip(300 + 60 * X[:, 0], 30, None).round(0),      # 10k KRW
        "visits_per_month": np.clip(6 + 1.6 * X[:, 1], 0.2, None).round(1),
        "discount_rate": np.clip(0.35 + 0.09 * X[:, 2], 0.0, 0.95).round(2),
        "tenure_months": np.clip(24 + 7 * X[:, 3], 1, None).round(0),
    })
    return df


def suggest_name(profile: pd.Series, overall: pd.Series) -> str:
    """Auto-suggest name candidates by comparing a cluster's mean profile
    to the overall mean. (In real work, this step is what teams debate.)"""
    tags = []
    if profile["annual_spend"] > overall["annual_spend"] * 1.15:
        tags.append("big spenders")
    elif profile["annual_spend"] < overall["annual_spend"] * 0.85:
        tags.append("light spenders")
    if profile["visits_per_month"] > overall["visits_per_month"] * 1.15:
        tags.append("regulars")
    elif profile["visits_per_month"] < overall["visits_per_month"] * 0.85:
        tags.append("infrequent visitors")
    if profile["discount_rate"] > overall["discount_rate"] * 1.15:
        tags.append("discount-sensitive")
    if profile["tenure_months"] < overall["tenure_months"] * 0.7:
        tags.append("newcomers")
    return " · ".join(tags) if tags else "average type"


if __name__ == "__main__":
    np.random.seed(42)

    # [1] Data -------------------------------------------------------------------
    df = make_customers()
    print(f"[1] {len(df)} synthetic customers — 4 hidden types (secret from the model)")
    print(df.head(3).to_string(index=False))
    print()

    # [2] Choosing k: elbow + silhouette --------------------------------------------
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df[COLS])       # distance-based, so standardization is mandatory!
    print("[2] Choosing the cluster count k (inertia=total distance, silhouette=separation quality)")
    print("     k    inertia    silhouette")
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
        sil = silhouette_score(Xs, km.labels_)
        print(f"     {k}   {km.inertia_:8.1f}      {sil:.3f}")
    print("    -> We pick k=4, where the inertia decrease bends (elbow) and the silhouette is high.")
    print("       The final call is not math but the business judgment 'does that number split into actions?'\n")

    # [3] k=4 clustering + profile interpretation ------------------------------------
    km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(Xs)
    df["cluster"] = km.labels_
    overall = df[COLS].mean()
    profile = df.groupby("cluster")[COLS].mean()
    sizes = df["cluster"].value_counts().sort_index()
    print("[3] Per-cluster profile (means) — the machine gives numbers; humans give names")
    header = "    cluster  size   " + "  ".join(f"{COLS_KO[c]:>22s}" for c in COLS) + "   suggested name"
    print(header)
    for c in profile.index:
        row = profile.loc[c]
        vals = "  ".join(f"{row[col]:22.1f}" for col in COLS)
        print(f"       {c}     {sizes[c]:>4}   {vals}   {suggest_name(row, overall)}")
    print("    -> Only named groups produce actions: discount-sensitive -> coupons,")
    print("       big-spending regulars -> exclusive perks, etc.\n")

    # [4] PCA: 4-D -> a 2-D map -------------------------------------------------------
    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(Xs)
    evr = pca.explained_variance_ratio_
    print("[4] PCA — choosing the shadow angle")
    print(f"    PC1 explains {evr[0]:.1%} of variance, PC2 {evr[1]:.1%} "
          f"(the 2-D map preserves {evr.sum():.1%} of the original information)")
    print("    Axis composition (loadings): each PC is a weighted sum of the original variables")
    print("    Variable                     PC1      PC2")
    for i, c in enumerate(COLS):
        print(f"    {COLS_KO[c]:24s} {pca.components_[0][i]:+7.2f}  {pca.components_[1][i]:+7.2f}")
    print("    -> Read the signs/sizes of the weights and (cautiously) name the axes.\n")

    # [5] Coloring the clusters on the 2-D map -------------------------------------------
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
    print(f"[5] Figure saved: {png}")
    print("    If the clusters look genuinely separated on the map, that's visual evidence")
    print("    the segmentation captured real structure.")
