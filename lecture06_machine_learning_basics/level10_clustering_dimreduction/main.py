"""
level10 — 군집화와 차원축소: 정답 없는 고객 세분화

합성 고객 데이터(숨은 4개 유형, 모델에게는 비밀)로
  [2] k-means + 엘보/실루엣으로 군집 수 k 고르기
  [3] 군집별 프로필 표 + 이름 붙이기(해석은 사람의 몫)
  [4] PCA 로 4개 변수 -> 2차원, 설명 분산과 축의 의미(loading) 읽기
  [5] PCA 2D 지도에 군집을 색칠한 PNG 저장
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
COLS_KO = {"annual_spend": "연 구매액(만원)", "visits_per_month": "월 방문수",
           "discount_rate": "할인 사용률", "tenure_months": "가입 개월"}


def make_customers(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """숨은 4개 유형을 가진 고객 데이터. 유형 정보는 버립니다(비지도 상황 재현)."""
    X, _hidden = make_blobs(n_samples=n, centers=4, n_features=4,
                            cluster_std=1.1, random_state=seed)
    # 표준정규 스케일의 blob 을 현실적인 척도로 변환
    df = pd.DataFrame({
        "annual_spend": np.clip(300 + 60 * X[:, 0], 30, None).round(0),      # 만원
        "visits_per_month": np.clip(6 + 1.6 * X[:, 1], 0.2, None).round(1),
        "discount_rate": np.clip(0.35 + 0.09 * X[:, 2], 0.0, 0.95).round(2),
        "tenure_months": np.clip(24 + 7 * X[:, 3], 1, None).round(0),
    })
    return df


def suggest_name(profile: pd.Series, overall: pd.Series) -> str:
    """군집 평균 프로필을 전체 평균과 비교해 이름 후보를 자동 제안.
    (실무에서는 이 단계가 사람의 토론 거리입니다)"""
    tags = []
    if profile["annual_spend"] > overall["annual_spend"] * 1.15:
        tags.append("고소비")
    elif profile["annual_spend"] < overall["annual_spend"] * 0.85:
        tags.append("저소비")
    if profile["visits_per_month"] > overall["visits_per_month"] * 1.15:
        tags.append("단골")
    elif profile["visits_per_month"] < overall["visits_per_month"] * 0.85:
        tags.append("뜸한 방문")
    if profile["discount_rate"] > overall["discount_rate"] * 1.15:
        tags.append("할인 민감")
    if profile["tenure_months"] < overall["tenure_months"] * 0.7:
        tags.append("신규")
    return "·".join(tags) if tags else "평균형"


if __name__ == "__main__":
    np.random.seed(42)

    # [1] 데이터 -------------------------------------------------------------
    df = make_customers()
    print(f"[1] 합성 고객 {len(df)}명 — 숨은 유형 4개 (모델에게는 비밀)")
    print(df.head(3).to_string(index=False))
    print()

    # [2] k 고르기: 엘보 + 실루엣 ---------------------------------------------
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df[COLS])       # 거리 기반이므로 표준화 필수!
    print("[2] 군집 수 k 고르기 (inertia=총 이동거리, silhouette=분리 품질)")
    print("     k    inertia    silhouette")
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
        sil = silhouette_score(Xs, km.labels_)
        print(f"     {k}   {km.inertia_:8.1f}      {sil:.3f}")
    print("    -> inertia 감소가 꺾이고(엘보) 실루엣이 높은 k=4 를 선택.")
    print("       최종 결정은 수학이 아니라 '그 수로 액션이 나뉘는가'라는 업무 판단.\n")

    # [3] k=4 군집화 + 프로필 해석 --------------------------------------------
    km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(Xs)
    df["cluster"] = km.labels_
    overall = df[COLS].mean()
    profile = df.groupby("cluster")[COLS].mean()
    sizes = df["cluster"].value_counts().sort_index()
    print("[3] 군집별 프로필 (평균) — 기계는 번호만 주고, 이름은 사람이 붙입니다")
    header = "    군집  인원   " + "  ".join(f"{COLS_KO[c]:>10s}" for c in COLS) + "   제안 이름"
    print(header)
    for c in profile.index:
        row = profile.loc[c]
        vals = "  ".join(f"{row[col]:10.1f}" for col in COLS)
        print(f"     {c}   {sizes[c]:>4}   {vals}   {suggest_name(row, overall)}")
    print("    -> 이름이 붙어야 액션이 나옵니다: 할인 민감형->쿠폰, 고소비 단골->전용 혜택 등\n")

    # [4] PCA: 4차원 -> 2차원 지도 --------------------------------------------
    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(Xs)
    evr = pca.explained_variance_ratio_
    print("[4] PCA — 그림자 각도 고르기")
    print(f"    PC1 설명 분산 {evr[0]:.1%}, PC2 설명 분산 {evr[1]:.1%} "
          f"(2차원 지도에 원본 정보의 {evr.sum():.1%} 보존)")
    print("    축의 성분(loading): 각 주성분은 원본 변수의 가중합")
    print("    변수               PC1      PC2")
    for i, c in enumerate(COLS):
        print(f"    {COLS_KO[c]:12s} {pca.components_[0][i]:+7.2f}  {pca.components_[1][i]:+7.2f}")
    print("    -> 가중치 부호/크기를 보고 축에 이름을 붙여 읽습니다 (조심스럽게).\n")

    # [5] 2D 지도에 군집 색칠 --------------------------------------------------
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
    print(f"[5] 그림 저장: {png}")
    print("    지도에서 군집이 실제로 분리되어 보이면, 세분화가 구조를 잡았다는 시각적 증거.")
