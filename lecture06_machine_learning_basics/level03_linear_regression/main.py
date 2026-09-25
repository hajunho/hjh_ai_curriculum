"""
level03 — 선형회귀: 광고비 -> 매출

카페 체인 매출 데이터에서 '광고비 1만 원당 매출이 얼마나 늘었나'를
최소제곱 직선으로 구합니다. 같은 답을 두 가지 방법으로 계산해 비교합니다.
  (1) numpy 공식: a = Cov(x,y)/Var(x), b = mean(y) - a*mean(x)
  (2) sklearn LinearRegression
산점도 + 회귀선을 outputs/regression.png 로 저장합니다.
"""

import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")   # 화면 없는 환경에서도 그림 저장 가능
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"


def load_daily_store_sales() -> pd.DataFrame:
    """[1] 원본은 (일 x 매장 x 카테고리) 단위 -> (일 x 매장) 단위로 집계.
    결측 revenue 와 음수 오염은 집계 전에 제거합니다 (lecture03 복습)."""
    raw = pd.DataFrame(hjh_data.sales_table(n_days=365, seed=42))
    n_before = len(raw)
    clean = raw.dropna(subset=["revenue"])          # 결측 제거
    clean = clean[clean["revenue"] > 0]             # 음수(입력 오류) 제거
    print(f"    정제: {n_before}행 -> {len(clean)}행 (결측 {raw['revenue'].isna().sum()}건, "
          f"음수 {(raw['revenue'].dropna() <= 0).sum()}건 제거)")
    # ad_cost 는 (일, 매장)마다 한 값이므로 first, revenue 는 카테고리 합계
    daily = (clean.groupby(["day_index", "store"], as_index=False)
                  .agg(ad_cost=("ad_cost", "first"), revenue=("revenue", "sum")))
    return daily


if __name__ == "__main__":
    np.random.seed(0)  # 재현성 (이 레벨은 난수를 쓰지 않지만 관례로 고정)

    # [1] 데이터 준비 ------------------------------------------------------
    print("[1] 데이터 준비 — 카페 체인 365일 x 5개 매장 매출")
    daily = load_daily_store_sales()
    x = daily["ad_cost"].to_numpy(dtype=float)      # 입력: 하루 광고비 (원)
    y = daily["revenue"].to_numpy(dtype=float)      # 정답: 하루 매출 (원)
    print(f"    분석 단위: (일, 매장) {len(daily)}건 / "
          f"광고비 범위 {x.min():,.0f}~{x.max():,.0f}원\n")

    # [2] numpy 공식으로 최소제곱 직선 구하기 -------------------------------
    slope_np = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)  # a = Cov/Var
    intercept_np = y.mean() - slope_np * x.mean()              # (x̄,ȳ)를 지난다
    print("[2] numpy 공식 (Cov/Var) 으로 직접 계산")
    print(f"    기울기 a = {slope_np:.4f}   절편 b = {intercept_np:,.0f}")

    # [3] sklearn 으로 같은 문제 풀기 ---------------------------------------
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)   # sklearn 은 2차원 입력 (행=샘플, 열=특징)
    slope_sk, intercept_sk = model.coef_[0], model.intercept_
    print("[3] sklearn LinearRegression")
    print(f"    기울기 a = {slope_sk:.4f}   절편 b = {intercept_sk:,.0f}")
    same = np.isclose(slope_np, slope_sk) and np.isclose(intercept_np, intercept_sk)
    print(f"    두 방법의 결과 일치? {same} — 라이브러리는 같은 공식의 포장입니다.\n")

    # [4] 비즈니스 해석 ----------------------------------------------------
    y_hat = model.predict(x.reshape(-1, 1))
    r2 = r2_score(y, y_hat)
    print("[4] 비즈니스 해석")
    print(f"    광고비 1만 원당 매출 +{slope_sk * 10_000:,.0f}원의 연관 (상관이지 인과 증명은 아님!)")
    print(f"    R^2 = {r2:.3f} -> 매출 출렁임의 {r2:.1%}를 광고비 하나로 설명")
    ad = 300_000
    pred = model.predict([[ad]])[0]
    print(f"    광고비 {ad:,}원인 날의 예측 매출: {pred:,.0f}원")
    print("    (주의: 전체 데이터로 학습하고 채점한 낙관적 성적 -> level04)\n")

    # [5] 산점도 + 회귀선 저장 ----------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x / 10_000, y / 10_000, s=8, alpha=0.3, label="daily data")
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs / 10_000, (slope_sk * xs + intercept_sk) / 10_000,
            color="crimson", linewidth=2,
            label=f"y = {slope_sk:.2f}x + {intercept_sk/10_000:,.0f}")
    ax.set_xlabel("ad cost (10k KRW)")
    ax.set_ylabel("daily revenue (10k KRW)")
    ax.set_title("Ad cost vs daily revenue (least squares fit)")
    ax.legend()
    fig.tight_layout()
    png = OUT_DIR / "regression.png"
    fig.savefig(png, dpi=120)
    print(f"[5] 그림 저장: {png}")
    print("    산점도의 흩어짐(노이즈) 속에서 직선이 '평균적 경향'을 요약합니다.")
