"""
실전 1 — 매출 수요 예측 전체 파이프라인.
카페 체인 1년치 매출을 정제·집계하고 요일/계절/시차/이동평균 피처를 만들어
마지막 4주를 시간 기반 백테스트로 평가하고 '다음 주' 7일을 자세히 봅니다.
기준선(어제와 같다 / 지난주 같은 요일)과 비교하고 예측 vs 실제 PNG 를 저장합니다.
"""

import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
BACKTEST_DAYS = 28     # 평가 구간: 마지막 4주 (7일만으로는 운이 많이 섞임)
FOCUS_DAYS = 7         # 표와 그림에서 자세히 볼 '다음 주'
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 66)
    print(" 실전 1: 다음 주 매출 예측 파이프라인 (sales_table)")
    print("=" * 66)

    # [1] 정제 + 일별 집계 -------------------------------------------------
    raw = pd.DataFrame(hjh_data.sales_table(n_days=365, seed=42))
    n0 = len(raw)
    raw = raw.dropna(subset=["revenue"])
    raw = raw[raw["revenue"] > 0]
    daily = raw.groupby("day_index")["revenue"].sum().reset_index()
    print(f"\n[1] 정제 {n0} -> {len(raw)}행, 일별 집계 {len(daily)}일")
    print(f"    하루 평균 매출 {daily['revenue'].mean():,.0f}원")

    # [2] 시간 피처 만들기 (예측 시점에 알 수 있는 값만!) -------------------
    print("\n[2] 피처 생성: 달력 + 계절 + 지난주 정보 (7일 앞 예측이므로 lag>=7 만 사용)")
    daily["weekday"] = daily["day_index"] % 7            # 0=월 ... 6=일
    daily["is_weekend"] = (daily["weekday"] >= 5).astype(int)
    daily["season_sin"] = np.sin(2 * np.pi * daily["day_index"] / 365)
    daily["season_cos"] = np.cos(2 * np.pi * daily["day_index"] / 365)
    daily["lag_7"] = daily["revenue"].shift(7)           # 지난주 같은 요일
    daily["ma7_prev"] = daily["revenue"].shift(7).rolling(7).mean()   # 지난주 기준 최근 7일 평균
    daily["ma28_prev"] = daily["revenue"].shift(7).rolling(28).mean() # 지난주 기준 최근 4주 평균
    for wd in range(7):                                  # 요일 원핫
        daily[f"wd_{WEEKDAYS[wd]}"] = (daily["weekday"] == wd).astype(int)
    daily = daily.dropna().reset_index(drop=True)
    features = (["is_weekend", "season_sin", "season_cos", "lag_7", "ma7_prev", "ma28_prev"]
                + [f"wd_{w}" for w in WEEKDAYS])
    print(f"    피처 {len(features)}개: 요일 원핫 7 + 주말/계절 3 + 지난주 lag/이동평균 3")

    # [3] 시간 기반 분할 ----------------------------------------------------
    train = daily.iloc[:-BACKTEST_DAYS]
    test = daily.iloc[-BACKTEST_DAYS:]
    print(f"\n[3] 시간 기반 분할: 학습 {len(train)}일 (day~{train['day_index'].max()}) / "
          f"평가 마지막 {BACKTEST_DAYS}일 (무작위 분할 금지!)")
    print("    (매주 예측을 갱신하는 운영을 가정: 각 날짜의 피처는 7일 전까지의 정보만 사용)")

    # [4] 기준선 vs 모델 ----------------------------------------------------
    print(f"\n[4] 기준선과 모델의 대결 — 최근 {BACKTEST_DAYS}일 백테스트 (MAE / MAPE)")
    base_naive = daily["revenue"].shift(1).iloc[-BACKTEST_DAYS:].to_numpy()  # 기준선1: 어제와 같다
    base_seasonal = test["lag_7"].to_numpy()             # 기준선2: 지난주 같은 요일

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(train[features], train["revenue"])
    pred = model.predict(test[features])

    for name, p in [("기준선1 어제와 같다", base_naive),
                    ("기준선2 지난주 같은 요일", base_seasonal),
                    ("RandomForest 모델", pred)]:
        print(f"    {name:<22} MAE {mean_absolute_error(test['revenue'], p):>11,.0f}원 | "
              f"MAPE {mape(test['revenue'], p):5.2f}%")
    improve = (1 - mean_absolute_error(test["revenue"], pred)
               / mean_absolute_error(test["revenue"], base_seasonal)) * 100
    print(f"    => 모델은 가장 강한 기준선 대비 MAE {improve:+.1f}% "
          f"({'개선' if improve > 0 else '악화'}) — 보고는 항상 기준선 대비로.")

    # [5] 다음 주 일별 예측표 + PNG ----------------------------------------
    print(f"\n[5] '다음 주'(마지막 {FOCUS_DAYS}일) 예측 vs 실제")
    print(f"    {'day':>4} {'요일':>4} {'실제':>12} {'예측':>12} {'오차':>10}")
    focus = test.iloc[-FOCUS_DAYS:]
    focus_pred = pred[-FOCUS_DAYS:]
    for (_, row), p in zip(focus.iterrows(), focus_pred):
        wd = WEEKDAYS[int(row["weekday"])]
        err = p - row["revenue"]
        print(f"    {int(row['day_index']):>4} {wd:>4} {row['revenue']:>12,.0f} "
              f"{p:>12,.0f} {err:>+10,.0f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    recent = daily.iloc[-56:]                            # 최근 8주 흐름
    ax.plot(recent["day_index"], recent["revenue"], label="actual", color="#4477aa")
    ax.plot(test["day_index"], pred, "o--", label="model forecast", color="#cc6677",
            markersize=4)
    ax.plot(focus["day_index"], base_seasonal[-FOCUS_DAYS:], "s:",
            label="seasonal naive (last week)", color="#999933", markersize=4)
    ax.axvline(train["day_index"].max() + 0.5, color="gray", ls="--", lw=1)
    ax.text(train["day_index"].max() + 0.7, ax.get_ylim()[1] * 0.97, "backtest start",
            fontsize=8, va="top")
    ax.set_xlabel("day_index")
    ax.set_ylabel("daily revenue (KRW)")
    ax.set_title("Next-week sales forecast vs actual")
    ax.legend()
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "forecast_vs_actual.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f"\n    그림 저장: {png}")

    # [6] 모델은 무엇을 보고 예측했나 ---------------------------------------
    print("\n[6] 피처 중요도 (상위 6개)")
    imp = sorted(zip(features, model.feature_importances_), key=lambda t: -t[1])[:6]
    for name, v in imp:
        print(f"    {name:<12} {v:.3f} {'#' * int(v * 40)}")
    print("\n    교훈: 수요 예측 = 점장의 직감(요일·계절·추세)을 피처로 번역해")
    print("          매일 자동으로, 일관되게 적용하는 일입니다.")


if __name__ == "__main__":
    main()
