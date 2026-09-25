"""
모델을 의사결정으로 — '누구에게 쿠폰을 줄까' 이익 극대화 실험.
이탈 확률에 기대가치 계산(p*V*s - C)을 얹어
이론 손익분기 임계값 p* = C/(V*s) 와 실험 최적 임계값을 교차 검증하고,
이익 곡선 PNG 와 캠페인 ROI 시나리오 보고를 만듭니다.
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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# ---- 비즈니스 단가 (level01 과 동일) ----
VALUE_V = 179_000   # 고객 잔존 가치 (원)
COST_C = 12_000     # 개입 비용: 쿠폰 + 상담 (원)
SUCCESS_S = 0.30    # 개입 성공률

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


def realized_profit(proba, y_true, threshold, s=SUCCESS_S):
    """임계값 이상 고객에게 개입했을 때 테스트 데이터에서의 실현 이익.
    실제 이탈 예정 고객이면 성공률 s 만큼 V 를 지키고, 아니면 쿠폰 비용만 나감."""
    target = proba >= threshold
    y = np.asarray(y_true)
    tp = int((target & (y == 1)).sum())     # 이탈 예정 고객에게 개입
    fp = int((target & (y == 0)).sum())     # 잔류 고객에게 개입 (비용 낭비)
    return tp * (VALUE_V * s - COST_C) - fp * COST_C, int(target.sum())


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 68)
    print(" 모델을 의사결정으로: 이익을 극대화하는 쿠폰 임계값 찾기")
    print("=" * 68)

    # [1] 이탈 모델과 확률 --------------------------------------------------
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES], df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=42, stratify=y)
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    print(f"\n[1] 이탈 모델 훈련 완료. 테스트 {len(y_te)}명의 이탈 확률 산출")
    print(f"    (참고: class_weight 없이 훈련 — 기대가치 계산에는 보정된 확률이 필요)")

    # [2] 이론 손익분기 확률 p* ---------------------------------------------
    p_star = COST_C / (VALUE_V * SUCCESS_S)
    print("\n[2] 이론 손익분기 확률 (산수 한 줄)")
    print(f"    개입 기대이익 = p*V*s - C > 0  <=>  p > C/(V*s)")
    print(f"    p* = {COST_C:,} / ({VALUE_V:,} x {SUCCESS_S:.0%}) = {p_star:.3f}")
    print(f"    => 이탈 확률 {p_star:.1%} 초과 고객에게만 쿠폰을 주는 것이 이론 최적.")
    print("       관습적 임계값 0.5 에는 아무 근거가 없습니다.")

    # [3] 실험: 임계값 훑기 --------------------------------------------------
    print("\n[3] 실험: 임계값 0 -> 1 을 훑으며 테스트 실현 이익 계산")
    grid = np.arange(0.0, 1.001, 0.01)
    profits = np.array([realized_profit(proba, y_te, t)[0] for t in grid])
    best_i = int(np.argmax(profits))
    best_th = float(grid[best_i])
    print(f"    실험 최적 임계값 = {best_th:.2f} (실현 이익 {profits[best_i]:+,.0f}원)")
    print(f"    이론값 {p_star:.3f} 과의 거리 = {abs(best_th - p_star):.3f}")
    print("    => 둘이 가까우면 확률 보정이 쓸 만하다는 뜻. 멀면 보정부터 점검!")

    # [4] 전략 비교표 --------------------------------------------------------
    print("\n[4] 전략 비교 (테스트 고객 기준 실현 이익)")
    strategies = [
        ("A. 아무것도 안 함", 1.01),
        ("B. 전원 쿠폰", 0.0),
        ("C. 관습 임계값 0.5", 0.5),
        ("D. 이론 p* 임계값", p_star),
        ("E. 실험 최적 임계값", best_th),
    ]
    for name, th in strategies:
        profit, n_target = realized_profit(proba, y_te, th)
        print(f"    {name:<18} 개입 {n_target:>4}명 | 이익 {profit:>+12,.0f}원")
    c_profit, _ = realized_profit(proba, y_te, 0.5)
    d_profit, _ = realized_profit(proba, y_te, p_star)
    if c_profit > 0:
        print(f"    => 모델은 그대로, 임계값만 0.5 -> p* 로 바꿔 이익 {d_profit/c_profit:.1f}배.")
    print("       '모델 개선'보다 '의사결정 설계'가 먼저인 이유입니다.")

    # [5] 이익 곡선 PNG ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grid, profits / 1e4, color="#4477aa", lw=2)
    ax.axvline(p_star, color="#228833", ls="--", lw=1.5,
               label=f"theory p* = {p_star:.3f}")
    ax.axvline(best_th, color="#cc6677", ls=":", lw=1.5,
               label=f"empirical best = {best_th:.2f}")
    ax.axvline(0.5, color="gray", ls="-.", lw=1, label="convention 0.5")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("coupon threshold (churn probability)")
    ax.set_ylabel("realized profit (10k KRW)")
    ax.set_title("Profit vs threshold: who should get the coupon?")
    ax.legend()
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "profit_curve.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f"\n[5] 이익 곡선 저장: {png}")

    # [6] 캠페인 ROI 시나리오 (10만 고객 규모) -------------------------------
    print("\n[6] 캠페인 ROI 보고 — 고객 10만 명 규모, 성공률 3개 시나리오")
    n_scale = 100_000 / len(y_te)
    target = proba >= best_th
    n_target = int(target.sum() * n_scale)
    cost = n_target * COST_C
    print(f"    개입 대상: 약 {n_target:,}명 / 쿠폰 예산: {cost/1e4:,.0f}만 원")
    print(f"    {'시나리오':<12} {'성공률':>5} | {'순이익':>12} | {'ROI':>7}")
    for label, s_val in [("보수적", 0.20), ("기본", 0.30), ("낙관", 0.40)]:
        profit, _ = realized_profit(proba, y_te, best_th, s=s_val)
        profit_scaled = profit * n_scale
        roi = profit_scaled / cost * 100
        print(f"    {label:<12} {s_val:>5.0%} | {profit_scaled/1e4:>+10,.0f}만 원 | {roi:>6.0f}%")
    base_profit, _ = realized_profit(proba, y_te, best_th, s=0.30)
    base_roi = base_profit * n_scale / cost * 100
    print("\n    기획서 문장 예시:")
    print(f'    "이탈 확률 {best_th:.0%} 초과 고객 약 {n_target:,}명에게 쿠폰을 제공하면')
    print(f'     기본 시나리오(성공률 30%)에서 기대 ROI 는 약 {base_roi:.0f}% 입니다.')
    print(f'     성공률은 캠페인 대조군으로 실측해 다음 분기에 재계산합니다."')
    print("\n    교훈: 모델의 확률에 가격표(V, C, s)를 곱하는 순간,")
    print("          머신러닝은 통계가 아니라 경영의 도구가 됩니다.")


if __name__ == "__main__":
    main()
