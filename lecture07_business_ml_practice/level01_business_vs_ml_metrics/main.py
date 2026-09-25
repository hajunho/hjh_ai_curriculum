"""
혼동행렬을 원화 금액으로 환산하는 계산기.
이탈 예측 모델(로지스틱 회귀)의 TP/FP/FN/TN 에 비즈니스 단가를 붙여
'이 모델은 얼마짜리인가', '재현율 1%p 는 몇 원인가'를 계산합니다.
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, recall_score, precision_score

# ---- 비즈니스 단가 (재무/마케팅 부서에서 받아 오는 숫자라고 가정) ----
VALUE_V = 179_000   # 고객 잔존 가치: 지키면 얻는 매출 (원/명)
COST_C = 12_000     # 개입 비용: 쿠폰 + 상담 원가 (원/명)
SUCCESS_S = 0.30    # 개입 성공률: 쿠폰을 받은 이탈 예정자가 잔류할 확률

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


def profit_of(tp: int, fp: int) -> float:
    """혼동행렬 -> 기대이익(원). TP는 일부를 구해내고, FP는 비용만 씁니다."""
    return tp * (VALUE_V * SUCCESS_S - COST_C) - fp * COST_C


def main() -> None:
    print("=" * 62)
    print(" 혼동행렬 -> 원화 번역기: 이탈 모델은 얼마짜리인가")
    print("=" * 62)

    # [1] 데이터 준비와 모델 훈련 ------------------------------------
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES], df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    # class_weight="balanced": 이탈(소수 클래스)을 놓치지 않도록 가중치 부여 (level07에서 자세히)
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(random_state=42, class_weight="balanced"))
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    print(f"\n[1] 로지스틱 회귀 훈련 완료 (학습 {len(X_tr)}명 / 테스트 {len(X_te)}명, "
          f"테스트 이탈률 {y_te.mean():.1%})")

    # [2] 혼동행렬 ----------------------------------------------------
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    rec = recall_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    print("\n[2] 테스트 혼동행렬 (양성 = 이탈)")
    print(f"    TP(이탈 적중)={tp:4d}  FN(이탈 놓침)={fn:4d}")
    print(f"    FP(헛다리)   ={fp:4d}  TN(정상 통과)={tn:4d}")
    print(f"    재현율 {rec:.1%} / 정밀도 {prec:.1%}")

    # [3] 칸마다 단가 붙이기 -----------------------------------------
    print("\n[3] 비즈니스 단가로 환산")
    print(f"    단가: 고객가치 V={VALUE_V:,}원, 개입비용 C={COST_C:,}원, 성공률 s={SUCCESS_S:.0%}")
    unit_tp = VALUE_V * SUCCESS_S - COST_C
    print(f"    TP 1건의 가치 = V*s - C = {unit_tp:+,.0f}원")
    print(f"    FP 1건의 가치 = -C      = {-COST_C:+,}원")
    print(f"    FN 1건 = 지출 0원, 그러나 기회손실 V*s = {VALUE_V*SUCCESS_S:,.0f}원")
    model_profit = profit_of(tp, fp)
    print(f"    => 모델 기대이익(테스트 {len(X_te)}명 기준): {model_profit:+,.0f}원")

    # [4] 비교군: 아무것도 안 함 vs 전원 쿠폰 ------------------------
    print("\n[4] 전략 비교 (같은 테스트 고객 기준)")
    n_pos = int(y_te.sum())
    do_nothing = profit_of(0, 0)
    give_all = profit_of(n_pos, len(y_te) - n_pos)  # 전원 개입: 이탈자 전원 TP, 나머지 전원 FP
    print(f"    A. 아무것도 안 함        : {do_nothing:+13,.0f}원")
    print(f"    B. 전원에게 쿠폰 발송    : {give_all:+13,.0f}원")
    print(f"    C. 모델이 찍은 사람에게만: {model_profit:+13,.0f}원")
    best = max([("A", do_nothing), ("B", give_all), ("C", model_profit)], key=lambda t: t[1])
    print(f"    => 최선 전략: {best[0]} — 모델의 가치는 '누구에게 쓸지 고르는 능력'입니다.")

    # [5] 재현율 1%p 의 가치 ------------------------------------------
    print("\n[5] '재현율 1%p'는 몇 원인가")
    n_customers = 100_000            # 실제 서비스 규모로 확장
    churn_rate = float(y.mean())     # 데이터에서 추정한 이탈률
    value_1pp = n_customers * churn_rate * 0.01 * unit_tp
    print(f"    공식: N * 이탈률 * 0.01 * (V*s - C)")
    print(f"        = {n_customers:,}명 * {churn_rate:.1%} * 1%p * {unit_tp:,.0f}원")
    print(f"        = 약 {value_1pp:,.0f}원 (월 단위 캠페인이면 연간 약 {value_1pp*12:,.0f}원)")
    print("\n    보고 문장 예시:")
    print(f'    "재현율을 {rec:.0%}에서 {rec+0.05:.0%}로 5%p 올리면')
    print(f'     연간 약 {value_1pp*5*12/1e8:.1f}억 원의 이탈 방어 이익이 추가됩니다."')
    print("\n    교훈: ML 지표는 단가를 붙이는 순간 예산 언어가 됩니다.")


if __name__ == "__main__":
    main()
