"""
level05 — 로지스틱 회귀: 구독 이탈 확률 모델

churn_table(고객 2000명)로 '이탈 확률'을 출력하는 분류 모델을 만듭니다.
  - 시그모이드: 선형 점수 z 를 0~1 확률로 바꾸는 깔때기
  - 계수 -> 오즈비(odds ratio) 번역: "문의 1단위 증가 -> 이탈 오즈 N배"
  - 임계값(threshold)은 모델이 아니라 비즈니스가 정한다
"""

import math
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]
FEATURE_KO = {"tenure_months": "가입 개월", "monthly_fee": "월 요금",
              "usage_days_30d": "이용일수(30일)", "support_calls_30d": "문의 수(30일)",
              "plan_changes": "요금제 변경", "auto_pay": "자동결제 여부"}


def sigmoid(z: float) -> float:
    """확률 깔때기: 아무리 큰 점수도 0~1 사이로 눌러 담는다."""
    return 1.0 / (1.0 + math.exp(-z))


if __name__ == "__main__":
    np.random.seed(0)

    # [1] 데이터 준비 + 분리 -------------------------------------------------
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X = df[FEATURES].astype(float)     # customer_id 는 의미 없는 열이라 제외
    y = df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)  # 이탈률 유지 분할
    print("[1] 데이터: 구독 고객 2000명, 이탈률 {:.1%}".format(y.mean()))
    print(f"    훈련 {len(X_tr)}명 / 테스트 {len(X_te)}명 (level04 원칙 준수)\n")

    # [2] 시그모이드 깔때기 관찰 ----------------------------------------------
    print("[2] 시그모이드: 선형 점수 z -> 확률 p")
    for z in [-4, -2, 0, 2, 4]:
        print(f"    z = {z:+d}  ->  p = {sigmoid(z):5.1%}")
    print("    -> 점수 0점이면 반반(50%), ±4점이면 거의 확정.\n")

    # [3] 학습 ---------------------------------------------------------------
    # Pipeline: 스케일러가 훈련 세트로만 fit -> 누설 자동 방지 (level04)
    model = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(random_state=0))])
    model.fit(X_tr, y_tr)
    proba_te = model.predict_proba(X_te)[:, 1]        # 이탈 확률
    pred_05 = (proba_te >= 0.5).astype(int)
    print("[3] 로지스틱 회귀 학습 완료 — 테스트 성적 (임계값 0.5)")
    print(f"    정확도 {accuracy_score(y_te, pred_05):.1%} / "
          f"정밀도 {precision_score(y_te, pred_05):.1%} / "
          f"재현율 {recall_score(y_te, pred_05):.1%}\n")

    # [4] 계수 -> 오즈비 번역 --------------------------------------------------
    clf = model.named_steps["clf"]
    print("[4] 계수 해석 (표준화된 특징 기준: '1 표준편차 증가'의 효과)")
    print("    특징                계수      오즈비    해석")
    order = np.argsort(-np.abs(clf.coef_[0]))
    for i in order:
        coef = clf.coef_[0][i]
        orat = math.exp(coef)
        direction = "이탈 위험 증가" if coef > 0 else "이탈 위험 감소"
        print(f"    {FEATURE_KO[FEATURES[i]]:14s} {coef:+7.3f}   {orat:6.2f}배   {direction}")
    print("    -> 데이터 생성기에 심어 둔 진짜 신호(이용일수↓, 문의↑, 자동결제-)와")
    print("       방향이 일치하는지 확인해 보세요. (단, 상관이지 인과 증명은 아님)\n")

    # [5] 개별 예측 + 임계값 실험 ---------------------------------------------
    print("[5] 개별 고객 예측과 임계값의 비즈니스 결정")
    samples = pd.DataFrame([
        {"tenure_months": 36, "monthly_fee": 9900, "usage_days_30d": 28,
         "support_calls_30d": 0, "plan_changes": 0, "auto_pay": 1},
        {"tenure_months": 3, "monthly_fee": 29900, "usage_days_30d": 2,
         "support_calls_30d": 4, "plan_changes": 2, "auto_pay": 0},
        {"tenure_months": 12, "monthly_fee": 14900, "usage_days_30d": 15,
         "support_calls_30d": 1, "plan_changes": 1, "auto_pay": 1},
    ])[FEATURES].astype(float)
    names = ["성실 이용 고객", "신규+저이용+문의폭주", "평범한 고객"]
    for name, p in zip(names, model.predict_proba(samples)[:, 1]):
        print(f"    {name:16s} 이탈 확률 {p:5.1%}")
    print()
    print("    임계값을 바꾸면 같은 모델도 다른 결정을 내립니다:")
    print("    임계값   위험 분류 인원   정밀도   재현율")
    for th in [0.5, 0.3]:
        pred = (proba_te >= th).astype(int)
        print(f"     {th:.1f}        {pred.sum():>4}명      "
              f"{precision_score(y_te, pred):6.1%}  {recall_score(y_te, pred):6.1%}")
    print("    -> 임계값을 내리면 놓치는 이탈자(재현율↑)는 줄지만 헛수고 상담(정밀도↓)이 늘어남.")
    print("       0.5 는 관례일 뿐 — 상담 비용과 고객 가치가 임계값을 정해야 합니다.")
