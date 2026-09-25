"""
실전 2 — 고객 이탈 예측 완결 프로젝트.
문제 명세 -> 미니 EDA -> 피처 -> 교차검증 -> 최종 모델 -> 계수 해석 ->
'위험 고객 톱10 + 주요 원인 + 추천 액션' 표까지,
마케팅팀이 그대로 실행할 수 있는 산출물을 만듭니다.
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, recall_score, precision_score

BASE = ["tenure_months", "monthly_fee", "usage_days_30d",
        "support_calls_30d", "plan_changes", "auto_pay"]
DERIVED = ["usage_per_tenure", "calls_plus_changes"]
FEAT_KO = {
    "tenure_months": "가입 개월수", "monthly_fee": "월 요금",
    "usage_days_30d": "최근 30일 이용일수", "support_calls_30d": "최근 30일 문의",
    "plan_changes": "요금제 변경", "auto_pay": "자동결제",
    "usage_per_tenure": "기간 대비 활동성", "calls_plus_changes": "불만 신호 합",
}
# 주요 원인 -> 추천 액션 (모델 밖에서 사람이 설계하는 대응 규칙)
ACTION_MAP = {
    "usage_days_30d": "재방문 유도 콘텐츠 + 7일 무료 이용권",
    "usage_per_tenure": "재방문 유도 콘텐츠 + 7일 무료 이용권",
    "support_calls_30d": "CS 우선 상담 배정, 불만 원인 해결",
    "calls_plus_changes": "CS 우선 상담 배정, 불만 원인 해결",
    "plan_changes": "요금제 맞춤 추천 상담",
    "monthly_fee": "요금제 맞춤 추천 상담",
    "auto_pay": "자동결제 전환 시 1개월 20% 할인",
    "tenure_months": "온보딩 가이드 + 첫 달 혜택 안내",
}


def main() -> None:
    print("=" * 70)
    print(" 실전 2: 고객 이탈 예측 — 명단·이유·액션까지 완결하기")
    print("=" * 70)

    # [1] 문제 명세 --------------------------------------------------------
    print("\n[1] 문제 명세 (코드보다 먼저)")
    print("    예측 대상: 이번 달 구독 이탈 여부 / 활용: 매주 위험 상위 고객 CRM 캠페인")
    print("    목표 지표: 재현율 55%+ 에서 정밀도 25%+ / 기준선: 무작위 발송(적중률=이탈률 약 15%)")

    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))

    # [2] 미니 EDA ---------------------------------------------------------
    print(f"\n[2] 미니 EDA: {len(df)}명, 이탈률 {df['churned'].mean():.1%}")
    grp = df.groupby("churned")[BASE].mean()
    gap = ((grp.loc[1] - grp.loc[0]) / grp.loc[0]).sort_values(key=abs, ascending=False)
    print("    이탈 그룹이 잔류 그룹과 가장 다른 신호 (평균 차이 비율):")
    for name, v in gap.head(3).items():
        print(f"      {FEAT_KO[name]:<14} {v:+.0%}")

    # [3] 피처 준비 --------------------------------------------------------
    print("\n[3] 피처: 원본 6개 + 파생 2개")
    df["usage_per_tenure"] = df["usage_days_30d"] / (df["tenure_months"] + 1)
    df["calls_plus_changes"] = df["support_calls_30d"] + df["plan_changes"]
    features = BASE + DERIVED
    X, y = df[features], df["churned"]

    # [4] 교차검증 + 최종 모델 ----------------------------------------------
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(random_state=42, class_weight="balanced"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print(f"\n[4] 5-fold 교차검증 AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=42, stratify=y)
    idx_te = X_te.index
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    rec, prec = recall_score(y_te, pred), precision_score(y_te, pred)
    print(f"    테스트 성능: AUC {roc_auc_score(y_te, proba):.3f} / "
          f"재현율 {rec:.1%} / 정밀도 {prec:.1%}")
    goal = "달성" if (rec >= 0.55 and prec >= 0.25) else "미달 -> 임계값·피처 재검토"
    print(f"    목표 지표(재현율 55%+, 정밀도 25%+) 판정: {goal}")
    print(f"    (무작위 발송 기준선의 적중률 {y_te.mean():.1%} 대비 정밀도 {prec/y_te.mean():.1f}배)")

    # [5] 해석: 표준화 계수 -------------------------------------------------
    print("\n[5] 모델 해석 — 무엇이 위험을 키우나 (표준화 계수)")
    scaler = pipe.named_steps["standardscaler"]
    lr = pipe.named_steps["logisticregression"]
    coefs = lr.coef_[0]
    for name, c in sorted(zip(features, coefs), key=lambda t: -abs(t[1])):
        arrow = "위험 증가" if c > 0 else "위험 감소"
        print(f"    {FEAT_KO[name]:<14} {c:+.2f} ({arrow}) {'#' * int(abs(c) * 6)}")

    # [6] 산출물: 위험 톱10 + 원인 + 액션 -----------------------------------
    print("\n[6] 최종 산출물 — 이탈 위험 톱10 명단 (테스트 고객 기준)")
    # 개별 고객의 '주요 원인' = 표준화 피처값 x 계수 중 위험 방향 기여가 가장 큰 피처
    Z = scaler.transform(X_te)                      # 표준화된 피처값
    contrib = Z * coefs                             # 고객별 x 피처별 위험 기여도
    top10 = np.argsort(proba)[::-1][:10]
    print(f"    {'고객ID':<9} {'이탈확률':>7}  {'주요 원인':<16} 추천 액션")
    print("    " + "-" * 66)
    for i in top10:
        cust_id = df.loc[idx_te[i], "customer_id"]
        main_feat = features[int(np.argmax(contrib[i]))]
        action = ACTION_MAP[main_feat]
        print(f"    {cust_id:<9} {proba[i]:>6.1%}  {FEAT_KO[main_feat]:<16} {action}")
    print("\n    캠페인 운영 메모:")
    print("      - 확률 80% 이상: 전화 상담 / 50~80%: 쿠폰+메시지 (확률 구간별 강도 조절)")
    print("      - 효과 검증: 위험 고객 일부를 무작위 대조군으로 남겨 이탈률 비교")
    print("\n    교훈: 프로젝트의 완성은 AUC 가 아니라 '월요일 아침 마케팅팀이")
    print("          그대로 실행할 수 있는 표'입니다.")


if __name__ == "__main__":
    main()
