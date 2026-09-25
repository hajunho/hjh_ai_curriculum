"""
불균형 데이터(사기 1.5%) 대처 기법 비교 실험.
기본 모델 / class_weight / 오버샘플링 / 임계값 조정 네 가지로
재현율-정밀도 트레이드오프를 표로 확인하고,
'하루 알람 처리 예산' 제약에서 합리적인 임계값을 골라 봅니다.
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             average_precision_score, precision_recall_curve)

# 이력 집계 시스템이 아직 없어 '결제 순간 아는 필드'만 쓸 수 있다고 가정
# (집계 피처 tx_count_1h 를 더하면 문제가 얼마나 쉬워지는지는 직접 해보기 과제)
FEATURES = ["amount", "hour", "is_foreign"]


def new_model(**kw):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(random_state=42, max_iter=1000, **kw))


def show(name, y_true, y_pred):
    print(f"    {name:<28} 정확도 {accuracy_score(y_true, y_pred):.3f} | "
          f"재현율 {recall_score(y_true, y_pred):.3f} | "
          f"정밀도 {precision_score(y_true, y_pred, zero_division=0):.3f} | "
          f"알람 {int(y_pred.sum())}건")


def main() -> None:
    print("=" * 66)
    print(" 불균형 데이터: 1.5% 의 사기를 잡는 네 가지 방법")
    print("=" * 66)

    df = pd.DataFrame(hjh_data.fraud_table(n=5000, seed=11))
    X, y = df[FEATURES], df["is_fraud"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=42, stratify=y)
    print(f"\n    데이터: {len(df)}건, 사기 {y.sum()}건({y.mean():.2%}) / "
          f"테스트 {len(y_te)}건에 사기 {y_te.sum()}건")

    # [1] 기본 설정: 게으른 모델 -----------------------------------------
    print("\n[1] 기본 설정 (임계값 0.5, 가중치 없음)")
    base = new_model()
    base.fit(X_tr, y_tr)
    show("기본 로지스틱 회귀", y_te, base.predict(X_te))
    missed = int(y_te.sum()) - int((base.predict(X_te) & y_te).sum())
    print(f"    => 정확도 99% 대라 훌륭해 보이지만, 사기 {int(y_te.sum())}건 중 {missed}건을 놓칩니다.")
    print("       (극단적으로는 '전부 정상' 예측도 정확도 98.6% — 정확도는 이 문제의 지표가 아닙니다)")

    # [2] class_weight: 벌점표 고치기 -------------------------------------
    print("\n[2] class_weight='balanced' — 소수 클래스 실수에 큰 벌점")
    weighted = new_model(class_weight="balanced")
    weighted.fit(X_tr, y_tr)
    show("가중치 모델", y_te, weighted.predict(X_te))

    # [3] 오버샘플링: 학습 데이터에만! ------------------------------------
    print("\n[3] 수동 오버샘플링 — 분할 '후' 학습 데이터의 사기 행만 복제")
    rng = np.random.default_rng(42)
    pos_idx = y_tr[y_tr == 1].index
    ratio = int((y_tr == 0).sum() / (y_tr == 1).sum())  # 대략 균형이 되는 배수
    dup_idx = rng.choice(pos_idx, size=len(pos_idx) * (ratio - 1), replace=True)
    X_bal = pd.concat([X_tr, X_tr.loc[dup_idx]])
    y_bal = pd.concat([y_tr, y_tr.loc[dup_idx]])
    print(f"    학습 데이터 {len(y_tr)}건 -> {len(y_bal)}건 (사기 비율 {y_bal.mean():.1%})")
    over = new_model()
    over.fit(X_bal, y_bal)
    show("오버샘플링 모델", y_te, over.predict(X_te))
    print("    => [2]와 비슷한 효과. 둘 다 '벌점표 고치기'의 변형입니다.")
    print("       (테스트 비율은 절대 조작 금지 — 시험은 실제 세상의 비율로)")

    # [4] 임계값 조정: 같은 모델, 다른 운영점 ------------------------------
    print("\n[4] 임계값 조정 — 기본 모델 하나로 운영점만 이동")
    proba = base.predict_proba(X_te)[:, 1]
    print(f"    {'임계값':>6} | {'재현율':>6} | {'정밀도':>6} | 알람 건수")
    rows = []
    for th in [0.9, 0.7, 0.5, 0.3, 0.2, 0.1]:
        pred = (proba >= th).astype(int)
        r = recall_score(y_te, pred)
        p = precision_score(y_te, pred, zero_division=0)
        rows.append((th, r, p, int(pred.sum())))
        print(f"    {th:6.2f} | {r:6.1%} | {p:6.1%} | {int(pred.sum()):4d}건")
    print("    => 아래로 갈수록 재현율은 오르고 정밀도는 내려갑니다. 공짜 점심은 없습니다.")

    # [5] PR 곡선 요약과 알람 예산 -----------------------------------------
    print("\n[5] PR 곡선 요약과 '알람 예산'으로 운영점 고르기")
    pr_auc = average_precision_score(y_te, proba)
    prec_c, rec_c, th_c = precision_recall_curve(y_te, proba)
    print(f"    PR-AUC(평균 정밀도) = {pr_auc:.3f}  (임계값 전 구간의 종합 점수)")
    budget = 15  # 조사팀이 이 테스트 기간에 처리할 수 있는 알람 건수
    order = np.argsort(proba)[::-1]
    top = order[:budget]
    caught = int(y_te.iloc[top].sum())
    th_budget = proba[order[budget - 1]]
    print(f"    제약: 조사팀 처리 가능량 = {budget}건")
    print(f"    => 의심 점수 상위 {budget}건만 알람 (해당 임계값 약 {th_budget:.2f})")
    print(f"       잡은 사기 {caught}건 / 전체 {int(y_te.sum())}건 "
          f"(재현율 {caught / y_te.sum():.1%}, 정밀도 {caught / budget:.1%})")
    print("\n    교훈: 어느 운영점을 고를지는 모델이 아니라 비즈니스(인력·비용)가 정합니다.")
    print("          모델의 일은 '좋은 곡선'을 만드는 것까지입니다.")


if __name__ == "__main__":
    main()
