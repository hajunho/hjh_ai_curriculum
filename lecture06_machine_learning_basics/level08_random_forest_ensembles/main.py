"""
level08 — 랜덤포레스트와 앙상블: 집단지성 실험

churn_table 로 단일 결정트리 vs 랜덤포레스트를 비교합니다.
  [2] 배깅(부트스트랩 + 투표)을 15줄로 직접 구현해 '평균의 힘' 확인
  [3] 성능 비교 (AUC / 재현율)
  [4] 안정성 비교: 분할을 12번 바꿔 AUC 의 요동(표준편차)을 측정
  [5] 특징 중요도: 한 그루 vs 숲
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


def bagged_predict(X_tr, y_tr, X_te, n_trees: int, seed: int = 0) -> np.ndarray:
    """미니 배깅 직접 구현: 복원추출 데이터로 깊은 트리들을 길러 확률 평균(투표).
    sklearn RandomForest = 여기에 '분기마다 특징 무작위 선택'을 더한 것."""
    rng = np.random.default_rng(seed)
    n = len(X_tr)
    probas = []
    for i in range(n_trees):
        idx = rng.choice(n, n, replace=True)          # 부트스트랩: 같은 크기 복원추출
        tree = DecisionTreeClassifier(random_state=i)  # 깊이 무제한(저편향·고분산)
        tree.fit(X_tr.iloc[idx], y_tr.iloc[idx])
        probas.append(tree.predict_proba(X_te)[:, 1])
    return np.mean(probas, axis=0)                     # 평균 = 출렁임(분산) 상쇄


if __name__ == "__main__":
    np.random.seed(0)

    # [1] 데이터 준비 --------------------------------------------------------
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES].astype(float), df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)
    print(f"[1] 데이터: 고객 {len(df)}명, 이탈률 {y.mean():.1%}\n")

    # [2] 미니 배깅 직접 구현 -------------------------------------------------
    single = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
    auc_single = roc_auc_score(y_te, single.predict_proba(X_te)[:, 1])
    print("[2] 배깅을 직접 구현 — 소 무게 맞히기 대회의 원리")
    print(f"    깊은 트리 1그루           테스트 AUC = {auc_single:.3f}")
    for n_trees in [5, 25]:
        auc_bag = roc_auc_score(y_te, bagged_predict(X_tr, y_tr, X_te, n_trees))
        print(f"    같은 트리 {n_trees:>2}그루의 투표    테스트 AUC = {auc_bag:.3f}")
    print("    -> 개개인(트리)은 과적합해도, 서로 다른 경험을 한 다수의 평균은 강해집니다.\n")

    # [3] 성능 비교: 단일 트리 vs 랜덤포레스트 --------------------------------
    print("[3] 성능 비교 (테스트 세트)")
    models = {
        "트리(깊이 5)": DecisionTreeClassifier(max_depth=5, random_state=0),
        "트리(무제한)": DecisionTreeClassifier(random_state=0),
        "랜덤포레스트(300그루)": RandomForestClassifier(
            n_estimators=300, random_state=0, n_jobs=-1),
    }
    print("    모델                     AUC     재현율(임계 0.5)")
    forest = None
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
        rec = recall_score(y_te, m.predict(X_te))
        print(f"    {name:20s} {auc:.3f}      {rec:6.1%}")
        if isinstance(m, RandomForestClassifier):
            forest = m
    print()

    # [4] 안정성 비교: 분할을 바꿔 가며 요동 측정 ------------------------------
    print("[4] 안정성 비교 — 분할을 12번 바꿔 AUC 반복 측정")
    aucs_tree, aucs_rf = [], []
    for rep in range(12):
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25,
                                          random_state=rep, stratify=y)
        t = DecisionTreeClassifier(max_depth=5, random_state=0).fit(Xa, ya)
        f = RandomForestClassifier(n_estimators=150, random_state=0, n_jobs=-1).fit(Xa, ya)
        aucs_tree.append(roc_auc_score(yb, t.predict_proba(Xb)[:, 1]))
        aucs_rf.append(roc_auc_score(yb, f.predict_proba(Xb)[:, 1]))
    aucs_tree, aucs_rf = np.array(aucs_tree), np.array(aucs_rf)
    print(f"    단일 트리     AUC 평균 {aucs_tree.mean():.3f} ± 표준편차 {aucs_tree.std():.3f}")
    print(f"    랜덤포레스트  AUC 평균 {aucs_rf.mean():.3f} ± 표준편차 {aucs_rf.std():.3f}")
    print(f"    포레스트가 이긴 횟수: {int((aucs_rf > aucs_tree).sum())}/12")
    print("    -> 평균 성능만이 아니라 '요동(표준편차)'이 작다는 것이 실무 신뢰의 핵심.\n")

    # [5] 특징 중요도: 한 그루 vs 숲 ------------------------------------------
    print("[5] 특징 중요도 비교 (불순도 감소 기여, 합계 1)")
    tree5 = models["트리(깊이 5)"]
    print("    특징                 트리1그루   포레스트300그루")
    for i in np.argsort(-forest.feature_importances_):
        print(f"    {FEATURES[i]:18s}   {tree5.feature_importances_[i]:.3f}       "
              f"{forest.feature_importances_[i]:.3f}")
    print("    -> 숲의 중요도는 여러 그루의 평균이라 더 안정적입니다.")
    print("       (여전히 '많이 쓰인 정도'일 뿐 인과 아님 — 공정한 측정은 level11)")
