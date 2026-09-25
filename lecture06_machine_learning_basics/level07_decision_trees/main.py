"""
level07 — 결정트리: 스무고개로 이탈 고객 찾기

churn_table 로 결정트리를 학습하고,
  - 지니 불순도를 직접 계산해 '좋은 질문'의 기준을 이해하고
  - 학습된 트리를 사람이 읽는 규칙 문장으로 출력해 해석하고
  - 깊이를 바꿔 가며 과적합의 갈림길을 관찰합니다.
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


def gini(labels: np.ndarray) -> float:
    """지니 불순도: 방이 한 종류면 0, 반반이면 0.5. 이 8줄이 트리 수학의 전부."""
    if len(labels) == 0:
        return 0.0
    p = labels.mean()               # 이탈 비율
    return 1.0 - (p ** 2 + (1 - p) ** 2)


def split_gain(y: np.ndarray, mask: np.ndarray) -> float:
    """질문(mask) 후 두 방의 크기 가중 평균 불순도가 얼마나 줄었나."""
    left, right = y[mask], y[~mask]
    after = (len(left) * gini(left) + len(right) * gini(right)) / len(y)
    return gini(y) - after


if __name__ == "__main__":
    np.random.seed(0)

    # [1] 데이터 준비 (트리는 표준화가 필요 없음: 임계값 비교 모델) -----------
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES].astype(float), df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)
    print(f"[1] 데이터: 고객 {len(df)}명, 이탈률 {y.mean():.1%} — 훈련 {len(X_tr)} / 테스트 {len(X_te)}\n")

    # [2] 지니 불순도 직접 계산: 좋은 질문이란? -------------------------------
    y_arr = y_tr.to_numpy()
    print("[2] 지니 불순도 — '방이 얼마나 섞여 있나'")
    print(f"    질문 전 전체 방의 불순도: {gini(y_arr):.4f}")
    for feat, th in [("usage_days_30d", 9.5), ("monthly_fee", 15000), ("support_calls_30d", 1.5)]:
        gain = split_gain(y_arr, (X_tr[feat] <= th).to_numpy())
        print(f"    질문 \"{feat} <= {th}\" 의 불순도 감소: {gain:.4f}")
    print("    -> 트리는 모든 특징 x 모든 임계값을 시험해 감소폭 최대 질문을 고릅니다.")
    print("       (level00 의 임계값 탐색이 재귀적으로 반복될 뿐 — 마법이 아닙니다)\n")

    # [3] 깊이 3 트리 학습 + 규칙을 텍스트로 출력 ------------------------------
    # class_weight="balanced": 이탈자(15%)를 유지자보다 무겁게 취급해서
    # 소수 클래스(이탈) 쪽 규칙이 트리에 드러나게 합니다.
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30,
                                  class_weight="balanced", random_state=0)
    tree.fit(X_tr, y_tr)
    print("[3] 깊이 3 트리의 규칙 전문 (export_text)")
    print(export_text(tree, feature_names=FEATURES))

    # 잎(leaf) 방들을 한국어 보고 문장으로 번역
    leaf_id = tree.apply(X_tr)
    print("    잎 방 요약 (훈련 데이터 기준):")
    rows = []
    for leaf in np.unique(leaf_id):
        members = y_tr[leaf_id == leaf]
        rows.append((leaf, len(members), members.mean()))
    for leaf, n, rate in sorted(rows, key=lambda r: -r[2])[:3]:
        print(f"      위험 상위 방 #{leaf}: {n}명, 이탈률 {rate:.1%}")
    print("    -> 상위 방의 경로(위 규칙 트리에서 추적)는 그대로 CRM 규칙으로 역수입 가능.\n")

    # [4] 깊이별 훈련/테스트 성적 — 과적합의 갈림길 ----------------------------
    print("[4] 깊이와 과적합 (level04 실험 A 재방문)")
    print("    깊이      훈련 정확도   테스트 정확도   테스트 재현율")
    for depth in [1, 3, 5, 10, None]:
        t = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_tr, y_tr)
        rec = recall_score(y_te, t.predict(X_te))
        label = "무제한" if depth is None else f"{depth:>4}"
        print(f"    {label:>6}      {t.score(X_tr, y_tr):6.1%}       {t.score(X_te, y_te):6.1%}        {rec:6.1%}")
    print("    -> 깊이 무제한 = 훈련 100% 암기, 테스트에선 오히려 손해.\n")

    # [5] 특징 중요도 ---------------------------------------------------------
    print("[5] 특징 중요도 (불순도 감소 기여 비율, 합계 1)")
    order = np.argsort(-tree.feature_importances_)
    for i in order:
        bar = "#" * int(tree.feature_importances_[i] * 40)
        print(f"    {FEATURES[i]:18s} {tree.feature_importances_[i]:.3f} {bar}")
    print("    -> '많이 써먹은 특징'이지 '원인 순위'가 아닙니다 (해석 주의, level11).")
