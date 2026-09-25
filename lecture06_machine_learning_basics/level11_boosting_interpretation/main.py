"""
level11 — 그래디언트 부스팅과 모델 해석

churn_table 로 부스팅 모델을 만들고 '왜?'까지 답합니다.
  [2] 로지스틱 vs 랜덤포레스트 vs HistGradientBoosting 성능 비교
  [3] 학습률(오답노트 반영 강도) x 트리 수 미니 실험
  [4] 순열 중요도: 특징 하나를 뒤섞으면 AUC 가 얼마나 무너지나
  [5] 부분의존: 특징 값에 따른 예측 확률의 모양 (텍스트 그래프)
  [6] 개별 예측 설명: 특징을 평균값으로 바꿔치기하며 근거 분해
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]
FEATURE_KO = {"tenure_months": "가입 개월", "monthly_fee": "월 요금",
              "usage_days_30d": "이용일수", "support_calls_30d": "문의 수",
              "plan_changes": "요금제 변경", "auto_pay": "자동결제"}


def explain_one(model, x_row: pd.DataFrame, baseline: pd.Series) -> list[tuple[str, float]]:
    """개별 예측 설명기(12줄): 특징 하나를 '평균 고객' 값으로 바꿔치기했을 때
    이탈 확률이 얼마나 변하는지를 특징별로 잰다. 변화폭 큰 특징 = 판정의 근거."""
    p0 = model.predict_proba(x_row)[0, 1]
    contribs = []
    for f in FEATURES:
        x_mod = x_row.copy()
        x_mod[f] = baseline[f]
        p_mod = model.predict_proba(x_mod)[0, 1]
        contribs.append((f, p0 - p_mod))     # +면 이 특징이 위험을 키운 근거
    return sorted(contribs, key=lambda t: -abs(t[1]))


if __name__ == "__main__":
    np.random.seed(0)

    # [1] 데이터 -------------------------------------------------------------
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES].astype(float), df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)
    print(f"[1] 데이터: 고객 {len(df)}명, 이탈률 {y.mean():.1%}\n")

    # [2] 세 모델 졸업 시험 ----------------------------------------------------
    print("[2] 지금까지 배운 모델들의 성능 비교 (테스트 AUC)")
    models = {
        "로지스틱 회귀 (level05)": Pipeline([
            ("s", StandardScaler()), ("c", LogisticRegression(random_state=0))]),
        "랜덤포레스트 (level08)": RandomForestClassifier(
            n_estimators=300, random_state=0, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=0, early_stopping=True),
    }
    boost = None
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
        print(f"    {name:24s} AUC = {auc:.3f}")
        if isinstance(m, HistGradientBoostingClassifier):
            boost = m
    print("    -> 뜻밖에도 로지스틱이 1등! 이 합성 데이터의 진짜 구조가 '선형 점수표'")
    print("       (로그오즈의 선형식)로 만들어졌기 때문입니다. 교훈 두 가지:")
    print("       (1) 강한 모델이 항상 이기는 게 아니라, 데이터 구조에 맞는 모델이 이긴다.")
    print("       (2) 비선형·상호작용이 많은 현실 데이터에서는 부스팅이 앞서는 경우가 많다.\n")

    # [3] 학습률 x 트리 수 미니 실험 -------------------------------------------
    print("[3] 학습률(오답노트 반영 강도) 실험 — 브레이크와 교시 수의 거래")
    print("    학습률   최대 트리   테스트 AUC")
    for lr, n_iter in [(1.0, 50), (0.3, 100), (0.1, 200), (0.03, 500)]:
        m = HistGradientBoostingClassifier(
            learning_rate=lr, max_iter=n_iter, random_state=0,
            early_stopping=True).fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
        print(f"    {lr:<6}    {n_iter:>5}      {auc:.3f}")
    print("    -> 학습률을 낮추면 트리는 더 필요하지만 결과가 안정되는 경향.")
    print("       (learning_rate 는 level09 의 규제 철학이 부스팅에 온 것)\n")

    # [4] 순열 중요도 ----------------------------------------------------------
    print("[4] 순열 중요도 — 특징 하나를 뒤섞었을 때 테스트 AUC 하락폭")
    perm = permutation_importance(boost, X_te, y_te, scoring="roc_auc",
                                  n_repeats=10, random_state=0)
    print("    특징           AUC 하락(평균±표준편차)")
    for i in np.argsort(-perm.importances_mean):
        bar = "#" * max(0, int(perm.importances_mean[i] * 200))
        print(f"    {FEATURE_KO[FEATURES[i]]:10s}   {perm.importances_mean[i]:+.4f} ± {perm.importances_std[i]:.4f}  {bar}")
    print("    -> '완성된 모델을 테스트 데이터로 심문'하는 방식이라 모델 종류 무관,")
    print("       불순도 중요도(level07~08)보다 보고용으로 정직합니다.\n")

    # [5] 부분의존 -------------------------------------------------------------
    print("[5] 부분의존 — 다른 조건 고정, 특징 하나만 움직일 때의 평균 이탈 확률")
    for feat in ["usage_days_30d", "support_calls_30d"]:
        # method="brute": predict_proba 기준으로 계산 -> 결과가 '확률' 단위
        pd_res = partial_dependence(boost, X_te, [feat], kind="average",
                                    grid_resolution=7, method="brute")
        grid = pd_res["grid_values"][0]
        avg = pd_res["average"][0]
        print(f"    {FEATURE_KO[feat]} ({feat})")
        for g, v in zip(grid, avg):
            bar = "#" * int(v * 60)
            print(f"      값 {g:6.1f} -> 평균 확률 {v:5.1%} {bar}")
    print("    -> 곡선의 '모양'(어느 구간에서 급변하나)이 로지스틱 계수 하나보다")
    print("       풍부한 정보를 줍니다. 단, 연관이지 인과가 아닙니다.\n")

    # [6] 개별 예측 설명 --------------------------------------------------------
    proba_te = boost.predict_proba(X_te)[:, 1]
    idx = int(np.argmax(proba_te))                     # 테스트에서 최고 위험 고객
    x_row = X_te.iloc[[idx]]
    baseline = X_tr.mean()                             # '평균 고객'
    print("[6] 개별 예측 설명 — 최고 위험 고객 1명의 판정 근거 분해")
    print(f"    이 고객의 이탈 확률: {proba_te[idx]:.1%} (평균 고객 기준값 대비)")
    print("    특징(고객 값 -> 평균값 바꿔치기)      확률 변화")
    for f, delta in explain_one(boost, x_row, baseline):
        if abs(delta) < 0.005:
            direction = "영향 미미"
        else:
            direction = "위험을 키운 근거" if delta > 0 else "위험을 낮춘 요소"
        print(f"    {FEATURE_KO[f]:10s} ({x_row[f].iloc[0]:>8.1f} -> {baseline[f]:>8.1f})   "
              f"{delta:+6.1%}p  {direction}")
    print("    -> '문의를 평균 수준으로 되돌리면 확률이 크게 내려간다'는 식의 문장이")
    print("       상담 스크립트의 근거가 됩니다. (전문 도구 SHAP 도 같은 원리의 정교화)")
    print()
    print("[7] lecture06 완주! 성능(부스팅)과 설명(중요도·부분의존·사례 설명)을")
    print("    세트로 보고할 수 있어야 모델이 조직에 채택됩니다.")
