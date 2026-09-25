"""
level06 — 평가지표: 정확도의 함정

사기 비율 약 1.5%의 카드 거래 데이터에서
  - 무조건 "정상"을 외치는 깡통 모델이 정확도 98.5%가 되는 함정을 시연하고
  - 혼동행렬 / 정밀도 / 재현율 / F1 / ROC-AUC 로 모델을 제대로 읽습니다.
AUC 의 확률적 해석("무작위 사기·정상 쌍에서 사기 쪽 확률이 높을 확률")을
무작위 쌍 추출 시뮬레이션으로 직접 검증합니다.
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

# tx_count_1h 는 이 합성 데이터에서 사기를 '완벽하게' 갈라 버리는 특징이라 제외.
# (현실에서 이렇게 완벽한 특징이 보이면 성능 축하가 아니라 누설(leakage) 의심부터!)
FEATURES = ["amount", "hour", "is_foreign"]


def print_metrics(name: str, y_true, y_pred) -> None:
    """혼동행렬과 4대 지표를 한 번에 출력."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp + tn) / len(y_true)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"    {name}")
    print(f"      혼동행렬: TP={tp:>3} (잘 잡음)   FN={fn:>3} (놓침!)")
    print(f"                FP={fp:>3} (헛경보)    TN={tn:>4} (통과)")
    print(f"      정확도 {acc:.1%} / 정밀도 {prec:.1%} / 재현율 {rec:.1%} / F1 {f1:.3f}")


def auc_by_sampling(y_true, proba, n_pairs: int = 10_000, seed: int = 0) -> float:
    """AUC 의 정의를 시뮬레이션으로 검증:
    사기 1건과 정상 1건을 무작위로 뽑아 '사기 쪽 확률이 더 높은' 비율."""
    rng = np.random.default_rng(seed)
    pos = proba[y_true == 1]
    neg = proba[y_true == 0]
    p = rng.choice(pos, n_pairs)
    n = rng.choice(neg, n_pairs)
    return float(np.mean((p > n) + 0.5 * (p == n)))


if __name__ == "__main__":
    np.random.seed(0)

    # [1] 데이터 준비 --------------------------------------------------------
    df = pd.DataFrame(hjh_data.fraud_table(n=8000, seed=11))
    X, y = df[FEATURES].astype(float), df["is_fraud"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)  # 불균형 비율 유지 분할
    print(f"[1] 데이터: 카드 거래 {len(df):,}건, 사기 비율 {y.mean():.2%}")
    print(f"    훈련 {len(X_tr):,}건 / 테스트 {len(X_te):,}건 (stratify 로 비율 유지)\n")

    # [2] 깡통 모델 — 전부 정상 예측 ------------------------------------------
    dummy_pred = np.zeros(len(y_te), dtype=int)
    print("[2] 깡통 모델: 무조건 '정상'이라고 우기기")
    print_metrics("전부 정상 예측:", y_te, dummy_pred)
    print("      -> 정확도 98.5% 인데 사기는 0건 검거. '정확도의 함정'의 실체입니다.")
    print("         불균형 데이터에서 정확도는 첫 줄이 아니라 각주에 쓰는 지표.\n")

    # [3] 진짜 모델 — 로지스틱 회귀 -------------------------------------------
    model = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(random_state=0))])
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    print("[3] 진짜 모델: 로지스틱 회귀 (임계값 0.5)")
    print_metrics("로지스틱 회귀:", y_te, (proba >= 0.5).astype(int))
    print("      -> 정확도는 깡통과 몇 %p 차이지만 혼동행렬은 완전히 다른 세계입니다.\n")

    # [4] 임계값 시소: 정밀도 vs 재현율 ---------------------------------------
    print("[4] 임계값을 훑으며 보는 정밀도-재현율 시소")
    print("    임계값   경보건수   정밀도    재현율")
    for th in [0.9, 0.7, 0.5, 0.3, 0.1]:
        pred = (proba >= th).astype(int)
        prec = precision_score(y_te, pred, zero_division=0)
        rec = recall_score(y_te, pred, zero_division=0)
        print(f"     {th:.1f}      {pred.sum():>4}     {prec:6.1%}   {rec:6.1%}")
    print("    -> 감도를 올리면(임계값↓) 놓침은 줄고 헛경보는 늘어납니다.")
    print("       어디에 앉을지는 '놓침 비용 vs 헛경보 비용'을 아는 사람이 정합니다.\n")

    # [5] ROC-AUC: 임계값과 무관한 분별력 -------------------------------------
    auc_dummy = roc_auc_score(y_te, np.zeros(len(y_te)))
    auc_model = roc_auc_score(y_te, proba)
    auc_sim = auc_by_sampling(y_te.to_numpy(), proba)
    print("[5] ROC-AUC — 임계값을 정하기 전 '모델의 급'")
    print(f"    깡통 모델 AUC   : {auc_dummy:.3f} (동전 던지기 수준)")
    print(f"    로지스틱 AUC    : {auc_model:.3f}")
    print(f"    시뮬레이션 검증 : 무작위 (사기, 정상) 쌍 10,000개 중")
    print(f"                      사기 쪽 확률이 높았던 비율 = {auc_sim:.3f}  (AUC 와 일치)")
    print()
    print("[6] 요약: 불균형 문제의 보고서 필수 3종 세트")
    print("    (1) 기준선(깡통 모델) 성적  (2) 운영 임계값의 혼동행렬  (3) AUC")
    print("    '정확도 98.5%' 한 줄짜리 보고를 보면 이 세 가지를 요구하세요.")
