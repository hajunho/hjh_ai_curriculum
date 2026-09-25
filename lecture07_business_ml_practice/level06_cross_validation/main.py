"""
교차검증 실험 3종.
(1) 단일 분할 점수가 분할 운에 따라 얼마나 출렁이는지 30회 실험으로 확인하고
(2) 5-fold 교차검증으로 '평균 ± 표준편차' 보고를 만들고
(3) 시계열 데이터에서 무작위 분할이 점수를 부풀리는 것을 TimeSeriesSplit 과 비교합니다.
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import numpy as np
import pandas as pd
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, KFold, TimeSeriesSplit)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


def main() -> None:
    print("=" * 62)
    print(" 교차검증: 성능 숫자의 '운빨'을 측정하고 길들이기")
    print("=" * 62)

    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES], df["churned"]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))

    # [1] 단일 분할 30회: 점수는 확률변수 --------------------------------
    print("\n[1] 같은 모델, 같은 데이터 — 분할 seed 만 30번 바꿔 AUC 측정")
    scores = []
    for seed in range(30):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                                  random_state=seed, stratify=y)
        pipe.fit(X_tr, y_tr)
        scores.append(roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1]))
    scores = np.array(scores)
    print(f"    최소 {scores.min():.4f} / 최대 {scores.max():.4f} / "
          f"폭 {scores.max()-scores.min():.4f} / 표준편차 {scores.std():.4f}")
    # 간이 히스토그램
    bins = np.linspace(scores.min(), scores.max() + 1e-9, 7)
    counts, _ = np.histogram(scores, bins=bins)
    for lo, hi, c in zip(bins[:-1], bins[1:], counts):
        print(f"    {lo:.3f}~{hi:.3f} | {'#' * c}")
    print("    => '단일 점수 0.87' 뒤에는 이만큼의 운이 숨어 있습니다.")

    # [2] 5-fold 교차검증 --------------------------------------------------
    print("\n[2] 5-fold StratifiedKFold 교차검증 (파이프라인째 투입)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print("    폴드별 AUC:", " ".join(f"{s:.4f}" for s in cv_scores))
    print(f"    보고 형식 => AUC {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # [3] 두 모델의 공정 비교 ----------------------------------------------
    print("\n[3] 모델 비교: 로지스틱 회귀 vs 랜덤포레스트 (같은 CV)")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
    print(f"    로지스틱 회귀 : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    랜덤포레스트  : {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
    diff = abs(cv_scores.mean() - rf_scores.mean())
    noise = max(cv_scores.std(), rf_scores.std())
    verdict = "의미 있는 차이로 보기 어렵습니다 (평균 차이 < 편차)" if diff < noise \
        else "차이가 편차보다 커서 의미가 있어 보입니다"
    print(f"    평균 차이 {diff:.4f} vs 편차 {noise:.4f} => {verdict}")

    # [4] 시계열 함정: 무작위 KFold vs TimeSeriesSplit ---------------------
    print("\n[4] 시계열 데이터: 미래로 과거를 예측하면 점수가 부풉니다")
    sales = pd.DataFrame(hjh_data.sales_table(n_days=365, seed=42))
    sales = sales.dropna(subset=["revenue"])
    sales = sales[sales["revenue"] > 0]
    daily = sales.groupby("day_index")["revenue"].sum().reset_index()
    # 시차(lag) 피처: 전일·이동평균 -> 무작위 분할 시 검증 정보가 학습 피처에 스며듦
    daily["lag1"] = daily["revenue"].shift(1)
    daily["ma7"] = daily["revenue"].shift(1).rolling(7).mean()
    daily = daily.dropna().reset_index(drop=True)
    Xs, ys = daily[["lag1", "ma7"]], daily["revenue"]

    model = Ridge(alpha=1.0)
    shuffled = cross_val_score(model, Xs, ys, scoring="r2",
                               cv=KFold(n_splits=5, shuffle=True, random_state=42))
    tssplit = TimeSeriesSplit(n_splits=5)
    ordered = cross_val_score(model, Xs, ys, scoring="r2", cv=tssplit)
    print(f"    무작위 KFold(shuffle)   R2: {shuffled.mean():.4f} ± {shuffled.std():.4f}")
    print(f"    TimeSeriesSplit(순서 유지) R2: {ordered.mean():.4f} ± {ordered.std():.4f}")
    print("    (R2 < 0 = '평균으로 찍기'보다 못하다는 뜻. 전일 매출만으로는 미래 예측이")
    print("     어렵다는 정직한 성적표이고, 무작위 분할은 그 사실을 숨겼던 것입니다.)")
    print("    TimeSeriesSplit 의 폴드 구조 (학습은 항상 검증보다 과거):")
    for i, (tr, te) in enumerate(tssplit.split(Xs)):
        print(f"      fold{i+1}: 학습 day {tr.min()}~{tr.max()} -> 검증 day {te.min()}~{te.max()}")
    print("\n    교훈: 성능은 한 번 재면 '운', 여러 번 재면 '실력 ± 오차'가 됩니다.")
    print("          그리고 시간 축이 있으면 시험지는 반드시 미래에서 가져오세요.")


if __name__ == "__main__":
    main()
