"""
스케일링과 파이프라인 실험.
단위가 제각각인 churn_table 피처로 KNN 이탈 예측을 하면서
(1) 스케일링 없이 vs 있이, (2) 전처리를 교차검증 밖에서 vs 파이프라인 안에서
두 가지 비교를 통해 '왜 전처리는 파이프라인 안으로'인지 확인합니다.
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


def report(name: str, y_true, proba) -> None:
    """AUC 와 '위험 상위 100명 중 실제 이탈자 수'(캠페인 관점 지표)를 출력."""
    auc = roc_auc_score(y_true, proba)
    top100 = np.asarray(y_true)[np.argsort(proba)[::-1][:100]].sum()
    print(f"    {name:<26} AUC {auc:.3f} / 위험 상위 100명 중 실제 이탈자 {int(top100)}명")


def main() -> None:
    print("=" * 62)
    print(" 스케일링과 파이프라인: 단위 통일 + 누설 원천 봉쇄")
    print("=" * 62)

    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES], df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=42, stratify=y)

    # [1] 피처 단위 확인 --------------------------------------------------
    print("\n[1] 피처들의 스케일 (같은 자로 재고 있는가?)")
    for c in FEATURES:
        print(f"    {c:<18} 범위 {X[c].min():>7,.0f} ~ {X[c].max():>7,.0f}")
    print("    => monthly_fee 가 다른 피처보다 수백~수천 배 큼: 거리 계산을 독점합니다.")

    # [2] 스케일링 없이 KNN ----------------------------------------------
    print("\n[2] 스케일링 없이 KNN (k=15)")
    knn_raw = KNeighborsClassifier(n_neighbors=15)
    knn_raw.fit(X_tr, y_tr)
    report("KNN (스케일링 없음)", y_te, knn_raw.predict_proba(X_te)[:, 1])
    base = int(y_te.sum() / len(y_te) * 100)
    print(f"    (참고: 무작위로 100명을 뽑아도 평균 {base}명은 이탈자입니다)")
    print("    (요금 몇 백 원 차이가 이용일수 30일 차이보다 크게 취급되는 상태)")

    # [3] 파이프라인: StandardScaler + KNN --------------------------------
    print("\n[3] 파이프라인 = 세척(스케일링) -> 조립(모델) 컨베이어 벨트")
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))
    pipe.fit(X_tr, y_tr)          # 스케일러 fit 은 자동으로 학습 데이터에만
    report("KNN + StandardScaler", y_te, pipe.predict_proba(X_te)[:, 1])
    print("    => 같은 모델·같은 데이터, 단위만 통일했는데 성능이 달라집니다.")

    # [4] 잘못된 순서 vs 올바른 순서 (교차검증에서) -----------------------
    print("\n[4] 전처리를 어디서 하는가: 교차검증 5-fold, 지표=AUC")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 잘못된 방식: 전체 데이터를 '먼저' 스케일링 -> 폴드 밖 정보가 새어 듦
    X_scaled_all = pd.DataFrame(StandardScaler().fit_transform(X), columns=FEATURES)
    bad = cross_val_score(KNeighborsClassifier(n_neighbors=15),
                          X_scaled_all, y, cv=cv, scoring="roc_auc")

    # 올바른 방식: 파이프라인째 넣기 -> 폴드마다 학습 부분만으로 다시 fit
    good = cross_val_score(make_pipeline(StandardScaler(),
                                         KNeighborsClassifier(n_neighbors=15)),
                           X, y, cv=cv, scoring="roc_auc")
    print(f"    잘못된 순서(전체 스케일링 후 CV): {bad.mean():.4f} ± {bad.std():.4f}")
    print(f"    올바른 순서(파이프라인째 CV)   : {good.mean():.4f} ± {good.std():.4f}")
    print("    => 스케일링 정도는 차이가 작아 보여도, 타깃 인코딩·결측 대체·피처 선택처럼")
    print("       정답/분포를 진하게 쓰는 전처리에서는 점수가 크게 부풀 수 있습니다.")
    print("       규칙은 하나: '전처리는 전부 파이프라인 안으로'.")

    # [5] 배포 관점: 객체 하나로 예측 -------------------------------------
    print("\n[5] 배포 시뮬레이션: 파이프라인 객체 하나 = 전처리 + 모델")
    new_customers = pd.DataFrame([
        {"tenure_months": 2, "monthly_fee": 29900, "usage_days_30d": 3,
         "support_calls_30d": 4, "plan_changes": 2, "auto_pay": 0},
        {"tenure_months": 36, "monthly_fee": 9900, "usage_days_30d": 28,
         "support_calls_30d": 0, "plan_changes": 0, "auto_pay": 1},
        {"tenure_months": 12, "monthly_fee": 14900, "usage_days_30d": 15,
         "support_calls_30d": 1, "plan_changes": 1, "auto_pay": 1},
    ])
    probs = pipe.predict_proba(new_customers)[:, 1]
    for i, p in enumerate(probs):
        print(f"    신규 고객 {i+1}: 이탈 확률 {p:.1%}")
    print("    => 배포 코드는 pipe.predict_proba(새 데이터) 한 줄. 전처리 누락 사고가 불가능합니다.")
    print("\n    교훈: 파이프라인은 편의 기능이 아니라 누설·불일치 사고를 막는 안전장치입니다.")


if __name__ == "__main__":
    main()
