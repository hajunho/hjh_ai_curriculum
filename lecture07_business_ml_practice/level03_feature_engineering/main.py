"""
피처 엔지니어링의 위력 증명 실험.
fraud_table 에서 같은 모델·같은 분할을 고정한 채
'원본 피처만' vs '원본+파생 피처'의 성능(AUC, PR-AUC)을 비교하고,
마지막에 데이터 누설 피처가 만드는 가짜 성능도 시연합니다.
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
from sklearn.metrics import roc_auc_score, average_precision_score

# '원본' = 결제 승인 전문에 바로 실려 오는 필드 (금액, 시각, 해외 여부)
RAW = ["amount", "hour", "is_foreign"]
# '파생' = 사람이 도메인 지식으로 만들어 붙인 피처
# (거래 이력 집계 피처 tx_count_1h 는 '직접 해보기' 과제로 남겨 둡니다)
DERIVED = ["log_amount", "is_night", "foreign_night"]


def evaluate(X_tr, X_te, y_tr, y_te, label: str) -> tuple[float, float]:
    """모델·전처리를 고정한 채 피처 세트만 바꿔 성능을 잰다 (통제 실험)."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000),
    )
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)
    pr_auc = average_precision_score(y_te, proba)
    print(f"    {label:<24} AUC={auc:.4f}  PR-AUC={pr_auc:.4f}")
    return auc, pr_auc


def main() -> None:
    print("=" * 62)
    print(" 원본 피처 vs 파생 피처 — 같은 모델로 공정하게 겨루기")
    print("=" * 62)

    # [1] 원본 데이터 ----------------------------------------------------
    df = pd.DataFrame(hjh_data.fraud_table(n=5000, seed=11))
    print(f"\n[1] fraud_table {len(df)}건, 사기 비율 {df['is_fraud'].mean():.2%}")
    print(f"    원본 피처(결제 순간 바로 아는 값): {RAW}")

    # [2] 도메인 지식 -> 파생 피처 ---------------------------------------
    print("\n[2] 도메인 지식을 숫자 컬럼으로 번역 (파생 피처 3개)")
    df["log_amount"] = np.log1p(df["amount"])                    # 치우친 금액 -> 배율 감각
    df["is_night"] = ((df["hour"] <= 5) | (df["hour"] >= 23)).astype(int)  # 새벽 플래그
    df["foreign_night"] = df["is_foreign"] * df["is_night"]       # 해외 x 새벽 상호작용
    recipes = {
        "log_amount": "로그 변환 — '2배 큰 금액'을 같은 간격으로",
        "is_night": "구간 플래그 — '새벽 거래는 수상하다'는 현업 상식",
        "foreign_night": "상호작용 — 해외이면서 '동시에' 새벽일 때만 위험",
    }
    for k, v in recipes.items():
        print(f"    - {k:<14}: {v}")

    # [3] 통제 실험: 분할·모델 고정, 피처만 교체 -------------------------
    print("\n[3] 성능 비교 (로지스틱 회귀, 동일 분할, 양성=사기)")
    y = df["is_fraud"]
    idx_tr, idx_te = train_test_split(df.index, test_size=0.3,
                                      random_state=42, stratify=y)
    _, pr_raw = evaluate(df.loc[idx_tr, RAW], df.loc[idx_te, RAW],
                         y.loc[idx_tr], y.loc[idx_te], "원본 3개")
    _, pr_full = evaluate(df.loc[idx_tr, RAW + DERIVED], df.loc[idx_te, RAW + DERIVED],
                          y.loc[idx_tr], y.loc[idx_te], "원본 + 파생 6개")
    print(f"    => PR-AUC {pr_raw:.4f} -> {pr_full:.4f} "
          f"({(pr_full - pr_raw) / pr_raw * 100:+.1f}%) — 모델은 한 글자도 안 바꿨습니다")

    # [4] 어떤 피처가 일하는가 -------------------------------------------
    print("\n[4] 파생 세트 모델의 표준화 계수 (절대값 클수록 영향 큼)")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000),
    )
    cols = RAW + DERIVED
    model.fit(df.loc[idx_tr, cols], y.loc[idx_tr])
    coefs = model.named_steps["logisticregression"].coef_[0]
    for name, c in sorted(zip(cols, coefs), key=lambda t: -abs(t[1])):
        bar = "#" * int(abs(c) * 4)
        print(f"    {name:<14} {c:+.2f} {bar}")

    # [5] 누설 데모: 정답의 그림자를 피처로 넣으면 -----------------------
    print("\n[5] 데이터 누설 시연 — '조사 결과 점수'라는 가짜 피처")
    rng = np.random.default_rng(0)
    # 사기 판정 '후'에나 알 수 있는 값: 정답 + 약간의 잡음 = 전형적 누설 피처
    df["inspection_score"] = df["is_fraud"] * 0.9 + rng.normal(0, 0.1, len(df))
    _, pr_leak = evaluate(df.loc[idx_tr, cols + ["inspection_score"]],
                          df.loc[idx_te, cols + ["inspection_score"]],
                          y.loc[idx_tr], y.loc[idx_te], "파생 + 누설 피처")
    print(f"    => PR-AUC {pr_leak:.4f}: 비현실적으로 완벽 = 축하가 아니라 '버그 경보'입니다.")
    print("       판별 질문: \"예측 시점에 이 값을 알 수 있는가?\" — 아니오면 즉시 제외.")
    print("\n    교훈: 성능의 열쇠는 모델 교체가 아니라 도메인 지식을 피처로 번역하는 일입니다.")


if __name__ == "__main__":
    main()
