"""
실전 3 — 이상거래 탐지: 지도학습과 비지도(IsolationForest) 병행.
같은 알람 예산에서 두 접근의 탐지력을 공정 비교하고,
'신종 의심 구역'(지도는 낮게, 비지도는 높게 본 거래)을 확인한 뒤
조사팀 알람 예산에 따른 재현율 한계 효용 표로 운영 임계값을 설계합니다.
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# level07 과 같은 가정: 이력 집계(tx_count_1h) 시스템이 아직 없는 상태에서
# 결제 순간의 필드 + level03 의 파생 피처로 승부합니다.
FEATURES = ["log_amount", "hour", "is_foreign", "is_night"]


def topk_stats(scores, y_true, k):
    """의심 점수 상위 k건만 알람할 때 (잡은 사기, 정밀도, 재현율)."""
    idx = np.argsort(scores)[::-1][:k]
    caught = int(np.asarray(y_true)[idx].sum())
    total = int(np.asarray(y_true).sum())
    return caught, caught / k, caught / total


def main() -> None:
    print("=" * 68)
    print(" 실전 3: 이상거래 탐지 — 사진첩(지도) + 감(비지도) + 예산(운영)")
    print("=" * 68)

    # [1] 데이터와 피처 -----------------------------------------------------
    df = pd.DataFrame(hjh_data.fraud_table(n=5000, seed=11))
    df["log_amount"] = np.log1p(df["amount"])
    df["is_night"] = ((df["hour"] <= 5) | (df["hour"] >= 23)).astype(int)
    X, y = df[FEATURES], df["is_fraud"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=42, stratify=y)
    print(f"\n[1] 거래 {len(df)}건 (사기 {y.mean():.2%}) / 테스트 {len(y_te)}건에 사기 {y_te.sum()}건")
    print(f"    피처: {FEATURES}")

    # [2] 지도학습: 라벨(사진첩)로 배우는 베테랑 ---------------------------
    print("\n[2] 지도학습 — 과거 적발 라벨로 학습 (로지스틱 회귀)")
    sup = make_pipeline(StandardScaler(),
                        LogisticRegression(random_state=42, class_weight="balanced",
                                           max_iter=1000))
    sup.fit(X_tr, y_tr)
    sup_score = sup.predict_proba(X_te)[:, 1]
    print("    -> 거래마다 사기 확률(의심 점수)을 산출했습니다.")

    # [3] 비지도: 라벨 없이 '평소와 다름'을 찾는 신입 -----------------------
    print("\n[3] 비지도 — IsolationForest, 라벨을 전혀 쓰지 않음")
    iso = IsolationForest(n_estimators=200, contamination=0.015, random_state=42)
    iso.fit(X_tr)                      # y_tr 없음! 정상 거래의 모양만 학습
    iso_score = -iso.score_samples(X_te)   # 클수록 이상 (부호 뒤집기)
    print("    -> '몇 번 만에 고립되는가'로 이상 점수를 산출했습니다.")

    # [4] 같은 알람 예산에서 공정 비교 --------------------------------------
    budget = 20
    print(f"\n[4] 공정 비교: 테스트 {len(y_te)}건 중 알람 예산 {budget}건일 때")
    for name, sc in [("지도학습(라벨 사용)", sup_score), ("IsolationForest(라벨 없음)", iso_score)]:
        caught, prec, rec = topk_stats(sc, y_te, budget)
        print(f"    {name:<26} 잡은 사기 {caught:2d}건 | 정밀도 {prec:5.1%} | 재현율 {rec:5.1%}")
    print("    => 라벨은 힘이 셉니다. 그러나 비지도는 '라벨에 없는 수법'이라는")
    print("       다른 각도를 봅니다. 승자독식이 아니라 역할 분담입니다.")

    # [5] 신종 의심 구역: 지도는 낮게, 비지도만 높게 본 거래 ---------------
    print("\n[5] 신종 의심 구역 — 지도 점수 하위 50%인데 비지도 점수 상위 5%")
    sup_rank = pd.Series(sup_score).rank(pct=True)
    iso_rank = pd.Series(iso_score).rank(pct=True)
    novel = (sup_rank < 0.5) & (iso_rank > 0.95)
    zone = X_te.reset_index(drop=True)[novel]
    zone_y = y_te.reset_index(drop=True)[novel]
    print(f"    해당 거래 {novel.sum()}건 (그중 실제 사기 {int(zone_y.sum())}건)")
    if len(zone) > 0:
        show = zone.head(3).copy()
        show["실제사기"] = zone_y.head(3).values
        print(show.to_string())
    print("    => 이 구역은 소량 샘플 조사 큐로 보냅니다. 조사 결과가 새 라벨이 되어")
    print("       지도 모델을 다시 가르치는 선순환이 실무 운영의 핵심입니다.")

    # [6] 알람 예산별 한계 효용과 운영 임계값 -------------------------------
    print("\n[6] 알람 예산을 늘리면 얼마나 더 잡나 (지도 모델 기준)")
    print(f"    {'예산':>4} | {'잡은 사기':>6} | {'정밀도':>6} | {'재현율':>6} | 예산 대비 임계값")
    prev_caught = 0
    for k in [10, 20, 30, 40, 60]:
        caught, prec, rec = topk_stats(sup_score, y_te, k)
        th = np.sort(sup_score)[::-1][k - 1]
        gain = caught - prev_caught
        print(f"    {k:>4} | {caught:>5}건 | {prec:6.1%} | {rec:6.1%} | "
              f"점수 {th:.3f} 이상 (직전 대비 +{gain}건)")
        prev_caught = caught
    print("\n    보고 문장 예시:")
    c20, p20, r20 = topk_stats(sup_score, y_te, 20)
    c40, p40, r40 = topk_stats(sup_score, y_te, 40)
    print(f'    "현재 인력(알람 20건)으로 사기의 {r20:.0%}를 잡습니다. 조사 인력을 2배로')
    print(f'     늘리면(40건) 재현율은 {r40:.0%}가 되지만 정밀도는 {p40:.0%}로 내려갑니다."')
    print("\n    교훈: 임계값은 통계에서 나오지 않습니다. 인력 계획(예산)에서 나옵니다.")


if __name__ == "__main__":
    main()
