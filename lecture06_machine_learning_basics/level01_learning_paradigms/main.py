"""
level01 — 지도·비지도·강화학습 미니 체험

세 가지 학습 패러다임을 장난감 크기 예제로 하나씩 돌려 봅니다.
  [1] 지도학습   = 과외 수업   (정답 레이블을 주고 이탈 예측 훈련)
  [2] 비지도학습 = 자율 스터디 (정답 없이 고객을 그룹으로 묶기)
  [3] 강화학습   = 강아지 훈련 (보상만 보고 최적 쿠폰을 찾는 밴딧)
"""

import pathlib
import random
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

FEATURES = ["tenure_months", "usage_days_30d", "support_calls_30d", "auto_pay"]


def demo_supervised() -> None:
    """[1] 지도학습: 문제(고객 특징)와 정답지(churned)를 함께 주고 가르친다."""
    rows = hjh_data.churn_table(n=2000, seed=7)
    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)
    y = np.array([r["churned"] for r in rows])

    scaler = StandardScaler()
    model = LogisticRegression(random_state=0)
    model.fit(scaler.fit_transform(X), y)          # <- 정답 y 를 넘긴다 (과외)

    print("[1] 지도학습 = 과외 수업 (정답지를 보며 패턴 학습)")
    print("    훈련: 고객 2000명의 특징 + 이탈 여부 정답")
    print("    시험: 처음 보는 고객 3명의 이탈 확률 예측")
    new_customers = [
        ("성실 이용 고객", [36, 28, 0, 1]),
        ("저이용 + 문의 폭주", [3, 2, 5, 0]),
        ("평범한 고객", [18, 15, 1, 1]),
    ]
    for name, feat in new_customers:
        p = model.predict_proba(scaler.transform([feat]))[0, 1]
        verdict = "이탈 위험" if p >= 0.5 else "유지 예상"
        print(f"      {name:12s} -> 이탈 확률 {p:5.1%} ({verdict})")
    print()


def demo_unsupervised() -> None:
    """[2] 비지도학습: 정답 열을 빼고 '비슷한 고객끼리 묶어 보라'고만 시킨다."""
    rows = hjh_data.churn_table(n=2000, seed=7)
    cols = ["tenure_months", "usage_days_30d", "support_calls_30d", "monthly_fee"]
    X = np.array([[r[c] for c in cols] for r in rows], dtype=float)  # 정답 없음!

    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=3, n_init=10, random_state=0)
    labels = km.fit_predict(Xs)                     # <- y 가 없다 (자율 스터디)

    print("[2] 비지도학습 = 자율 스터디 (정답 없이 구조 찾기)")
    print("    k-means 가 고객 2000명을 3그룹으로 묶음. 그룹별 평균 프로필:")
    print("      그룹  인원   가입개월  이용일수  문의수  월요금")
    for g in range(3):
        member = X[labels == g]
        m = member.mean(axis=0)
        print(f"      {g:>2}   {len(member):>4}   {m[0]:7.1f}  {m[1]:7.1f}  {m[2]:6.2f}  {m[3]:7.0f}")
    print("    -> 각 그룹에 이름을 붙이는 것(예: '우량', '휴면 위험')은 사람의 몫입니다.")
    print()


def demo_reinforcement() -> None:
    """[3] 강화학습: 어떤 쿠폰이 좋은지 모른 채, 보내 보고 반응(보상)으로 배운다.
    멀티암드 밴딧 + epsilon-greedy 를 순수 파이썬으로 구현."""
    rng = random.Random(42)
    true_rates = {"A. 5%할인": 0.05, "B. 무료배송": 0.12, "C. 1+1": 0.08}
    coupons = list(true_rates)
    counts = {c: 0 for c in coupons}       # 각 쿠폰을 보낸 횟수
    wins = {c: 0 for c in coupons}         # 각 쿠폰이 사용된 횟수
    EPSILON = 0.1                          # 10% 확률로 탐험(아무거나 시도)
    history = []

    for _ in range(1000):
        if rng.random() < EPSILON or not any(counts.values()):
            choice = rng.choice(coupons)                       # 탐험
        else:
            choice = max(coupons, key=lambda c: wins[c] / max(counts[c], 1))  # 활용
        reward = 1 if rng.random() < true_rates[choice] else 0  # 고객 반응 = 간식
        counts[choice] += 1
        wins[choice] += reward
        history.append(reward)

    print("[3] 강화학습 = 강아지 훈련 (시도 -> 보상 -> 행동 개선)")
    print("    쿠폰 3종의 진짜 사용률은 비밀. 1000명에게 보내며 스스로 학습:")
    for c in coupons:
        est = wins[c] / max(counts[c], 1)
        print(f"      {c:10s} 발송 {counts[c]:>4}회, 추정 사용률 {est:5.1%} (진실 {true_rates[c]:.0%})")
    early = sum(history[:100]) / 100
    late = sum(history[-100:]) / 100
    print(f"    초반 100회 평균 보상 {early:.2f}  ->  마지막 100회 평균 보상 {late:.2f}")
    print("    -> 정답을 가르쳐 준 적이 없는데도 최선의 행동(B)에 수렴합니다.")
    print()


if __name__ == "__main__":
    np.random.seed(0)   # 재현성
    demo_supervised()
    demo_unsupervised()
    demo_reinforcement()

    print("[4] 요약")
    print("    패러다임     주어진 것            배우는 것")
    print("    지도학습     입력 + 정답 레이블   입력 -> 정답 대응 (과외)")
    print("    비지도학습   입력만               숨은 구조/그룹     (자율 스터디)")
    print("    강화학습     환경 + 보상          보상 최대 행동     (강아지 훈련)")
    print("    실무 ML 의 대부분은 지도학습이며, level03 부터 본격적으로 다룹니다.")
