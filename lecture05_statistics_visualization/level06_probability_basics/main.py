"""
level06 — 확률의 기초

세 가지 시뮬레이션으로 직관과 계산을 대결시킵니다.
1) 큰 수의 법칙: 동전 던지기 비율의 수렴
2) 질병 검사의 역설: 양성 판정자 중 진짜 병자는 몇 %인가 (베이즈 맛보기)
3) 몬티홀 문제: 바꾸기 vs 유지 전략의 승률
"""

import numpy as np


def law_of_large_numbers() -> None:
    """동전 던지기 횟수가 늘수록 앞면 비율이 0.5 에 다가갑니다."""
    rng = np.random.default_rng(606)  # seed 고정
    flips = rng.integers(0, 2, size=10_000)  # 1=앞면
    print("[1] 큰 수의 법칙 — 동전 앞면 비율의 수렴")
    for n in [10, 100, 1_000, 10_000]:
        ratio = flips[:n].mean()
        print(f"    {n:>6,}회 던짐: 앞면 비율 {ratio:.4f} (0.5 와의 거리 {abs(ratio - 0.5):.4f})")
    print("    -> 확률 0.5 는 '다음 번' 예언이 아니라 '장기 비율'의 약속입니다.")


def disease_test_paradox() -> None:
    """유병률 1%, 민감도 99%, 특이도 95% 검사를 10만 명에게 시행합니다."""
    rng = np.random.default_rng(607)
    n = 100_000
    prevalence = 0.01     # 유병률: 병자인 사전확률
    sensitivity = 0.99    # P(양성|병) — 병자를 잡아낼 확률
    specificity = 0.95    # P(음성|건강) — 건강한 사람을 통과시킬 확률

    sick = rng.random(n) < prevalence
    # 병자는 민감도 확률로 양성, 건강한 사람은 (1-특이도) 확률로 억울한 양성
    positive = np.where(sick,
                        rng.random(n) < sensitivity,
                        rng.random(n) < (1 - specificity))

    n_pos = positive.sum()
    n_true = (sick & positive).sum()          # 진짜 병자인 양성
    n_false = (~sick & positive).sum()        # 억울한 양성(오탐)
    p_sick_given_pos = n_true / n_pos

    # 베이즈 정리로 구한 이론값과 비교
    theory = (prevalence * sensitivity) / (
        prevalence * sensitivity + (1 - prevalence) * (1 - specificity))

    print()
    print("[2] 질병 검사의 역설 — 가상 인구 100,000명 시뮬레이션")
    print(f"    병자 {sick.sum():,}명 / 건강 {(~sick).sum():,}명 (유병률 {prevalence:.0%})")
    print(f"    양성 판정 {n_pos:,}명 = 진짜 병자 {n_true:,}명 + 억울한 양성 {n_false:,}명")
    print(f"    P(병 | 양성) 시뮬레이션 = {p_sick_given_pos:.3f}")
    print(f"    P(병 | 양성) 베이즈 이론값 = {theory:.3f}")
    print("    -> '99% 정확한 검사'에서 양성이어도 실제 병일 확률은 약 17%!")
    print("       건강한 사람이 압도적으로 많아, 그들의 5% 오탐이 병자 수를 압도합니다.")
    print("       (P(양성|병)=0.99 와 P(병|양성)=0.17 은 전혀 다른 확률입니다)")


def monty_hall(n_games: int = 10_000) -> None:
    """몬티홀: 문 3개, 사회자는 꽝 문을 알고 열어 줍니다."""
    rng = np.random.default_rng(608)
    prize = rng.integers(0, 3, size=n_games)       # 상품이 든 문
    first_pick = rng.integers(0, 3, size=n_games)  # 참가자의 첫 선택

    # '유지' 전략: 첫 선택이 정답이면 승리
    stay_wins = (first_pick == prize).sum()
    # '바꾸기' 전략: 첫 선택이 정답이 '아니면' 승리
    #   (사회자가 남은 꽝을 열어 주므로, 바꾸면 반드시 남은 문 = 정답)
    switch_wins = (first_pick != prize).sum()

    print()
    print(f"[3] 몬티홀 문제 — {n_games:,}판 시뮬레이션")
    print(f"    유지 전략   승률: {stay_wins / n_games:.3f} (이론값 1/3 = 0.333)")
    print(f"    바꾸기 전략 승률: {switch_wins / n_games:.3f} (이론값 2/3 = 0.667)")
    print("    -> 사회자는 '꽝을 골라서' 열 수 있으므로 그 행동 자체가 정보입니다.")
    print("       직관이 50:50 이라 우겨도, 만 판의 장부는 2/3 라고 말합니다.")


def main() -> None:
    law_of_large_numbers()
    disease_test_paradox()
    monty_hall()
    print()
    print("[4] 오늘의 교훈: 확률 논쟁이 붙으면 공식보다 먼저 시뮬레이션을 돌려 보세요.")
    print("    '만들어서 세어 보기'는 언제나 통하는 확률 계산기입니다.")


if __name__ == "__main__":
    main()
