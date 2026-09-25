"""
level04 — 학습·검증·테스트 분리: 자기 채점의 낙관 편향을 실험으로 증명

실험 A: 훈련 데이터로 채점하면 성적이 얼마나 부풀려지나 (트리 깊이별 gap)
실험 B: 테스트 세트를 '훔쳐보며' 설정을 고르면 최종 보고 성적이
        얼마나 체계적으로 부풀려지나 (30회 반복 통계)
교훈: 훈련=교과서, 검증=모의고사, 테스트=수능(마지막 1번).
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

K_CANDIDATES = list(range(1, 30, 2))     # 설정(하이퍼파라미터) 후보 15개


def make_data(seed: int, n: int = 1500):
    """잡음이 섞인 이진 분류 합성 데이터 (다운로드 없음, seed 로 재현)."""
    return make_classification(
        n_samples=n, n_features=8, n_informative=4, n_redundant=2,
        flip_y=0.08,             # 8% 는 레이블 자체가 잡음 -> 암기 유혹 발생
        class_sep=0.9, random_state=seed)


def experiment_a() -> None:
    """[2] 실험 A: 훈련 성적 vs 테스트 성적 — 유연할수록 벌어지는 간격."""
    X, y = make_data(seed=0, n=1000)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    print("[2] 실험 A — 자기 채점(훈련 데이터 채점)의 낙관 편향")
    print("    트리 깊이   훈련 정확도   테스트 정확도   간격(gap)")
    for depth in [1, 2, 4, 8, 16, None]:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_tr, y_tr)
        acc_tr = tree.score(X_tr, y_tr)
        acc_te = tree.score(X_te, y_te)
        label = "무제한" if depth is None else f"{depth:>4}"
        print(f"    {label:>7}      {acc_tr:6.1%}        {acc_te:6.1%}       {acc_tr - acc_te:+6.1%}")
    print("    -> 깊어질수록 훈련 성적은 100%로 가지만(기출 암기) 실전 성적은 하락.")
    print("       훈련 성적은 실력의 증명이 아닙니다.\n")


def experiment_b(n_repeats: int = 30) -> None:
    """[3] 실험 B: 테스트 훔쳐보기 vs 올바른 3분할 — 30회 반복 통계.
    '새 데이터' 세트(모델 선택에 전혀 안 쓴 데이터)를 진짜 실력으로 간주."""
    inflate_peek, inflate_proper = [], []
    for rep in range(n_repeats):
        X, y = make_data(seed=100 + rep)
        # 60:20:20 분할 + 진짜 실력 측정용 '새 데이터' 500건은 별도 생성
        X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, random_state=rep)
        X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=rep)
        X_new, y_new = make_data(seed=9000 + rep, n=800)   # 배포 후 만나는 데이터

        models = {k: KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr)
                  for k in K_CANDIDATES}

        # (a) 반칙: 테스트 성적을 보며 k 를 고르고, 그 성적을 그대로 보고
        k_peek = max(K_CANDIDATES, key=lambda k: models[k].score(X_te, y_te))
        reported_peek = models[k_peek].score(X_te, y_te)
        real_peek = models[k_peek].score(X_new, y_new)
        inflate_peek.append(reported_peek - real_peek)

        # (b) 정석: 검증으로 k 를 고르고, 테스트는 마지막에 딱 1번
        k_ok = max(K_CANDIDATES, key=lambda k: models[k].score(X_va, y_va))
        reported_ok = models[k_ok].score(X_te, y_te)
        real_ok = models[k_ok].score(X_new, y_new)
        inflate_proper.append(reported_ok - real_ok)

    peek = np.array(inflate_peek)
    proper = np.array(inflate_proper)
    print(f"[3] 실험 B — 설정 k 후보 {len(K_CANDIDATES)}개를 고르는 두 방식, {n_repeats}회 반복")
    print("    부풀림 = (보고한 성적) - (새 데이터에서의 진짜 성적)")
    print(f"    (a) 테스트를 훔쳐보며 선택: 평균 부풀림 {peek.mean():+.2%} (표준편차 {peek.std():.2%})")
    print(f"    (b) 검증으로 선택(정석)  : 평균 부풀림 {proper.mean():+.2%} (표준편차 {proper.std():.2%})")
    print(f"    (a)가 (b)보다 부풀려진 횟수: {int((peek > proper).sum())}/{n_repeats}회")
    print("    -> 훔쳐보기의 부풀림은 우연이 아니라 구조적입니다.")
    print("       후보가 많을수록 '그 시험지에서 우연히 잘 본 설정'을 고르게 됩니다.\n")


if __name__ == "__main__":
    np.random.seed(0)

    # [1] 3분할 소개 --------------------------------------------------------
    X, y = make_data(seed=0, n=1000)
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=0)
    print("[1] 데이터 1000건을 60:20:20 으로 3분할")
    print(f"    훈련(교과서) {len(X_tr)}건 / 검증(모의고사) {len(X_va)}건 / 테스트(수능) {len(X_te)}건\n")

    experiment_a()
    experiment_b()

    print("[4] 요약")
    print("    1. 훈련 데이터 채점은 항상 낙관적이다 (실험 A의 gap).")
    print("    2. 테스트로 설정을 고르면 최종 보고도 오염된다 (실험 B).")
    print("    3. 보고서의 성능 숫자를 보면 반드시 물어라:")
    print("       \"그 숫자, 어떤 데이터로 잰 겁니까? 그 데이터를 몇 번 봤습니까?\"")
