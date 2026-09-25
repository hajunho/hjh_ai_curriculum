"""
Lecture 08 · Level 00 — 신경망이 왜 등장했나
선형 모델이 AND/OR 는 완벽히 풀지만 XOR 은 정확도 75% 를
절대 넘지 못한다는 사실을 최소제곱법과 전수 탐색으로 증명합니다.
마지막으로 좌표 변환(은닉 표현)을 손으로 넣어 주면 같은 선형 모델이
XOR 을 100% 로 푸는 것을 보여 주며, 표현 학습이라는 발상으로 연결합니다.
"""

import numpy as np

np.random.seed(42)  # 재현성을 위해 난수 시드 고정 (이 레벨은 난수 의존이 거의 없지만 규칙으로 고정)

# 입력: (x1, x2) 네 가지 조합
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)

# 정답표: 세 가지 논리 규칙
TARGETS = {
    "AND": np.array([0, 0, 0, 1]),
    "OR":  np.array([0, 1, 1, 1]),
    "XOR": np.array([0, 1, 1, 0]),
}


def fit_least_squares(features, y):
    """편향(bias) 항을 붙인 뒤 최소제곱법으로 선형 모델의 최적 계수를 구합니다."""
    A = np.hstack([features, np.ones((len(features), 1))])  # 마지막 열 1 = 편향 자리
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef  # (w1, w2, ..., b)


def linear_accuracy(features, y, coef):
    """점수가 0.5 이상이면 1 로 분류했을 때의 정확도와 예측을 돌려줍니다."""
    A = np.hstack([features, np.ones((len(features), 1))])
    pred = (A @ coef >= 0.5).astype(int)
    return float((pred == y).mean()), pred


def best_linear_accuracy(features, y, n_w=61, n_b=61, limit=3.0):
    """가중치·편향의 격자를 전수 탐색하여 선형 결정경계가 낼 수 있는 최고 정확도를 찾습니다.

    점수 = w1*x1 + w2*x2 + b, 점수 >= 0 이면 1 로 분류합니다.
    numpy 브로드캐스팅으로 (n_w * n_w * n_b) 개 조합을 한 번에 평가합니다.
    """
    w_grid = np.linspace(-limit, limit, n_w)
    b_grid = np.linspace(-limit, limit, n_b)
    W1, W2, B = np.meshgrid(w_grid, w_grid, b_grid, indexing="ij")
    # scores 의 마지막 축이 데이터 4개: shape = (n_w, n_w, n_b, 4)
    scores = (W1[..., None] * features[:, 0]
              + W2[..., None] * features[:, 1]
              + B[..., None])
    pred = (scores >= 0.0).astype(int)
    acc = (pred == y).mean(axis=-1)
    best_idx = np.unravel_index(np.argmax(acc), acc.shape)
    best = float(acc[best_idx])
    n_tried = acc.size
    return best, n_tried


def hidden_features(features):
    """손으로 설계한 은닉 표현(hidden representation) 두 개를 만듭니다.

    h1 은 OR 처럼(합이 0.5 이상이면 1), h2 는 AND 처럼(합이 1.5 이상이면 1) 동작합니다.
    나중에 배울 은닉층 뉴런 두 개가 하는 일과 정확히 같습니다.
    """
    s = features[:, 0] + features[:, 1]
    h1 = (s >= 0.5).astype(float)  # OR 역할
    h2 = (s >= 1.5).astype(float)  # AND 역할
    return np.stack([h1, h2], axis=1)


def main():
    print("[1] 진리표 데이터 준비 — 입력 (x1, x2) 네 가지 조합과 세 가지 규칙의 정답")
    print("    x1 x2 | AND OR XOR")
    for i in range(len(X)):
        print(f"     {int(X[i, 0])}  {int(X[i, 1])} |  {TARGETS['AND'][i]}   {TARGETS['OR'][i]}   {TARGETS['XOR'][i]}")

    print()
    print("[2] 최소제곱법으로 각 문제에 '가장 잘 맞는' 선형 모델을 학습")
    for name, y in TARGETS.items():
        coef = fit_least_squares(X, y)
        acc, pred = linear_accuracy(X, y, coef)
        print(f"    {name:>3}: w1={coef[0]:+.3f}, w2={coef[1]:+.3f}, b={coef[2]:+.3f}"
              f" -> 정확도 {acc * 100:5.1f}%  (예측 {pred.tolist()}, 정답 {y.tolist()})")
    print("    => AND/OR 는 100%, XOR 은 직선 하나로는 네 점을 다 맞히지 못합니다.")

    print()
    print("[3] '더 좋은 직선이 숨어 있지 않을까?' — 가중치·편향 격자 전수 탐색")
    for name, y in TARGETS.items():
        best, n_tried = best_linear_accuracy(X, y)
        print(f"    {name:>3}: {n_tried:,}개 (w1, w2, b) 조합을 모두 시도 -> 최고 정확도 {best * 100:5.1f}%")
    print("    => XOR 은 어떤 직선을 골라도 75% 가 상한입니다. 노력 부족이 아니라 구조적 한계입니다.")

    print()
    print("[4] 좌표 변환(은닉 표현)을 손으로 넣어 주면?")
    H = hidden_features(X)
    print("    새 좌표 h1=OR(x1,x2), h2=AND(x1,x2) 로 데이터를 옮깁니다.")
    print("    x1 x2 -> h1 h2 | XOR 정답")
    for i in range(len(X)):
        print(f"     {int(X[i, 0])}  {int(X[i, 1])} ->  {int(H[i, 0])}  {int(H[i, 1])} |    {TARGETS['XOR'][i]}")
    coef_h = fit_least_squares(H, TARGETS["XOR"])
    acc_h, pred_h = linear_accuracy(H, TARGETS["XOR"], coef_h)
    best_h, _ = best_linear_accuracy(H, TARGETS["XOR"])
    print(f"    은닉 표현 위의 선형 모델: w1={coef_h[0]:+.3f}, w2={coef_h[1]:+.3f}, b={coef_h[2]:+.3f}"
          f" -> 정확도 {acc_h * 100:.1f}%")
    print(f"    (전수 탐색으로도 최고 정확도 {best_h * 100:.1f}% 확인 — 사실상 y = h1 - h2 한 줄)")

    print()
    print("[5] 결론")
    print("    - 같은 선형 모델인데 좌표(표현)만 바꾸니 XOR 정확도가 75% -> 100% 가 되었습니다.")
    print("    - 즉 모델이 약했던 것이 아니라 데이터를 바라보는 표현이 나빴던 것입니다.")
    print("    - 이 좌표 변환을 사람이 아니라 기계가 데이터에서 배우게 하자는 것이")
    print("      표현 학습(representation learning), 곧 신경망과 딥러닝의 출발점입니다.")


if __name__ == "__main__":
    main()
