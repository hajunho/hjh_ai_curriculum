"""
역전파(backpropagation)를 numpy 만으로 밑바닥부터 구현합니다.
2층 신경망(2-8-1, tanh + sigmoid)의 순전파와 역전파를 손으로 짜서
선형 모델이 못 풀던 XOR 4점을 100% 분류할 때까지 학습하고,
수치미분(중앙차분)으로 해석적 그래디언트가 맞는지 검증합니다.
"""

import numpy as np

rng = np.random.default_rng(0)   # 재현성: 시드 고정


# ----------------------------- 신경망 부품 -----------------------------

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def init_params():
    """2-8-1 신경망의 가중치를 작은 난수로 초기화합니다."""
    return {
        "W1": rng.normal(0, 0.8, size=(2, 8)),
        "b1": np.zeros(8),
        "W2": rng.normal(0, 0.8, size=(8, 1)),
        "b2": np.zeros(1),
    }


def forward(params, X):
    """순전파: 입력 -> 은닉(tanh) -> 출력(sigmoid). 중간값은 역전파용으로 보관."""
    z1 = X @ params["W1"] + params["b1"]      # (n, 8)
    h = np.tanh(z1)                           # 은닉 표현
    z2 = h @ params["W2"] + params["b2"]      # (n, 1)
    p = sigmoid(z2)                           # 이탈 확률처럼 0~1
    cache = {"X": X, "z1": z1, "h": h, "p": p}
    return p, cache


def bce_loss(p, y):
    """이진 크로스엔트로피: 확신에 찬 오답에 큰 벌점."""
    eps = 1e-9
    return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


def backward(params, cache, y):
    """역전파: 출력의 오차 책임을 연쇄법칙으로 거꾸로 배분합니다."""
    X, h, p = cache["X"], cache["h"], cache["p"]
    n = X.shape[0]

    # 출력층: BCE+sigmoid 조합의 미분은 (p - y) 로 깔끔하게 정리됩니다
    dz2 = (p - y) / n                          # (n, 1)
    grads = {
        "W2": h.T @ dz2,                       # 은닉 출력이 기여한 만큼 책임 배분
        "b2": dz2.sum(axis=0),
    }
    # 은닉층: 출력층에서 넘어온 책임 × tanh 통과 시 줄어든 비율
    dh = dz2 @ params["W2"].T                  # (n, 8)
    dz1 = dh * (1.0 - h ** 2)                  # tanh'(z) = 1 - tanh(z)^2
    grads["W1"] = X.T @ dz1
    grads["b1"] = dz1.sum(axis=0)
    return grads


def numerical_grads(params, X, y, key, h_eps=1e-6):
    """수치미분 검산: 파라미터를 눈곱만큼 움직여 loss 변화량을 직접 잽니다."""
    W = params[key]
    num = np.zeros_like(W)
    it = np.nditer(W, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = W[idx]
        W[idx] = orig + h_eps
        loss_plus = bce_loss(forward(params, X)[0], y)
        W[idx] = orig - h_eps
        loss_minus = bce_loss(forward(params, X)[0], y)
        W[idx] = orig
        num[idx] = (loss_plus - loss_minus) / (2 * h_eps)
        it.iternext()
    return num


# ------------------------------- 실습 시나리오 -------------------------------

def main():
    # XOR: 선형 모델(level00)이 절대 못 풀던 문제
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    params = init_params()

    print("[1] 문제: XOR — (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0")
    print("    2-8-1 신경망(tanh + sigmoid)을 순전파+역전파 수제 구현으로 학습합니다.\n")

    print("[2] 그래디언트 검증 — 역전파(해석) vs 수치미분(실측)")
    p, cache = forward(params, X)
    ana = backward(params, cache, y)
    worst = 0.0
    for key in ["W1", "b1", "W2", "b2"]:
        num = numerical_grads(params, X, y, key)
        # 상대오차: |해석 - 실측| / (|해석| + |실측|)
        denom = np.abs(ana[key]) + np.abs(num) + 1e-12
        rel = float(np.max(np.abs(ana[key] - num) / denom))
        worst = max(worst, rel)
        print(f"    {key}: 최대 상대오차 = {rel:.2e}")
    assert worst < 1e-4, "역전파 구현에 버그가 있습니다!"
    print("    => 전부 1e-4 미만. 수제 역전파가 정확합니다.\n")

    print("[3] 학습 시작 (경사하강, 학습률 0.5)")
    lr, n_epochs = 0.5, 3000
    for epoch in range(1, n_epochs + 1):
        p, cache = forward(params, X)
        grads = backward(params, cache, y)
        for key in params:                       # 모든 가중치를 경사 반대 방향으로
            params[key] -= lr * grads[key]
        if epoch in (1, 10, 100, 500, 1000, 2000, 3000):
            acc = float(np.mean((p > 0.5) == y))
            print(f"    epoch {epoch:4d}: loss = {bce_loss(p, y):.4f}, 정확도 = {acc:.0%}")

    print("\n[4] 학습 결과 — 4개 입력에 대한 예측 확률")
    p, _ = forward(params, X)
    for xi, yi, pi in zip(X, y, p):
        mark = "O" if (pi[0] > 0.5) == bool(yi[0]) else "X"
        print(f"    입력 {xi} -> 예측 p = {pi[0]:.4f} (정답 {int(yi[0])}) {mark}")
    acc = float(np.mean((p > 0.5) == y))
    assert acc == 1.0, "XOR 수렴 실패"
    print(f"    최종 정확도: {acc:.0%} — 선형 모델의 75% 벽을 넘었습니다.\n")

    print("[5] 은닉층이 만든 '새 좌표'(표현) — 각 입력의 은닉값 일부")
    _, cache = forward(params, X)
    for xi, hi in zip(X, cache["h"]):
        print(f"    {xi} -> h[:3] = [{hi[0]:+.2f}, {hi[1]:+.2f}, {hi[2]:+.2f}]")
    print("    이 새 좌표에서는 XOR 이 직선 하나로 갈라집니다. 이것이 표현 학습입니다.")


if __name__ == "__main__":
    main()
