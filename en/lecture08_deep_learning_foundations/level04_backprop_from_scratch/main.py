"""
Implements backpropagation from scratch in pure numpy.
We hand-write the forward and backward passes of a 2-layer network
(2-8-1, tanh + sigmoid), train it until it classifies the 4 XOR points —
which a linear model could never solve — at 100%, and verify the analytic
gradients with numerical differentiation (central difference).
"""

import numpy as np

rng = np.random.default_rng(0)   # Reproducibility: fixed seed


# ----------------------------- Network parts -----------------------------

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def init_params():
    """Initialize the 2-8-1 network's weights with small random numbers."""
    return {
        "W1": rng.normal(0, 0.8, size=(2, 8)),
        "b1": np.zeros(8),
        "W2": rng.normal(0, 0.8, size=(8, 1)),
        "b2": np.zeros(1),
    }


def forward(params, X):
    """Forward pass: input -> hidden (tanh) -> output (sigmoid). Intermediates kept for backprop."""
    z1 = X @ params["W1"] + params["b1"]      # (n, 8)
    h = np.tanh(z1)                           # hidden representation
    z2 = h @ params["W2"] + params["b2"]      # (n, 1)
    p = sigmoid(z2)                           # 0..1, like a churn probability
    cache = {"X": X, "z1": z1, "h": h, "p": p}
    return p, cache


def bce_loss(p, y):
    """Binary cross-entropy: big penalties for confident wrong answers."""
    eps = 1e-9
    return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


def backward(params, cache, y):
    """Backward pass: distribute the output's error responsibility backwards via the chain rule."""
    X, h, p = cache["X"], cache["h"], cache["p"]
    n = X.shape[0]

    # Output layer: the derivative of the BCE+sigmoid combo simplifies neatly to (p - y)
    dz2 = (p - y) / n                          # (n, 1)
    grads = {
        "W2": h.T @ dz2,                       # responsibility in proportion to each hidden output's contribution
        "b2": dz2.sum(axis=0),
    }
    # Hidden layer: responsibility handed down from the output layer × the shrink factor through tanh
    dh = dz2 @ params["W2"].T                  # (n, 8)
    dz1 = dh * (1.0 - h ** 2)                  # tanh'(z) = 1 - tanh(z)^2
    grads["W1"] = X.T @ dz1
    grads["b1"] = dz1.sum(axis=0)
    return grads


def numerical_grads(params, X, y, key, h_eps=1e-6):
    """Numerical-diff check: nudge a parameter a hair and measure the loss change directly."""
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


# ------------------------------- The exercise -------------------------------

def main():
    # XOR: the problem a linear model (level00) could never solve
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    params = init_params()

    print("[1] Problem: XOR — (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0")
    print("    Training a 2-8-1 network (tanh + sigmoid) with a hand-made forward+backward pass.\n")

    print("[2] Gradient verification — backprop (analytic) vs numerical differentiation (measured)")
    p, cache = forward(params, X)
    ana = backward(params, cache, y)
    worst = 0.0
    for key in ["W1", "b1", "W2", "b2"]:
        num = numerical_grads(params, X, y, key)
        # relative error: |analytic - measured| / (|analytic| + |measured|)
        denom = np.abs(ana[key]) + np.abs(num) + 1e-12
        rel = float(np.max(np.abs(ana[key] - num) / denom))
        worst = max(worst, rel)
        print(f"    {key}: max relative error = {rel:.2e}")
    assert worst < 1e-4, "There is a bug in the backprop implementation!"
    print("    => All under 1e-4. The hand-made backprop is correct.\n")

    print("[3] Training starts (gradient descent, learning rate 0.5)")
    lr, n_epochs = 0.5, 3000
    for epoch in range(1, n_epochs + 1):
        p, cache = forward(params, X)
        grads = backward(params, cache, y)
        for key in params:                       # every weight moves against its gradient
            params[key] -= lr * grads[key]
        if epoch in (1, 10, 100, 500, 1000, 2000, 3000):
            acc = float(np.mean((p > 0.5) == y))
            print(f"    epoch {epoch:4d}: loss = {bce_loss(p, y):.4f}, accuracy = {acc:.0%}")

    print("\n[4] Training results — predicted probabilities for the 4 inputs")
    p, _ = forward(params, X)
    for xi, yi, pi in zip(X, y, p):
        mark = "O" if (pi[0] > 0.5) == bool(yi[0]) else "X"
        print(f"    input {xi} -> predicted p = {pi[0]:.4f} (truth {int(yi[0])}) {mark}")
    acc = float(np.mean((p > 0.5) == y))
    assert acc == 1.0, "XOR failed to converge"
    print(f"    final accuracy: {acc:.0%} — the linear model's 75% wall has been crossed.\n")

    print("[5] The 'new coordinates' (representation) built by the hidden layer — part of each input's hidden values")
    _, cache = forward(params, X)
    for xi, hi in zip(X, cache["h"]):
        print(f"    {xi} -> h[:3] = [{hi[0]:+.2f}, {hi[1]:+.2f}, {hi[2]:+.2f}]")
    print("    In these new coordinates, a single line splits XOR. This is representation learning.")


if __name__ == "__main__":
    main()
