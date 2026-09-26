"""
Lecture 08 · Level 00 — Why Neural Networks Emerged
Proves, via least squares and a brute-force grid search, that a linear model
solves AND/OR perfectly but can never exceed 75% accuracy on XOR.
Finally it shows that inserting a hand-made coordinate transformation
(a hidden representation) lets the same linear model solve XOR at 100%,
connecting to the idea of representation learning.
"""

import numpy as np

np.random.seed(42)  # Fix the RNG seed for reproducibility (this level barely uses randomness, but we fix it as a rule)

# Inputs: the four combinations of (x1, x2)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)

# Answer table: three logic rules
TARGETS = {
    "AND": np.array([0, 0, 0, 1]),
    "OR":  np.array([0, 1, 1, 1]),
    "XOR": np.array([0, 1, 1, 0]),
}


def fit_least_squares(features, y):
    """Append a bias column, then solve for the linear model's optimal coefficients by least squares."""
    A = np.hstack([features, np.ones((len(features), 1))])  # last column of 1s = the bias slot
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef  # (w1, w2, ..., b)


def linear_accuracy(features, y, coef):
    """Return the accuracy and predictions when scores >= 0.5 are classified as 1."""
    A = np.hstack([features, np.ones((len(features), 1))])
    pred = (A @ coef >= 0.5).astype(int)
    return float((pred == y).mean()), pred


def best_linear_accuracy(features, y, n_w=61, n_b=61, limit=3.0):
    """Brute-force a grid of weights and biases to find the best accuracy any linear decision boundary can achieve.

    score = w1*x1 + w2*x2 + b; classify as 1 when score >= 0.
    numpy broadcasting evaluates all (n_w * n_w * n_b) combinations at once.
    """
    w_grid = np.linspace(-limit, limit, n_w)
    b_grid = np.linspace(-limit, limit, n_b)
    W1, W2, B = np.meshgrid(w_grid, w_grid, b_grid, indexing="ij")
    # The last axis of scores holds the 4 data points: shape = (n_w, n_w, n_b, 4)
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
    """Build two hand-designed hidden representations.

    h1 behaves like OR (1 when the sum is >= 0.5), h2 like AND (1 when the sum is >= 1.5).
    This is exactly what the two hidden-layer neurons you'll learn about later do.
    """
    s = features[:, 0] + features[:, 1]
    h1 = (s >= 0.5).astype(float)  # plays the role of OR
    h2 = (s >= 1.5).astype(float)  # plays the role of AND
    return np.stack([h1, h2], axis=1)


def main():
    print("[1] Truth-table data — the four (x1, x2) input combinations and the answers for three rules")
    print("    x1 x2 | AND OR XOR")
    for i in range(len(X)):
        print(f"     {int(X[i, 0])}  {int(X[i, 1])} |  {TARGETS['AND'][i]}   {TARGETS['OR'][i]}   {TARGETS['XOR'][i]}")

    print()
    print("[2] Fitting the 'best possible' linear model for each problem by least squares")
    for name, y in TARGETS.items():
        coef = fit_least_squares(X, y)
        acc, pred = linear_accuracy(X, y, coef)
        print(f"    {name:>3}: w1={coef[0]:+.3f}, w2={coef[1]:+.3f}, b={coef[2]:+.3f}"
              f" -> accuracy {acc * 100:5.1f}%  (pred {pred.tolist()}, truth {y.tolist()})")
    print("    => AND/OR reach 100%; a single line cannot get all four XOR points right.")

    print()
    print("[3] 'Maybe a better line is hiding somewhere?' — brute-force grid search over weights and biases")
    for name, y in TARGETS.items():
        best, n_tried = best_linear_accuracy(X, y)
        print(f"    {name:>3}: tried all {n_tried:,} (w1, w2, b) combinations -> best accuracy {best * 100:5.1f}%")
    print("    => For XOR, 75% is the ceiling no matter which line you pick. Not lack of effort — a structural limit.")

    print()
    print("[4] What if we insert a coordinate transformation (hidden representation) by hand?")
    H = hidden_features(X)
    print("    Move the data to new coordinates h1=OR(x1,x2), h2=AND(x1,x2).")
    print("    x1 x2 -> h1 h2 | XOR truth")
    for i in range(len(X)):
        print(f"     {int(X[i, 0])}  {int(X[i, 1])} ->  {int(H[i, 0])}  {int(H[i, 1])} |    {TARGETS['XOR'][i]}")
    coef_h = fit_least_squares(H, TARGETS["XOR"])
    acc_h, pred_h = linear_accuracy(H, TARGETS["XOR"], coef_h)
    best_h, _ = best_linear_accuracy(H, TARGETS["XOR"])
    print(f"    Linear model on the hidden representation: w1={coef_h[0]:+.3f}, w2={coef_h[1]:+.3f}, b={coef_h[2]:+.3f}"
          f" -> accuracy {acc_h * 100:.1f}%")
    print(f"    (Brute force confirms best accuracy {best_h * 100:.1f}% — effectively the one-liner y = h1 - h2)")

    print()
    print("[5] Conclusion")
    print("    - Same linear model, but changing the coordinates (representation) took XOR accuracy from 75% -> 100%.")
    print("    - So the model was never weak; the representation we viewed the data in was bad.")
    print("    - Letting the machine, not a human, learn this coordinate transformation from data is")
    print("      representation learning — the starting point of neural networks and deep learning.")


if __name__ == "__main__":
    main()
