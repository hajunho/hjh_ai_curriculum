"""
Lecture 08 · Level 02 — Activation Functions
Saves PNG comparisons of the shapes and slopes (derivatives) of
sigmoid/tanh/ReLU/Leaky ReLU, and puts numbers on vanishing gradients
in the saturated regions and on dead ReLUs.
Key proof: composing 3 linear layers is exactly one matrix (W3 @ W2 @ W1),
but inserting ReLU between layers cannot be imitated by any single linear layer.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend that saves figures to files only, no display window
import matplotlib.pyplot as plt

np.random.seed(42)  # Fixed seed for reproducibility

# Save figures under outputs/ next to this file
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# One fixed color per function from the colorblind-friendly Okabe-Ito palette (same across runs and figures)
COLORS = {"sigmoid": "#0072B2", "tanh": "#E69F00",
          "ReLU": "#009E73", "Leaky ReLU": "#CC79A7"}


# ---- Activation functions and their analytic derivatives -------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def d_sigmoid(z):
    s = sigmoid(z)
    return s * (1.0 - s)          # maximum 0.25 (at z=0)


def tanh(z):
    return np.tanh(z)


def d_tanh(z):
    return 1.0 - np.tanh(z) ** 2  # maximum 1.0 (at z=0), saturates to 0 at both ends


def relu(z):
    return np.maximum(0.0, z)


def d_relu(z):
    return (z > 0).astype(float)  # 1 for positive, 0 for negative (the cause of dead ReLUs)


def leaky_relu(z, slope=0.01):
    return np.where(z > 0, z, slope * z)


def d_leaky_relu(z, slope=0.01):
    return np.where(z > 0, 1.0, slope)  # keeps a small slope on the negative side too


FUNCS = [("sigmoid", sigmoid, d_sigmoid),
         ("tanh", tanh, d_tanh),
         ("ReLU", relu, d_relu),
         ("Leaky ReLU", leaky_relu, d_leaky_relu)]


def plot_activations(x):
    """[1] Save a 2x2 panel of function shapes (solid) and derivatives (dashed)."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), constrained_layout=True)
    for ax, (name, f, df) in zip(axes.ravel(), FUNCS):
        c = COLORS[name]
        ax.plot(x, f(x), color=c, linewidth=2, label="f(z)")
        ax.plot(x, df(x), color=c, linewidth=2, linestyle="--", alpha=0.55, label="f'(z)")
        ax.set_title(name)
        ax.axhline(0, color="#999999", linewidth=0.6)
        ax.axvline(0, color="#999999", linewidth=0.6)
        ax.grid(True, color="#dddddd", linewidth=0.5)
        ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle("Activation functions f(z) and derivatives f'(z)")
    path = os.path.join(OUT_DIR, "activations.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_gradients(x):
    """[2] Overlay just the four derivatives on one axis so the vanishing gradient is visible."""
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for name, _f, df in FUNCS:
        # ReLU and Leaky ReLU overlap on the positive side, so only Leaky ReLU is dashed
        style = "--" if name == "Leaky ReLU" else "-"
        width = 3.0 if name == "ReLU" else 2.0
        ax.plot(x, df(x), color=COLORS[name], linewidth=width, linestyle=style, label=name)
    ax.set_title("Derivatives compared: saturation vs. constant gradient")
    ax.set_xlabel("z")
    ax.set_ylabel("f'(z)")
    ax.grid(True, color="#dddddd", linewidth=0.5)
    ax.legend(loc="center right", frameon=False)
    path = os.path.join(OUT_DIR, "gradients.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def linear_stack_vs_single(rng):
    """[4] Prove numerically that composing 3 linear layers = the single matrix W3@W2@W1."""
    W1 = rng.normal(size=(5, 3))   # 3-dim input -> 5-dim
    W2 = rng.normal(size=(4, 5))   # 5-dim -> 4-dim
    W3 = rng.normal(size=(2, 4))   # 4-dim -> 2-dim output
    x = rng.normal(size=(3, 8))    # batch of 8 inputs (each column is one sample)
    deep = W3 @ (W2 @ (W1 @ x))    # pass through the layers one by one
    W_single = W3 @ W2 @ W1        # the premultiplied 'single sheet' matrix
    single = W_single @ x
    return W1, W2, W3, x, float(np.max(np.abs(deep - single)))


def relu_breaks_linearity(W1, W2, W3, x):
    """[5] Show that inserting ReLU between layers differs from any single linear layer, and breaks additivity."""
    deep_relu = W3 @ relu(W2 @ relu(W1 @ x))
    single = (W3 @ W2 @ W1) @ x
    diff = float(np.max(np.abs(deep_relu - single)))

    # A linear function must satisfy f(a+b) = f(a) + f(b) (additivity).
    def net(v):
        return W3 @ relu(W2 @ relu(W1 @ v))

    a, b = x[:, :1], x[:, 1:2]
    additivity_gap = float(np.max(np.abs(net(a + b) - (net(a) + net(b)))))
    return diff, additivity_gap


def main():
    x = np.linspace(-6.0, 6.0, 601)

    path1 = plot_activations(x)
    print("[1] Drew the shapes and derivatives of the 4 activation functions")
    print(f"    saved to: {path1}")

    path2 = plot_gradients(x)
    print("[2] Saved the comparison figure overlaying just the derivatives")
    print(f"    saved to: {path2}")

    print()
    print("[3] Saturation, vanishing gradients, and dead ReLUs in numbers")
    print(f"    sigmoid'(0) = {d_sigmoid(0.0):.4f} (max), sigmoid'(5) = {d_sigmoid(5.0):.6f}")
    print(f"    -> by z = 5 the gradient has shrunk to 1/{d_sigmoid(0.0) / d_sigmoid(5.0):.0f} of the max (saturation)")
    print(f"    sigmoid's max gradient 0.25 multiplied over 10 layers: 0.25**10 = {0.25 ** 10:.2e}")
    print("    -> the signal reaching the bottom layers is on the order of one millionth: the vanishing gradient")
    print(f"    ReLU'(-3) = {d_relu(np.array(-3.0)):.2f} -> a neuron stuck in the negative region has gradient 0 and never wakes up (dead ReLU)")
    print(f"    Leaky ReLU'(-3) = {d_leaky_relu(np.array(-3.0)):.2f} -> keeps a small gradient, leaving room for revival")

    print()
    print("[4] Proof: 3 linear layers without activation = 1 linear layer")
    rng = np.random.default_rng(42)  # random matrices with a fixed seed
    W1, W2, W3, xs, max_diff = linear_stack_vs_single(rng)
    print("    Comparing with random matrices W1(5x3), W2(4x5), W3(2x4) and 8 inputs.")
    print(f"    max | W3@(W2@(W1@x)) - (W3@W2@W1)@x | = {max_diff:.2e}")
    print("    -> Zero at floating-point precision. The three layers compress completely into one matrix.")

    print()
    print("[5] What if we insert ReLU between the layers?")
    diff, gap = relu_breaks_linearity(W1, W2, W3, xs)
    print(f"    max | W3@relu(W2@relu(W1@x)) - (W3@W2@W1)@x | = {diff:.3f}")
    print(f"    Additivity error, which would be 0 if linear: max |f(a+b) - f(a) - f(b)| = {gap:.3f}")
    print("    -> No single linear layer can imitate this anymore.")
    print("    Conclusion: the part that turns depth into real depth is the activation function.")


if __name__ == "__main__":
    main()
