"""
A hands-on lesson that makes loss functions and gradient descent visible.
On the one-variable function f(x) = (x-3)^2 + 0.7*sin(3x),
we run gradient descent with 3 learning rates (too small / good / too big)
and compare the step-by-step paths in numbers and a figure (PNG).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def f(x):
    # Plays the role of the loss: a smooth valley plus gentle ripples (minimum near x≈3)
    return (x - 3.0) ** 2 + 0.7 * np.sin(3.0 * x)


def grad_f(x):
    # The derivative of f (analytic differentiation): the slope under your feet
    return 2.0 * (x - 3.0) + 2.1 * np.cos(3.0 * x)


def gradient_descent(x0, lr, n_steps):
    """Starting from x0, repeatedly move 'lr in the direction opposite the slope'."""
    xs = [x0]
    x = x0
    for _ in range(n_steps):
        x = x - lr * grad_f(x)
        xs.append(x)
        if abs(x) > 1e6:          # Stop on divergence (for the large-learning-rate demo)
            break
    return np.array(xs)


def numerical_grad(x, h=1e-5):
    # Central-difference numerical differentiation: cross-checks the analytic derivative
    return (f(x + h) - f(x - h)) / (2 * h)


def main():
    np.random.seed(42)                       # Reproducibility (this exercise uses almost no randomness)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1] Think of the loss function f(x) = (x-3)^2 + 0.7*sin(3x) as a 'mountain in fog'.")
    print("    Seeing only the slope (gradient) at your current position, you descend toward the valley (minimum).\n")

    x_check = 7.0
    print("[2] Gradient check — analytic derivative vs numerical differentiation (x=7.0)")
    print(f"    analytic: {grad_f(x_check):+.6f} / numerical: {numerical_grad(x_check):+.6f}")
    print("    The two agree, so the 'slope meter under your feet' can be trusted.\n")

    x0, n_steps = 9.0, 30
    settings = [("too_small", 0.01), ("good", 0.15), ("too_big", 1.05)]

    print(f"[3] Start x0={x0}, {n_steps} steps of gradient descent — comparing 3 learning rates")
    trajs = {}
    for name, lr in settings:
        xs = gradient_descent(x0, lr, n_steps)
        trajs[name] = (lr, xs)
        marks = [0, 1, 2, 5, 10, len(xs) - 1]
        print(f"\n    learning rate {lr} ({name})")
        for k in marks:
            if k < len(xs):
                print(f"      step {k:2d}: x = {xs[k]:+10.4f}, f(x) = {f(xs[k]):12.4f}")

    print("\n[4] Interpretation")
    print("    - 0.01 : right direction, but the steps are so short that 30 of them never reach the valley.")
    print("    - 0.15 : settles near the minimum (x≈3.5) within a few steps. A good learning rate.")
    print("    - 1.05 : leaps over the valley every time and |x| grows → divergence (loss explosion).")

    # Figure: the three trajectories side by side on the function curve
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    grid = np.linspace(-2, 11, 400)
    titles = {"too_small": "lr=0.01 (too small)",
              "good": "lr=0.15 (good)",
              "too_big": "lr=1.05 (too big → diverge)"}
    for ax, (name, (lr, xs)) in zip(axes, trajs.items()):
        ax.plot(grid, f(grid), color="#888888", lw=1.5)
        xs_plot = xs[np.abs(xs) < 12]        # only points within the plot range
        ax.plot(xs_plot, f(xs_plot), "o-", color="#d62728", ms=4, lw=1)
        ax.plot(xs_plot[0], f(xs_plot[0]), "s", color="#1f77b4", ms=8, label="start")
        ax.set_title(titles[name])
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()
    fig.suptitle("Gradient descent trajectories with 3 learning rates")
    fig.tight_layout()
    png_path = os.path.join(OUT_DIR, "gd_learning_rates.png")
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"\n[5] Trajectory figure saved: {png_path}")

    print("\n[6] Recap: the learning rate is your 'stride'. Too small and it takes forever; too big and it's a cliff.")
    print("    When deep-learning training fails, the learning rate is the first dial to suspect.")


if __name__ == "__main__":
    main()
