"""
level09 — Overfitting, regularization, and hyperparameters

On synthetic curve data (true pattern + noise, 30 training points):
  [2] validation curve over polynomial degrees 1-15: training error keeps
      falling, validation error makes a U
  [3] save underfit/good/overfit curves + the validation curve as a PNG
  [4] Ridge (L2): calming the runaway degree-15 model with alpha (the fine)
  [5] Lasso (L1): automatic feature selection that drives coefficients to exactly 0
"""

import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
RNG = np.random.default_rng(42)          # fixed seed: always the same experiment


def true_pattern(x: np.ndarray) -> np.ndarray:
    """Nature's true pattern (the model never sees this — only the dots)."""
    return np.sin(1.5 * np.pi * x) + 0.5 * x


def make_dataset(n: int):
    """n samples on x in [0, 1]. y = true pattern + noise (that day's accidents)."""
    x = np.sort(RNG.uniform(0, 1, n))
    y = true_pattern(x) + RNG.normal(0, 0.25, n)
    return x.reshape(-1, 1), y


def poly_model(degree: int, alpha: float | None = None, l1: bool = False) -> Pipeline:
    """Polynomial features + standardization + (regularized) regression.
    Standardization is mandatory so the regularization fine is levied fairly."""
    if alpha is None:
        reg = LinearRegression()
    elif l1:
        reg = Lasso(alpha=alpha, max_iter=50_000)
    else:
        reg = Ridge(alpha=alpha)
    return Pipeline([("poly", PolynomialFeatures(degree)),
                     ("scaler", StandardScaler()),
                     ("reg", reg)])


def rmse(model, X, y) -> float:
    return float(np.sqrt(mean_squared_error(y, model.predict(X))))


def coef_size(model) -> float:
    return float(np.abs(model.named_steps["reg"].coef_).max())


if __name__ == "__main__":
    # [1] Data: 30 training points (deliberately few), 200 validation points ------
    X_tr, y_tr = make_dataset(30)
    X_va, y_va = make_dataset(200)
    print("[1] Data: true pattern is a curve, observations are noisy — train 30 / validation 200")
    print("    (The fewer the samples, the stronger the temptation to memorize noise,")
    print("     i.e., the more visible the overfitting.)\n")

    # [2] Validation curve: raise the degree, watch train/valid error --------------
    degrees = range(1, 16)
    tr_errs, va_errs = [], []
    print("[2] Validation curve by polynomial degree (error = RMSE, lower is better)")
    print("    degree   train err   valid err   max coefficient size")
    for d in degrees:
        m = poly_model(d).fit(X_tr, y_tr)
        tr_errs.append(rmse(m, X_tr, y_tr))
        va_errs.append(rmse(m, X_va, y_va))
        marker = "  <- new validation low" if va_errs[-1] == min(va_errs) else ""
        print(f"    {d:>4}    {tr_errs[-1]:7.3f}    {va_errs[-1]:7.3f}    {coef_size(m):12,.1f}{marker}")
    best_d = int(np.argmin(va_errs)) + 1
    print(f"    -> Training error keeps falling (capacity to memorize grows) but")
    print(f"       validation error bottoms out around degree {best_d}.")
    print("       All complexity past the divergence goes into memorizing noise.\n")

    # [3] Save the figure: curves for 3 degrees + the validation curve --------------
    os.makedirs(OUT_DIR, exist_ok=True)
    grid = np.linspace(0, 1, 300).reshape(-1, 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(X_tr, y_tr, color="black", s=25, zorder=3, label="train points")
    axes[0].plot(grid, true_pattern(grid.ravel()), "k--", alpha=0.5, label="true pattern")
    for d, color in [(1, "tab:blue"), (best_d, "tab:green"), (15, "tab:red")]:
        m = poly_model(d).fit(X_tr, y_tr)
        axes[0].plot(grid, m.predict(grid), color=color,
                     label=f"degree {d} ({'under' if d == 1 else 'good' if d == best_d else 'OVERFIT'})")
    axes[0].set_ylim(-2.5, 2.5)
    axes[0].set_title("Underfit vs good fit vs overfit")
    axes[0].legend(fontsize=8)
    axes[1].plot(list(degrees), tr_errs, "o-", label="train RMSE")
    axes[1].plot(list(degrees), va_errs, "s-", label="valid RMSE")
    axes[1].axvline(best_d, color="gray", linestyle=":", label=f"best degree={best_d}")
    axes[1].set_xlabel("polynomial degree")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Validation curve: the two lines diverge")
    axes[1].legend()
    fig.tight_layout()
    png = OUT_DIR / "overfitting.png"
    fig.savefig(png, dpi=120)
    print(f"[3] Figure saved: {png}\n")

    # [4] Ridge (L2): fining the degree-15 model into composure ---------------------
    print("[4] Ridge regularization — degree-15 model + a coefficient-size fine (alpha)")
    print("    alpha      train err   valid err   max coefficient size")
    for alpha in [0.0, 0.001, 0.1, 10.0]:
        m = poly_model(15, alpha=alpha if alpha > 0 else None).fit(X_tr, y_tr)
        print(f"    {alpha:<8}   {rmse(m, X_tr, y_tr):7.3f}    {rmse(m, X_va, y_va):7.3f}    {coef_size(m):12,.1f}")
    print("    -> The moment the fine kicks in, the coefficient explosion stops and")
    print("       validation error recovers. 'Complex model + the right regularization'")
    print("       rivals a well-sized model.")
    print("       Alpha is a hyperparameter too: choose it by validation score (no test peeking).\n")

    # [5] Lasso (L1): automatic feature selection by cutting coefficients to 0 --------
    m_l1 = poly_model(15, alpha=0.01, l1=True).fit(X_tr, y_tr)
    coefs = m_l1.named_steps["reg"].coef_
    n_zero = int(np.sum(np.abs(coefs) < 1e-8))
    kept = [i for i in range(len(coefs)) if abs(coefs[i]) >= 1e-8]
    print("[5] Lasso (L1) — the absolute-value fine drives coefficients to 'exactly 0'")
    print(f"    Of {len(coefs)} degree-15 features, {n_zero} got a coefficient of 0 (auto-dropped)")
    print(f"    Surviving degree terms: {kept}")
    print(f"    Validation error: {rmse(m_l1, X_va, y_va):.3f}")
    print("    -> With hundreds of features, L1 is the tool for 'sifting out the real signals'.")
