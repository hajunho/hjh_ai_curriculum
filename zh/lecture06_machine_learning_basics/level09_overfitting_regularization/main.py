"""
level09 — 过拟合、正则化与超参数

用合成曲线数据(真实规律 + 噪声, 训练 30 个点)
  [2] 多项式次数 1~15 的验证曲线: 训练误差一路↓, 验证误差呈 U 形
  [3] 欠拟合/恰好/过拟合曲线 + 验证曲线保存为 PNG
  [4] Ridge(L2) 正则化: 用 alpha(罚金)安抚失控的 15 次模型
  [5] Lasso(L1): 把系数压成精确 0 的自动特征筛选
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
RNG = np.random.default_rng(42)          # seed 固定: 实验永远相同


def true_pattern(x: np.ndarray) -> np.ndarray:
    """大自然的真实规律 (模型不知道它，只看得到点)。"""
    return np.sin(1.5 * np.pi * x) + 0.5 * x


def make_dataset(n: int):
    """在 x 0~1 区间取 n 个样本。y = 真实规律 + 噪声(当天的偶然)。"""
    x = np.sort(RNG.uniform(0, 1, n))
    y = true_pattern(x) + RNG.normal(0, 0.25, n)
    return x.reshape(-1, 1), y


def poly_model(degree: int, alpha: float | None = None, l1: bool = False) -> Pipeline:
    """多项式特征 + 标准化 + (正则化)回归。标准化是罚金公平性的必需品。"""
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
    # [1] 数据: 训练 30 点(故意少), 验证 200 点 --------------------------
    X_tr, y_tr = make_dataset(30)
    X_va, y_va = make_dataset(200)
    print("[1] 数据: 真实规律是曲线, 观测带噪声 — 训练 30 点 / 验证 200 点")
    print("    (样本越少, 背噪声的诱惑即过拟合越明显)\n")

    # [2] 验证曲线: 升次数观察 train/valid 误差 -----------------------
    degrees = range(1, 16)
    tr_errs, va_errs = [], []
    print("[2] 各多项式次数的验证曲线 (误差 = RMSE, 越小越好)")
    print("    次数   训练误差   验证误差    最大系数大小")
    for d in degrees:
        m = poly_model(d).fit(X_tr, y_tr)
        tr_errs.append(rmse(m, X_tr, y_tr))
        va_errs.append(rmse(m, X_va, y_va))
        marker = "  <- 验证最低刷新" if va_errs[-1] == min(va_errs) else ""
        print(f"    {d:>3}    {tr_errs[-1]:7.3f}    {va_errs[-1]:7.3f}    {coef_size(m):12,.1f}{marker}")
    best_d = int(np.argmin(va_errs)) + 1
    print(f"    -> 训练误差一路降(背题余力↑)，验证误差在 {best_d} 次附近最低。")
    print("       两条线分岔之后的复杂度，全花在背噪声上。\n")

    # [3] 保存图片: 3 个次数的曲线 + 验证曲线 -------------------------------
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
    print(f"[3] 图片已保存: {png}\n")

    # [4] Ridge(L2): 给 15 次模型上罚金来安抚 --------------------------
    print("[4] Ridge 正则化 — 15 次模型 + 系数大小罚金(alpha)")
    print("    alpha      训练误差   验证误差    最大系数大小")
    for alpha in [0.0, 0.001, 0.1, 10.0]:
        m = poly_model(15, alpha=alpha if alpha > 0 else None).fit(X_tr, y_tr)
        print(f"    {alpha:<8}   {rmse(m, X_tr, y_tr):7.3f}    {rmse(m, X_va, y_va):7.3f}    {coef_size(m):12,.1f}")
    print("    -> 罚金一上，系数飙升立刻止住，验证误差恢复。")
    print("       '复杂模型 + 适度正则化'能与恰好次数的模型匹敌。")
    print("       alpha 也是超参数: 用验证成绩挑 (禁止偷看测试集)。\n")

    # [5] Lasso(L1): 把系数剪成 0 的自动特征筛选 ---------------------
    m_l1 = poly_model(15, alpha=0.01, l1=True).fit(X_tr, y_tr)
    coefs = m_l1.named_steps["reg"].coef_
    n_zero = int(np.sum(np.abs(coefs) < 1e-8))
    kept = [i for i in range(len(coefs)) if abs(coefs[i]) >= 1e-8]
    print("[5] Lasso(L1) — 绝对值罚金能把系数压成'精确的 0'")
    print(f"    15 次特征 {len(coefs)} 个中有 {n_zero} 个系数为 0 (自动淘汰)")
    print(f"    幸存的次数项: {kept}")
    print(f"    验证误差: {rmse(m_l1, X_va, y_va):.3f}")
    print("    -> 特征有几百个时，'只筛出真信号'的用途就交给 L1。")
