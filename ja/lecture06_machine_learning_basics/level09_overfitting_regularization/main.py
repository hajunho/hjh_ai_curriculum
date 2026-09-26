"""
level09 — 過学習・正則化・ハイパーパラメータ

合成した曲線データ (本物のパターン + ノイズ, 訓練 30点) で
  [2] 多項式次数 1〜15 の検証曲線: 訓練誤差はずっと↓、検証誤差は U 字
  [3] 過小適合/適合/過学習の曲線 + 検証曲線を PNG に保存
  [4] Ridge (L2) 正則化: 暴走した 15 次モデルをアルファ (罰金) で鎮める
  [5] Lasso (L1): 係数をちょうど 0 にする自動特徴量選別
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
RNG = np.random.default_rng(42)          # seed 固定: 常に同じ実験


def true_pattern(x: np.ndarray) -> np.ndarray:
    """自然界の本物のパターン (モデルはこれを知らないまま点だけを見ます)。"""
    return np.sin(1.5 * np.pi * x) + 0.5 * x


def make_dataset(n: int):
    """x 0〜1 区間の n 個の標本。y = 本物のパターン + ノイズ (その日の偶然)。"""
    x = np.sort(RNG.uniform(0, 1, n))
    y = true_pattern(x) + RNG.normal(0, 0.25, n)
    return x.reshape(-1, 1), y


def poly_model(degree: int, alpha: float | None = None, l1: bool = False) -> Pipeline:
    """多項式特徴量 + 標準化 + (正則化つき) 回帰。標準化は罰金の公平性のため必須。"""
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
    # [1] データ: 訓練 30点 (わざと少なく), 検証 200点 --------------------------
    X_tr, y_tr = make_dataset(30)
    X_va, y_va = make_dataset(200)
    print("[1] データ: 本物のパターンは曲線、観測にはノイズ — 訓練 30点 / 検証 200点")
    print("    (標本が少ないほどノイズ暗記の誘惑、つまり過学習がよく見えます)\n")

    # [2] 検証曲線: 次数を上げながら train/valid 誤差を観察 -----------------------
    degrees = range(1, 16)
    tr_errs, va_errs = [], []
    print("[2] 多項式次数ごとの検証曲線 (誤差 = RMSE, 小さいほど良い)")
    print("    次数   訓練誤差    検証誤差    最大係数の大きさ")
    for d in degrees:
        m = poly_model(d).fit(X_tr, y_tr)
        tr_errs.append(rmse(m, X_tr, y_tr))
        va_errs.append(rmse(m, X_va, y_va))
        marker = "  <- 検証の最小を更新" if va_errs[-1] == min(va_errs) else ""
        print(f"    {d:>3}    {tr_errs[-1]:7.3f}    {va_errs[-1]:7.3f}    {coef_size(m):12,.1f}{marker}")
    best_d = int(np.argmin(va_errs)) + 1
    print(f"    -> 訓練誤差はずっと減りますが (暗記する余力↑)、検証誤差は {best_d} 次あたりが最小。")
    print("       2 本の線が分かれた後の複雑さは、すべてノイズの暗記に使われます。\n")

    # [3] 図の保存: 3 つの次数の曲線 + 検証曲線 -------------------------------
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
    print(f"[3] 図を保存: {png}\n")

    # [4] Ridge (L2): 15 次モデルに罰金をかけて鎮める --------------------------
    print("[4] Ridge 正則化 — 15 次モデル + 係数の大きさへの罰金 (アルファ)")
    print("    アルファ    訓練誤差    検証誤差    最大係数の大きさ")
    for alpha in [0.0, 0.001, 0.1, 10.0]:
        m = poly_model(15, alpha=alpha if alpha > 0 else None).fit(X_tr, y_tr)
        print(f"    {alpha:<8}   {rmse(m, X_tr, y_tr):7.3f}    {rmse(m, X_va, y_va):7.3f}    {coef_size(m):12,.1f}")
    print("    -> 罰金が付いた瞬間、係数の暴走が止まり、検証誤差が回復します。")
    print("       『複雑なモデル + 適切な正則化』は適正次数のモデルに匹敵します。")
    print("       アルファもハイパーパラメータ: 検証の成績で選びます (テストの盗み見は禁止)。\n")

    # [5] Lasso (L1): 係数を 0 に切り落とす自動特徴量選別 ---------------------
    m_l1 = poly_model(15, alpha=0.01, l1=True).fit(X_tr, y_tr)
    coefs = m_l1.named_steps["reg"].coef_
    n_zero = int(np.sum(np.abs(coefs) < 1e-8))
    kept = [i for i in range(len(coefs)) if abs(coefs[i]) >= 1e-8]
    print("[5] Lasso (L1) — 絶対値の罰金は係数を『ちょうど 0』にします")
    print(f"    15 次の特徴量 {len(coefs)}個のうち {n_zero}個の係数が 0 (自動脱落)")
    print(f"    生き残った次数の項: {kept}")
    print(f"    検証誤差: {rmse(m_l1, X_va, y_va):.3f}")
    print("    -> 特徴量が数百個あるとき『本物のシグナルだけを選び出す』用途に使うのが L1 です。")
