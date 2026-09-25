"""
level09 — 과적합·규제·하이퍼파라미터

합성 곡선 데이터(진짜 패턴 + 잡음, 훈련 30점)로
  [2] 다항 차수 1~15 검증곡선: 훈련 오차는 계속↓, 검증 오차는 U자
  [3] 과소적합/적정/과적합 곡선 + 검증곡선을 PNG 로 저장
  [4] Ridge(L2) 규제: 폭주한 15차 모델을 알파(벌금)로 진정시키기
  [5] Lasso(L1): 계수를 정확히 0으로 만드는 자동 특징 선별
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
RNG = np.random.default_rng(42)          # seed 고정: 항상 같은 실험


def true_pattern(x: np.ndarray) -> np.ndarray:
    """자연의 진짜 패턴 (모델은 이것을 모른 채 점만 봅니다)."""
    return np.sin(1.5 * np.pi * x) + 0.5 * x


def make_dataset(n: int):
    """x 0~1 구간에서 n개 표본. y = 진짜 패턴 + 잡음(그날의 우연)."""
    x = np.sort(RNG.uniform(0, 1, n))
    y = true_pattern(x) + RNG.normal(0, 0.25, n)
    return x.reshape(-1, 1), y


def poly_model(degree: int, alpha: float | None = None, l1: bool = False) -> Pipeline:
    """다항 특징 + 표준화 + (규제) 회귀. 표준화는 규제 벌금의 공정성을 위해 필수."""
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
    # [1] 데이터: 훈련 30점(일부러 적게), 검증 200점 --------------------------
    X_tr, y_tr = make_dataset(30)
    X_va, y_va = make_dataset(200)
    print("[1] 데이터: 진짜 패턴은 곡선, 관측에는 잡음 — 훈련 30점 / 검증 200점")
    print("    (표본이 적을수록 잡음 암기의 유혹, 즉 과적합이 잘 보입니다)\n")

    # [2] 검증곡선: 차수를 올리며 train/valid 오차 관찰 -----------------------
    degrees = range(1, 16)
    tr_errs, va_errs = [], []
    print("[2] 다항 차수별 검증곡선 (오차 = RMSE, 작을수록 좋음)")
    print("    차수   훈련 오차   검증 오차   최대 계수 크기")
    for d in degrees:
        m = poly_model(d).fit(X_tr, y_tr)
        tr_errs.append(rmse(m, X_tr, y_tr))
        va_errs.append(rmse(m, X_va, y_va))
        marker = "  <- 검증 최저 갱신" if va_errs[-1] == min(va_errs) else ""
        print(f"    {d:>3}    {tr_errs[-1]:7.3f}    {va_errs[-1]:7.3f}    {coef_size(m):12,.1f}{marker}")
    best_d = int(np.argmin(va_errs)) + 1
    print(f"    -> 훈련 오차는 계속 줄지만(외울 여력↑) 검증 오차는 {best_d}차 부근이 최저.")
    print("       두 선이 갈라진 뒤의 복잡도는 전부 잡음 암기에 쓰입니다.\n")

    # [3] 그림 저장: 3개 차수의 곡선 + 검증곡선 -------------------------------
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
    print(f"[3] 그림 저장: {png}\n")

    # [4] Ridge(L2): 15차 모델에 벌금 걸어 진정시키기 --------------------------
    print("[4] Ridge 규제 — 15차 모델 + 계수 크기 벌금(알파)")
    print("    알파       훈련 오차   검증 오차   최대 계수 크기")
    for alpha in [0.0, 0.001, 0.1, 10.0]:
        m = poly_model(15, alpha=alpha if alpha > 0 else None).fit(X_tr, y_tr)
        print(f"    {alpha:<8}   {rmse(m, X_tr, y_tr):7.3f}    {rmse(m, X_va, y_va):7.3f}    {coef_size(m):12,.1f}")
    print("    -> 벌금이 붙는 순간 계수 폭주가 멈추고 검증 오차가 회복됩니다.")
    print("       '복잡한 모델 + 적절한 규제'는 적정 차수 모델에 필적합니다.")
    print("       알파도 하이퍼파라미터: 검증 성적으로 고릅니다 (테스트 훔쳐보기 금지).\n")

    # [5] Lasso(L1): 계수를 0으로 잘라 내는 자동 특징 선별 ---------------------
    m_l1 = poly_model(15, alpha=0.01, l1=True).fit(X_tr, y_tr)
    coefs = m_l1.named_steps["reg"].coef_
    n_zero = int(np.sum(np.abs(coefs) < 1e-8))
    kept = [i for i in range(len(coefs)) if abs(coefs[i]) >= 1e-8]
    print("[5] Lasso(L1) — 절댓값 벌금은 계수를 '정확히 0'으로 만듭니다")
    print(f"    15차 특징 {len(coefs)}개 중 {n_zero}개의 계수가 0 (자동 탈락)")
    print(f"    살아남은 차수 항: {kept}")
    print(f"    검증 오차: {rmse(m_l1, X_va, y_va):.3f}")
    print("    -> 특징이 수백 개일 때 '진짜 신호만 추리는' 용도로 쓰는 것이 L1 입니다.")
