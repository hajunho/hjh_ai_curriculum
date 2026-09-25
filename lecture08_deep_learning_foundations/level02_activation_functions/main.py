"""
Lecture 08 · Level 02 — 활성화 함수
sigmoid/tanh/ReLU/Leaky ReLU 의 모양과 기울기(도함수)를 PNG 로 저장해 비교하고,
포화 구간의 기울기 소실과 죽은 ReLU 를 숫자로 확인합니다.
핵심 증명: 선형 층 3개를 합성하면 (W3 @ W2 @ W1) 하나와 완전히 같지만,
층 사이에 ReLU 를 끼우면 어떤 단일 선형 층으로도 흉내 낼 수 없음을 보입니다.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 화면 없이 파일로만 그림을 저장하는 백엔드
import matplotlib.pyplot as plt

np.random.seed(42)  # 재현성을 위한 시드 고정

# 이 파일이 있는 폴더 아래 outputs/ 에 그림을 저장
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# 함수별 색: 색약 친화 팔레트(Okabe-Ito)에서 고정 배정 (실행마다/그림마다 동일)
COLORS = {"sigmoid": "#0072B2", "tanh": "#E69F00",
          "ReLU": "#009E73", "Leaky ReLU": "#CC79A7"}


# ---- 활성화 함수와 해석적 도함수 --------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def d_sigmoid(z):
    s = sigmoid(z)
    return s * (1.0 - s)          # 최대값 0.25 (z=0)


def tanh(z):
    return np.tanh(z)


def d_tanh(z):
    return 1.0 - np.tanh(z) ** 2  # 최대값 1.0 (z=0), 양끝에서 0 으로 포화


def relu(z):
    return np.maximum(0.0, z)


def d_relu(z):
    return (z > 0).astype(float)  # 양수면 1, 음수면 0 (죽은 ReLU 의 원인)


def leaky_relu(z, slope=0.01):
    return np.where(z > 0, z, slope * z)


def d_leaky_relu(z, slope=0.01):
    return np.where(z > 0, 1.0, slope)  # 음수 구간에도 작은 기울기를 남김


FUNCS = [("sigmoid", sigmoid, d_sigmoid),
         ("tanh", tanh, d_tanh),
         ("ReLU", relu, d_relu),
         ("Leaky ReLU", leaky_relu, d_leaky_relu)]


def plot_activations(x):
    """[1] 함수 모양(실선)과 도함수(점선)를 2x2 패널로 저장합니다."""
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
    """[2] 네 함수의 도함수만 한 축에 겹쳐 그려 기울기 소실을 눈으로 비교합니다."""
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for name, _f, df in FUNCS:
        # ReLU 와 Leaky ReLU 는 양수 구간에서 겹치므로 Leaky ReLU 만 점선으로 구분
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
    """[4] 선형 층 3개의 합성 = 단일 행렬 W3@W2@W1 임을 수치로 증명합니다."""
    W1 = rng.normal(size=(5, 3))   # 3차원 입력 -> 5차원
    W2 = rng.normal(size=(4, 5))   # 5차원 -> 4차원
    W3 = rng.normal(size=(2, 4))   # 4차원 -> 2차원 출력
    x = rng.normal(size=(3, 8))    # 입력 8개 묶음(각 열이 샘플 하나)
    deep = W3 @ (W2 @ (W1 @ x))    # 층을 하나씩 통과
    W_single = W3 @ W2 @ W1        # 미리 곱해 둔 '한 장짜리' 행렬
    single = W_single @ x
    return W1, W2, W3, x, float(np.max(np.abs(deep - single)))


def relu_breaks_linearity(W1, W2, W3, x):
    """[5] 층 사이에 ReLU 를 끼우면 단일 선형 층과 달라지고, 가법성도 깨짐을 보입니다."""
    deep_relu = W3 @ relu(W2 @ relu(W1 @ x))
    single = (W3 @ W2 @ W1) @ x
    diff = float(np.max(np.abs(deep_relu - single)))

    # 선형 함수라면 반드시 f(a+b) = f(a) + f(b) 여야 합니다 (가법성).
    def net(v):
        return W3 @ relu(W2 @ relu(W1 @ v))

    a, b = x[:, :1], x[:, 1:2]
    additivity_gap = float(np.max(np.abs(net(a + b) - (net(a) + net(b)))))
    return diff, additivity_gap


def main():
    x = np.linspace(-6.0, 6.0, 601)

    path1 = plot_activations(x)
    print("[1] 활성화 함수 4종의 모양과 도함수를 그렸습니다")
    print(f"    저장 경로: {path1}")

    path2 = plot_gradients(x)
    print("[2] 도함수만 겹쳐 그린 비교 그림을 저장했습니다")
    print(f"    저장 경로: {path2}")

    print()
    print("[3] 포화와 기울기 소실, 죽은 ReLU 를 숫자로 확인")
    print(f"    sigmoid'(0) = {d_sigmoid(0.0):.4f} (최대), sigmoid'(5) = {d_sigmoid(5.0):.6f}")
    print(f"    -> z 가 5 만 되어도 기울기가 {d_sigmoid(0.0) / d_sigmoid(5.0):.0f}분의 1 수준으로 줄어듭니다 (포화)")
    print(f"    sigmoid 최대 기울기 0.25 를 10층 곱하면: 0.25**10 = {0.25 ** 10:.2e}")
    print("    -> 아래층에 도착하는 신호가 백만분의 1 수준, 이것이 기울기 소실입니다")
    print(f"    ReLU'(-3) = {d_relu(np.array(-3.0)):.2f} -> 음수 구간에 갇힌 뉴런은 기울기 0, 다시는 안 깨어남 (죽은 ReLU)")
    print(f"    Leaky ReLU'(-3) = {d_leaky_relu(np.array(-3.0)):.2f} -> 작은 기울기를 남겨 되살아날 여지를 둡니다")

    print()
    print("[4] 증명: 활성화 없는 선형 층 3개 = 선형 층 1개")
    rng = np.random.default_rng(42)  # 시드 고정한 난수 행렬
    W1, W2, W3, xs, max_diff = linear_stack_vs_single(rng)
    print("    난수 행렬 W1(5x3), W2(4x5), W3(2x4) 와 입력 8개로 비교합니다.")
    print(f"    max | W3@(W2@(W1@x)) - (W3@W2@W1)@x | = {max_diff:.2e}")
    print("    -> 부동소수점 한계 수준의 0. 세 층이 한 장짜리 행렬로 완전히 압축됩니다.")

    print()
    print("[5] 층 사이에 ReLU 를 끼우면?")
    diff, gap = relu_breaks_linearity(W1, W2, W3, xs)
    print(f"    max | W3@relu(W2@relu(W1@x)) - (W3@W2@W1)@x | = {diff:.3f}")
    print(f"    선형이라면 0 이어야 할 가법성 오차 max |f(a+b) - f(a) - f(b)| = {gap:.3f}")
    print("    -> 더 이상 어떤 단일 선형 층으로도 흉내 낼 수 없습니다.")
    print("    결론: 깊이를 진짜 깊이로 만들어 주는 부품이 활성화 함수입니다.")


if __name__ == "__main__":
    main()
