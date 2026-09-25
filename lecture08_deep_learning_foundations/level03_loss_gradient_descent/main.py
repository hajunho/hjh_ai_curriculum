"""
손실 함수와 경사하강법을 눈으로 확인하는 실습입니다.
1변수 함수 f(x) = (x-3)^2 + 0.7*sin(3x) 위에서
학습률(learning rate) 3종(너무 작음/적당/너무 큼)으로 경사하강을 돌려
스텝별 이동 경로를 숫자와 그림(PNG)으로 비교합니다.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def f(x):
    # 손실 함수 역할: 매끈한 골짜기 + 완만한 굴곡 (최저점은 x≈3 부근)
    return (x - 3.0) ** 2 + 0.7 * np.sin(3.0 * x)


def grad_f(x):
    # f 의 도함수(해석적 미분): 발밑의 경사
    return 2.0 * (x - 3.0) + 2.1 * np.cos(3.0 * x)


def gradient_descent(x0, lr, n_steps):
    """x0 에서 시작해 '경사 반대 방향으로 lr 만큼' 이동을 반복합니다."""
    xs = [x0]
    x = x0
    for _ in range(n_steps):
        x = x - lr * grad_f(x)
        xs.append(x)
        if abs(x) > 1e6:          # 발산하면 중단 (큰 학습률 시연용)
            break
    return np.array(xs)


def numerical_grad(x, h=1e-5):
    # 중앙차분 수치미분: 해석적 미분이 맞는지 검산
    return (f(x + h) - f(x - h)) / (2 * h)


def main():
    np.random.seed(42)                       # 재현성 (이 실습은 난수 거의 없음)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1] 손실 함수 f(x) = (x-3)^2 + 0.7*sin(3x) 를 '안개 낀 산'이라 생각합니다.")
    print("    현재 위치의 경사(기울기)만 보고 골짜기(최저점)를 찾아 내려갑니다.\n")

    x_check = 7.0
    print("[2] 기울기 검산 — 해석적 미분 vs 수치미분 (x=7.0)")
    print(f"    해석적: {grad_f(x_check):+.6f} / 수치: {numerical_grad(x_check):+.6f}")
    print("    두 값이 같으므로 '발밑 경사 측정기'는 믿을 수 있습니다.\n")

    x0, n_steps = 9.0, 30
    settings = [("too_small", 0.01), ("good", 0.15), ("too_big", 1.05)]

    print(f"[3] 시작점 x0={x0}, {n_steps}스텝 경사하강 — 학습률 3종 비교")
    trajs = {}
    for name, lr in settings:
        xs = gradient_descent(x0, lr, n_steps)
        trajs[name] = (lr, xs)
        marks = [0, 1, 2, 5, 10, len(xs) - 1]
        print(f"\n    학습률 {lr} ({name})")
        for k in marks:
            if k < len(xs):
                print(f"      step {k:2d}: x = {xs[k]:+10.4f}, f(x) = {f(xs[k]):12.4f}")

    print("\n[4] 해석")
    print("    - 0.01 : 방향은 맞지만 걸음이 너무 짧아 30스텝으로는 골짜기에 못 도착.")
    print("    - 0.15 : 몇 스텝 만에 최저점 근처(x≈3.5)에 안착. 적당한 학습률.")
    print("    - 1.05 : 골짜기를 매번 건너뛰며 |x| 가 커짐 → 발산(loss 폭발).")

    # 그림: 함수 곡선 위에 세 궤적을 나란히
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    grid = np.linspace(-2, 11, 400)
    titles = {"too_small": "lr=0.01 (too small)",
              "good": "lr=0.15 (good)",
              "too_big": "lr=1.05 (too big → diverge)"}
    for ax, (name, (lr, xs)) in zip(axes, trajs.items()):
        ax.plot(grid, f(grid), color="#888888", lw=1.5)
        xs_plot = xs[np.abs(xs) < 12]        # 그림 범위 안의 점만
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
    print(f"\n[5] 궤적 그림 저장: {png_path}")

    print("\n[6] 정리: 학습률은 '보폭'입니다. 너무 작으면 하세월, 너무 크면 낭떠러지.")
    print("    딥러닝 학습이 안 될 때 가장 먼저 의심하는 다이얼이 학습률입니다.")


if __name__ == "__main__":
    main()
