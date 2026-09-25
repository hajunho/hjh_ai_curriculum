"""
level08 — 이미지 분류 실전 파이프라인

데이터 검수 -> 분할 -> 학습 -> 오류 분석 -> 개선의 전체 루프를 돌립니다.
학습 라벨의 8% 를 일부러 오염시킨 뒤(라벨 노이즈),
'손실이 큰 학습 샘플 재검토'라는 오류 분석이 오염 라벨을 찾아내는 것을 시연합니다.
"""

import os
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
CLASS_EN = ["square", "circle", "triangle"]
NOISE_RATE = 0.08      # 학습 라벨의 8% 를 엉뚱한 클래스로 오염
EPOCHS = 18


def make_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(256, 3))


def train_model(xtr, ytr, seed: int = 8) -> nn.Module:
    """고정 설정 학습 (비교 실험용)."""
    torch.manual_seed(seed)
    model = make_cnn()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss_fn(model(xtr[idx]), ytr[idx]).backward()
            opt.step()
    return model


@torch.no_grad()
def accuracy(model, x, y) -> float:
    model.eval()
    return float((model(x).argmax(1) == y).float().mean())


@torch.no_grad()
def per_sample_loss(model, x, y) -> np.ndarray:
    """샘플별 교차 엔트로피 — '모델이 납득 못 하는 정도'."""
    model.eval()
    return nn.CrossEntropyLoss(reduction="none")(model(x), y).numpy()


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(8)
    np.random.seed(8)  # seed 고정(재현성)

    # ------------------------------------------------------------------
    print("[1] 데이터 검수 — 모델보다 데이터를 먼저 본다")
    X, y_true = hjh_data.shape_images(n=900, size=16, seed=13)
    print(f"    수량: {len(X)}장 / 크기: {X.shape[1]}x{X.shape[2]} / 값 범위: {X.min():.2f}~{X.max():.2f}")
    counts = np.bincount(y_true)
    print(f"    클래스 균형: " + ", ".join(f"{CLASS_EN[c]} {counts[c]}" for c in range(3)) + "  (균형 OK)")

    # 현실 재현: 라벨링 외주 과정에서 학습 라벨 일부가 잘못 붙었다고 하자
    y_noisy = y_true.copy()
    n_corrupt = int(600 * NOISE_RATE)
    corrupt_idx = rng.choice(600, size=n_corrupt, replace=False)
    for i in corrupt_idx:
        y_noisy[i] = (y_true[i] + int(rng.integers(1, 3))) % 3   # 반드시 다른 클래스로
    print(f"    (시나리오) 학습 라벨 600개 중 {n_corrupt}개({NOISE_RATE:.0%})가 잘못 붙어 있음 — 우리는 모른 척!")

    # ------------------------------------------------------------------
    print("\n[2] 분할 — 학습 600 / 검증 150 / 테스트 150 (검증·테스트 라벨은 깨끗)")
    to_t = lambda a: torch.from_numpy(a)
    xt = to_t(X).unsqueeze(1)
    xtr, ytr = xt[:600], to_t(y_noisy[:600])
    xva, yva = xt[600:750], to_t(y_true[600:750])
    xte, yte = xt[750:], to_t(y_true[750:])

    # ------------------------------------------------------------------
    print("\n[3] 1차 학습 — 오염된 라벨로 학습하면?")
    model_v1 = train_model(xtr, ytr)
    acc_v1 = accuracy(model_v1, xte, yte)
    print(f"    v1 테스트 정확도: {acc_v1:.3f}  (도형 분류치고 낮다… 왜?)")

    # ------------------------------------------------------------------
    print("\n[4] 오류 분석 — '모델이 끝까지 납득 못 한 학습 샘플'을 재검토")
    losses = per_sample_loss(model_v1, xtr, ytr)
    order = np.argsort(-losses)                        # 손실 큰 순
    top_k = n_corrupt + 12                             # 오염 수보다 조금 넉넉히 재검토
    flagged = order[:top_k]
    hit = np.isin(flagged, corrupt_idx).sum()
    print(f"    학습 손실 상위 {top_k}개를 사람이 재검토한다고 하자.")
    print(f"    -> 그중 실제 오염 라벨: {hit}개 / 전체 오염 {n_corrupt}개 중 {hit / n_corrupt:.0%} 검출!")
    print(f"    무작위로 {top_k}개를 뽑았다면 기대 검출은 {top_k * n_corrupt / 600:.1f}개.")
    print("    '손실 큰 샘플부터 본다'는 단순 규칙이 라벨 오류 탐지기로 작동합니다.")

    # ------------------------------------------------------------------
    print("\n[5] 개선 루프 — 재검토로 라벨을 고쳐 재학습")
    y_fixed = y_noisy.copy()
    y_fixed[flagged] = y_true[flagged]                 # 재검토한 표본은 올바른 라벨로 수정
    model_v2 = train_model(xtr, to_t(y_fixed[:600]))
    acc_v2 = accuracy(model_v2, xte, yte)
    model_oracle = train_model(xtr, to_t(y_true[:600]))   # 참고: 처음부터 깨끗했다면
    acc_oracle = accuracy(model_oracle, xte, yte)
    print(f"      {'버전':<26} {'테스트 정확도':>10}")
    print(f"      {'v1: 오염 라벨 그대로':<26} {acc_v1:>10.3f}")
    print(f"      {'v2: 상위손실 재검토·수정':<24} {acc_v2:>10.3f}")
    print(f"      {'참고: 완전 무오염(이상적)':<24} {acc_oracle:>10.3f}")
    print(f"    -> 모델 구조는 그대로, 데이터만 고쳐서 +{(acc_v2 - acc_v1) * 100:.1f}%p.")
    print("       실전에서 성능을 올리는 가장 싼 버튼은 종종 '데이터 청소'입니다.")

    # ------------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plt.figure(figsize=(11, 6))
    for k in range(8):                                  # 손실 상위 8개 = 재검토 대상
        i = int(order[k])
        ax = fig.add_subplot(2, 8, k + 1 + (8 if k >= 4 else 0) - (0 if k < 4 else 4))
        ax.imshow(X[i], cmap="gray", vmin=0, vmax=1)
        bad = i in corrupt_idx
        ax.set_title(f"label {CLASS_EN[int(y_noisy[i])]}\n{'WRONG!' if bad else 'hard'}",
                     fontsize=7, color="red" if bad else "black")
        ax.axis("off")
    ax = fig.add_subplot(1, 2, 2)
    names = ["v1 noisy labels", "v2 after review", "oracle clean"]
    vals = [acc_v1, acc_v2, acc_oracle]
    bars = ax.bar(names, vals, width=0.5, color=["#c66", "#69c", "#6a6"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0.6, 1.05)
    ax.set_ylabel("test accuracy")
    ax.set_title("Error analysis loop: fix data, not model")
    fig.suptitle("Left: highest-loss training samples for human review (red = truly mislabeled)",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "pipeline_error_analysis.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    저장 완료: {path}")

    print(f"\n[정리] 검수->분할->학습->오류분석->개선. 루프를 돈 횟수가 실력입니다 (소요 {time.time() - t0:.1f}초).")
    print("       다음 레벨: '무엇이 있나'를 넘어 '어디에 있나' — 객체 탐지의 원리.")


if __name__ == "__main__":
    main()
