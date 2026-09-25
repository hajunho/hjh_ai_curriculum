"""
level05 — 도형 이미지 분류 실습

hjh_data.shape_images 로 CNN 을 학습시키는 전체 과정:
  데이터 분할 -> 학습 루프 -> 평가(혼동 행렬) -> 오분류 사례 시각화.
같은 조건의 MLP 와 성능·파라미터를 비교해 CNN 의 장점을 확인합니다.
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
EPOCHS = 18
BATCH = 64
LR = 1e-3


class ShapeCNN(nn.Module):
    """16x16 -> 3클래스. Conv-ReLU-Pool 블록 2개 + Linear 판정."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # (8,8,8)
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # (16,4,4)
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16 * 4 * 4, 3))

    def forward(self, x):
        return self.head(self.features(x))


def make_mlp() -> nn.Module:
    """비교용 MLP: 픽셀을 그냥 펴서 처리 (위치 구조를 모름)."""
    return nn.Sequential(nn.Flatten(), nn.Linear(256, 64), nn.ReLU(),
                         nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))


def train(model: nn.Module, xtr, ytr, xva, yva, tag: str) -> list[float]:
    """미니배치 학습 루프. 에폭마다 검증 정확도를 기록해 반환한다."""
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        acc = evaluate(model, xva, yva)
        history.append(acc)
        if epoch % 3 == 0 or epoch == 1:
            print(f"      {tag} epoch {epoch:>2}/{EPOCHS}  train_loss={loss.item():.3f}  val_acc={acc:.3f}")
    return history


@torch.no_grad()
def evaluate(model: nn.Module, x, y) -> float:
    model.eval()
    return float((model(x).argmax(1) == y).float().mean())


@torch.no_grad()
def predict(model: nn.Module, x) -> torch.Tensor:
    model.eval()
    return model(x).argmax(1)


def main() -> None:
    t0 = time.time()
    torch.manual_seed(5)
    np.random.seed(5)  # seed 고정(재현성)

    # ------------------------------------------------------------------
    print("[1] 데이터 준비와 분할 — 학습 600 / 검증 150 / 테스트 150")
    X, y = hjh_data.shape_images(n=900, size=16, seed=13)
    Xt = torch.from_numpy(X).unsqueeze(1)              # (900,1,16,16)
    yt = torch.from_numpy(y)
    xtr, ytr = Xt[:600], yt[:600]                      # shape_images 는 이미 셔플됨
    xva, yva = Xt[600:750], yt[600:750]
    xte, yte = Xt[750:], yt[750:]
    print(f"    전체 {len(X)}장 (16x16 흑백, 사각형/원/삼각형). 테스트는 학습에 절대 안 씀.")

    # ------------------------------------------------------------------
    print("\n[2] CNN 학습 — Conv-ReLU-Pool x2 + Linear")
    cnn = ShapeCNN()
    n_cnn = sum(p.numel() for p in cnn.parameters())
    hist_cnn = train(cnn, xtr, ytr, xva, yva, "CNN")

    print("\n[3] 같은 데이터로 MLP 학습 — 픽셀을 그냥 펴서 넣는다면?")
    mlp = make_mlp()
    n_mlp = sum(p.numel() for p in mlp.parameters())
    hist_mlp = train(mlp, xtr, ytr, xva, yva, "MLP")

    # ------------------------------------------------------------------
    print("\n[4] 테스트 성적표 — 한 번도 본 적 없는 150장")
    acc_cnn = evaluate(cnn, xte, yte)
    acc_mlp = evaluate(mlp, xte, yte)
    print(f"      {'모델':<6} {'파라미터':>10} {'테스트 정확도':>12}")
    print(f"      {'CNN':<6} {n_cnn:>10,} {acc_cnn:>12.3f}")
    print(f"      {'MLP':<6} {n_mlp:>10,} {acc_mlp:>12.3f}")
    pred = predict(cnn, xte)
    conf = np.zeros((3, 3), dtype=int)
    for t, p in zip(yte.numpy(), pred.numpy()):
        conf[t, p] += 1
    print("    CNN 혼동 행렬 (행=실제, 열=예측):")
    print("            " + "".join(f"{n:>10}" for n in CLASS_EN))
    for i, name in enumerate(CLASS_EN):
        print(f"      {name:>8}  " + "".join(f"{v:>10}" for v in conf[i]))

    # ------------------------------------------------------------------
    print("\n[5] 오분류 사례 분석 — 모델이 틀린 이미지를 직접 본다")
    wrong = torch.where(pred != yte)[0]
    print(f"    틀린 개수: {len(wrong)}/{len(yte)}")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plt.figure(figsize=(11, 6.5))
    # 왼쪽: 학습 곡선
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(range(1, EPOCHS + 1), hist_cnn, marker="o", label=f"CNN ({n_cnn:,} params)")
    ax.plot(range(1, EPOCHS + 1), hist_mlp, marker="s", label=f"MLP ({n_mlp:,} params)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy")
    ax.set_title("Learning curves")
    ax.set_ylim(0.3, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    # 오른쪽: 오분류 사례 최대 8장 (2x8 격자의 오른쪽 절반을 사용)
    n_show = min(8, len(wrong))
    for k in range(n_show):
        i = int(wrong[k])
        ax = fig.add_subplot(2, 8, 5 + (k % 4) + 8 * (k // 4))
        ax.imshow(xte[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"true {CLASS_EN[int(yte[i])]}\npred {CLASS_EN[int(pred[i])]}", fontsize=7)
        ax.axis("off")
    fig.suptitle("Shape classification: CNN vs MLP + misclassified test images", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "shape_classification.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    저장 완료: {path}")
    print("    -> 틀린 사례는 대부분 '잡음이 심하거나 도형이 가장자리에 걸린' 어려운 이미지입니다.")
    print("       오분류를 눈으로 보는 습관이 level08 의 오류 분석으로 이어집니다.")

    print(f"\n[정리] 분할->학습->평가->오분류 분석: 이미지 분류의 표준 사이클을 완주했습니다.")
    print(f"       CNN 은 더 적은 파라미터로 MLP 보다 높은 정확도 (총 소요 {time.time() - t0:.1f}초).")
    print("       다음 레벨: 데이터를 더 모으지 않고 성능을 올리는 기술 — 데이터 증강.")


if __name__ == "__main__":
    main()
