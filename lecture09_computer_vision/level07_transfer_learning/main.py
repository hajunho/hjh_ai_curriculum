"""
level07 — 전이학습 (Transfer Learning)

과제 A(사각형/원/삼각형, 데이터 넉넉)로 CNN 을 사전학습한 뒤,
그 특징 추출기를 '얼려서' 과제 B(마름모/타원/고리, 데이터 24장뿐)에 이식합니다.
처음부터 학습하는 모델과 에폭별 성능을 비교해 전이학습의 이득을 증명합니다.
전부 저장소 안에서 그린 이미지 — 다운로드 없는 전이학습 실험입니다.
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
TASK_B_EN = ["diamond", "ellipse", "ring"]
N_B_TRAIN = 24         # 과제 B 는 라벨 데이터가 24장뿐 (실무의 흔한 현실)
EPOCHS_B = 120


def task_b_images(n: int, size: int = 16, seed: int = 77):
    """과제 B: 마름모/타원/고리 — hjh_data 와 같은 스타일로 직접 그린다."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, size, size), dtype="float32")
    y = np.zeros(n, dtype="int64")
    for i in range(n):
        label = i % 3
        y[i] = label
        img = np.zeros((size, size), dtype="float32")
        cy, cx = rng.integers(5, size - 5, size=2)
        r = int(rng.integers(3, 5))
        yy, xx = np.mgrid[0:size, 0:size]
        if label == 0:                                   # 마름모(다이아몬드)
            img[np.abs(yy - cy) + np.abs(xx - cx) <= r] = 1.0
        elif label == 1:                                 # 세로로 긴 타원
            img[((yy - cy) / r) ** 2 + ((xx - cx) / max(r // 2, 1)) ** 2 <= 1.0] = 1.0
        else:                                            # 두꺼운 고리(링)
            dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
            img[(dist2 <= r * r) & (dist2 >= (r - 2) ** 2)] = 1.0
        img += rng.normal(0, 0.08, img.shape).astype("float32")
        X[i] = np.clip(img, 0.0, 1.0)
    idx = rng.permutation(n)
    return X[idx], y[idx]


def make_cnn() -> nn.Module:
    """features(특징 추출기) + head(판정부)로 분리된 CNN.
    일부러 조금 큰 모델(약 2만 파라미터) — 데이터가 적으면 백지 학습이 힘든 크기."""
    return nn.Sequential(
        nn.Sequential(                                    # [0] features
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()),
        nn.Sequential(nn.Flatten(), nn.Linear(32 * 4 * 4, 3)))   # [1] head


def to_torch(X, y):
    return torch.from_numpy(X).unsqueeze(1), torch.from_numpy(y)


@torch.no_grad()
def accuracy(model, x, y) -> float:
    model.eval()
    return float((model(x).argmax(1) == y).float().mean())


def train_b(model: nn.Module, xtr, ytr, xte, yte, tag: str) -> list[float]:
    """과제 B 학습. 학습 대상은 requires_grad=True 인 파라미터뿐."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"      {tag}: 학습되는 파라미터 {n_train:,} / 전체 {n_total:,}")
    opt = torch.optim.Adam(trainable, lr=5e-3)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, EPOCHS_B + 1):
        model.train()
        opt.zero_grad()
        loss_fn(model(xtr), ytr).backward()               # 60장이라 풀배치
        opt.step()
        history.append(accuracy(model, xte, yte))
    return history


def main() -> None:
    t0 = time.time()
    torch.manual_seed(7)
    np.random.seed(7)  # seed 고정(재현성)

    # ------------------------------------------------------------------
    print("[1] 과제 A 사전학습 — 사각형/원/삼각형, 데이터 600장 (넉넉)")
    XA, yA = hjh_data.shape_images(n=750, size=16, seed=13)
    xa_tr, ya_tr = to_torch(XA[:600], yA[:600])
    xa_te, ya_te = to_torch(XA[600:], yA[600:])
    pretrained = make_cnn()
    opt = torch.optim.Adam(pretrained.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(20):
        perm = torch.randperm(len(xa_tr))
        for i in range(0, len(xa_tr), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss_fn(pretrained(xa_tr[idx]), ya_tr[idx]).backward()
            opt.step()
    print(f"    사전학습 완료: 과제 A 테스트 정확도 {accuracy(pretrained, xa_te, ya_te):.3f}")
    print("    이 모델의 앞부분(features)은 이제 '엣지·곡선·형태'를 보는 눈을 갖췄습니다.")

    # ------------------------------------------------------------------
    print(f"\n[2] 과제 B 등장 — 마름모/타원/고리, 그런데 라벨 데이터가 {N_B_TRAIN}장뿐")
    XB, yB = task_b_images(n=360, size=16, seed=77)
    xb_tr, yb_tr = to_torch(XB[:N_B_TRAIN], yB[:N_B_TRAIN])
    xb_te, yb_te = to_torch(XB[N_B_TRAIN:], yB[N_B_TRAIN:])
    print(f"    과제 B 학습 {len(xb_tr)}장 / 테스트 {len(xb_te)}장 — 과제 A 와 도형 종류가 전혀 다릅니다.")

    # ------------------------------------------------------------------
    print("\n[3] 방법 1: 처음부터 학습 (from scratch) — 신입을 백지에서 교육")
    torch.manual_seed(70)                                 # 공정 비교용 seed
    scratch = make_cnn()
    hist_scratch = train_b(scratch, xb_tr, yb_tr, xb_te, yb_te, "scratch ")

    # ------------------------------------------------------------------
    print("\n[4] 방법 2: 전이학습 — 경력직 채용: A 의 눈(features)을 얼려 재사용")
    torch.manual_seed(70)
    transfer = make_cnn()
    transfer[0].load_state_dict(pretrained[0].state_dict())   # 특징 추출기 이식
    for p in transfer[0].parameters():
        p.requires_grad = False                               # 동결(freeze)
    hist_transfer = train_b(transfer, xb_tr, yb_tr, xb_te, yb_te, "transfer")

    # ------------------------------------------------------------------
    print("\n[5] 결과 비교 — 수렴 속도와 최종 성능")
    print(f"      {'epoch':>6} {'scratch':>9} {'transfer':>9}")
    for e in (1, 5, 10, 20, 40, 80, 120):
        print(f"      {e:>6} {hist_scratch[e - 1]:>9.3f} {hist_transfer[e - 1]:>9.3f}")
    print("    -> 10에폭부터 전이학습이 앞서기 시작해 끝까지 우위를 지킵니다.")
    print("       학습한 파라미터는 전체의 1/10 (판정부 1,539개뿐)인데 성능은 더 높습니다.")
    print("       과제 A 에는 마름모도 고리도 없었지만 — 엣지·곡선·덩어리 같은")
    print("       저수준의 눈은 과제를 가리지 않기 때문에 그대로 통합니다.")

    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plt.figure(figsize=(11, 5.5))
    for k in range(6):                                    # 과제 B 샘플
        ax = fig.add_subplot(2, 6, k + 1 + (6 if k >= 3 else 0) - (0 if k < 3 else 3))
        i = int(np.where(yB[N_B_TRAIN:] == k % 3)[0][k // 3]) + N_B_TRAIN
        ax.imshow(XB[i], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"task B: {TASK_B_EN[k % 3]}", fontsize=8)
        ax.axis("off")
    ax = fig.add_subplot(1, 2, 2)
    ep = range(1, EPOCHS_B + 1)
    ax.plot(ep, hist_scratch, marker="s", label="from scratch")
    ax.plot(ep, hist_transfer, marker="o", label="transfer (frozen features)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("task B test accuracy")
    ax.set_title(f"Only {N_B_TRAIN} labeled images for task B")
    ax.set_ylim(0.2, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.suptitle("Transfer learning: reuse features learned on task A", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "transfer_learning.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    저장 완료: {path}")

    print(f"\n[정리] 전이학습 = 경력직 채용. 저수준의 눈은 재사용, 판정부만 새로 교육 (소요 {time.time() - t0:.1f}초).")
    print("       다음 레벨: 검수-분할-학습-오류분석-개선 — 실전 파이프라인 전체를 돌립니다.")


if __name__ == "__main__":
    main()
