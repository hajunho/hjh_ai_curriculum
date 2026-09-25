"""
level06 — 데이터 증강 (Data Augmentation)

numpy 로 회전·이동·반전·잡음 증강을 직접 구현하고,
학습 데이터가 적을 때(120장) 증강 유무에 따른 테스트 성능 차이를
같은 모델·같은 설정으로 비교 실험합니다.
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
N_TRAIN = 120          # 일부러 '데이터 부족' 상황을 만든다
EPOCHS = 30
LR = 1e-3


# ----------------------------------------------------------------------
# numpy 증강 함수들 — 라이브러리 없이 전부 배열 연산
# ----------------------------------------------------------------------

def aug_rotate90(img: np.ndarray, k: int) -> np.ndarray:
    """90도 단위 회전. 도형의 클래스는 변하지 않는다(라벨 보존 변형)."""
    return np.rot90(img, k).copy()


def aug_flip(img: np.ndarray) -> np.ndarray:
    """좌우 반전."""
    return img[:, ::-1].copy()


def aug_shift(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """상하좌우 이동 (빈 자리는 0). np.roll 후 밀려 들어온 부분을 지운다."""
    out = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
    if dy > 0:
        out[:dy] = 0
    elif dy < 0:
        out[dy:] = 0
    if dx > 0:
        out[:, :dx] = 0
    elif dx < 0:
        out[:, dx:] = 0
    return out


def aug_noise(img: np.ndarray, rng: np.random.Generator, std: float = 0.10) -> np.ndarray:
    """가우시안 잡음 추가 — 촬영 조건의 잔떨림을 흉내."""
    return np.clip(img + rng.normal(0, std, img.shape), 0.0, 1.0).astype("float32")


def augment_dataset(X: np.ndarray, y: np.ndarray, per_image: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """이미지마다 무작위 증강본 per_image 장을 만들어 원본과 합친다."""
    outs, labels = [X], [y]
    for _ in range(per_image):
        batch = np.empty_like(X)
        for i, img in enumerate(X):
            a = img
            a = aug_rotate90(a, int(rng.integers(0, 4)))     # 회전 0/90/180/270
            if rng.random() < 0.5:
                a = aug_flip(a)                               # 반전
            a = aug_shift(a, int(rng.integers(-2, 3)), int(rng.integers(-2, 3)))
            a = aug_noise(a, rng)                             # 잡음
            batch[i] = a
        outs.append(batch)
        labels.append(y)
    return np.concatenate(outs), np.concatenate(labels)


# ----------------------------------------------------------------------
# 소형 CNN (level05 와 동일 구조)
# ----------------------------------------------------------------------

def make_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(256, 3))


def train_and_eval(xtr, ytr, xte, yte, tag: str) -> float:
    """고정 설정으로 학습해 테스트 정확도를 반환 (공정 비교)."""
    torch.manual_seed(6)                                      # 두 실험의 초기 무게 동일
    model = make_cnn()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss_fn(model(xtr[idx]), ytr[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = float((model(xte).argmax(1) == yte).float().mean())
    print(f"      {tag:<28} 학습 {len(xtr):>4}장 -> 테스트 정확도 {acc:.3f}")
    return acc


def to_torch(X: np.ndarray, y: np.ndarray):
    return torch.from_numpy(X).unsqueeze(1), torch.from_numpy(y)


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(6)
    np.random.seed(6)  # seed 고정(재현성)

    # ------------------------------------------------------------------
    print("[1] 데이터 부족 상황 만들기 — 학습 120장 / 테스트 300장")
    X, y = hjh_data.shape_images(n=420, size=16, seed=13)
    Xtr, ytr = X[:N_TRAIN], y[:N_TRAIN]
    Xte, yte = X[N_TRAIN:], y[N_TRAIN:]
    print(f"    실무에서 흔한 상황: 라벨 붙은 이미지가 {N_TRAIN}장뿐입니다.")

    # ------------------------------------------------------------------
    print("\n[2] 증강 구현 — 원본 1장에서 변형본을 만들어 본다")
    demo = Xtr[0]
    variants = [("original", demo),
                ("rotate 90", aug_rotate90(demo, 1)),
                ("rotate 180", aug_rotate90(demo, 2)),
                ("flip", aug_flip(demo)),
                ("shift (+2,+2)", aug_shift(demo, 2, 2)),
                ("noise", aug_noise(demo, rng))]
    print("    회전/반전/이동/잡음 — 전부 '라벨이 변하지 않는' 변형입니다.")
    print(f"    ({CLASS_EN[int(ytr[0])]} 는 돌리고 밀고 잡음을 섞어도 {CLASS_EN[int(ytr[0])]})")

    # ------------------------------------------------------------------
    print("\n[3] 증강으로 데이터셋 부풀리기 — 120장 -> 600장")
    Xaug, yaug = augment_dataset(Xtr, ytr, per_image=4, rng=rng)
    print(f"    원본 {len(Xtr)}장 + 증강본 {len(Xaug) - len(Xtr)}장 = {len(Xaug)}장")
    print("    (테스트 300장에는 증강을 절대 하지 않습니다 — 시험지는 원본 그대로)")

    # ------------------------------------------------------------------
    print("\n[4] 공정 비교 실험 — 같은 모델, 같은 에폭, 같은 초기 무게")
    xte_t, yte_t = to_torch(Xte, yte)
    acc_plain = train_and_eval(*to_torch(Xtr, ytr), xte_t, yte_t, "증강 없음 (120장)")
    acc_aug = train_and_eval(*to_torch(Xaug, yaug), xte_t, yte_t, "증강 있음 (600장)")
    gain = (acc_aug - acc_plain) * 100
    print(f"    -> 증강 효과: 테스트 정확도 {gain:+.1f}%p. 데이터를 한 장도 새로 만들지 않고 얻은 이득입니다.")

    # ------------------------------------------------------------------
    print("\n[5] 비교판 PNG 저장")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plt.figure(figsize=(11, 6))
    for k, (name, im) in enumerate(variants):
        ax = fig.add_subplot(2, 6, k + 1)
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    ax = fig.add_subplot(2, 1, 2)
    bars = ax.bar(["no augmentation\n(120 imgs)", "with augmentation\n(600 imgs)"],
                  [acc_plain, acc_aug], width=0.4)
    for b, v in zip(bars, [acc_plain, acc_aug]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("test accuracy (300 imgs)")
    ax.set_title("Same model & epochs - augmentation only difference")
    fig.suptitle("Data augmentation: label-preserving transforms buy accuracy for free", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "augmentation.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    저장 완료: {path}")

    print(f"\n[정리] 증강 = 라벨 보존 변형으로 공짜 데이터 만들기 (소요 {time.time() - t0:.1f}초).")
    print("       다음 레벨: 남의 학습 결과를 빌려 오는 더 큰 지렛대 — 전이학습.")


if __name__ == "__main__":
    main()
