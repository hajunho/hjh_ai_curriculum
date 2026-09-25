"""
level04 — 풀링과 특징의 계층

맥스 풀링을 numpy 로 구현해
  1) 숫자 검산  2) 특징 맵 요약  3) 이동 강건성 실험을 하고,
미니 CNN 의 1층(엣지)과 2층(조합) feature map 을 비교 시각화합니다.
"""

import os
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
K_VERTICAL = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="float32")


def conv2d(img: np.ndarray, kernel: np.ndarray, padding: int = 1) -> np.ndarray:
    """level02 의 순수 구현 재사용 (곱해서 더하기)."""
    img = np.pad(img, padding)
    h, w = img.shape
    k = kernel.shape[0]
    out = np.zeros((h - k + 1, w - k + 1), dtype="float32")
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y, x] = float((img[y:y + k, x:x + k] * kernel).sum())
    return out


def maxpool2x2(fm: np.ndarray) -> np.ndarray:
    """2x2 구역마다 최댓값 하나만 남긴다 — '구역별 요약 보고'."""
    h, w = fm.shape
    return fm.reshape(h // 2, 2, w // 2, 2).max(axis=(1, 3))


def shift_right(img: np.ndarray, px: int) -> np.ndarray:
    """도형을 오른쪽으로 px 픽셀 민다 (빈 자리는 0)."""
    out = np.zeros_like(img)
    out[:, px:] = img[:, :img.shape[1] - px]
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """두 특징 맵의 코사인 유사도 (1=같은 패턴, 0=겹침 없음)."""
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main() -> None:
    np.random.seed(4)
    torch.manual_seed(4)  # seed 고정(재현성)

    # ------------------------------------------------------------------
    print("[1] 맥스 풀링 숫자 검산 — 4x4 를 2x2 로 요약")
    mini = np.array([[1, 3, 2, 0], [5, 2, 1, 4], [0, 1, 7, 2], [2, 0, 3, 1]], dtype="float32")
    pooled = maxpool2x2(mini)
    print("    입력 4x4:")
    for row in mini:
        print("      " + " ".join(f"{v:.0f}" for v in row))
    print("    2x2 맥스 풀링 결과 (구역별 최댓값):")
    for row in pooled:
        print("      " + " ".join(f"{v:.0f}" for v in row))
    print("    -> 왼쪽 위 구역 [1,3,5,2] 의 대표는 5. 위치는 버리고 '있었다'만 남깁니다.")

    # ------------------------------------------------------------------
    print("\n[2] 특징 맵 요약 — 엣지 맵 16x16 -> 풀링 8x8")
    X, y = hjh_data.shape_images(n=60, size=16, seed=13)
    circle = X[np.where(y == 1)[0][0]]
    fm = np.abs(conv2d(circle, K_VERTICAL))            # 세로 엣지 강도 맵
    fm_pooled = maxpool2x2(fm)
    print(f"    특징 맵 {fm.shape} -> 풀링 후 {fm_pooled.shape}  (숫자 {fm.size} -> {fm_pooled.size}개)")
    print(f"    최댓값 위치의 값: 풀링 전 {fm.max():.1f} / 후 {fm_pooled.max():.1f}  (강한 증거는 보존)")

    # ------------------------------------------------------------------
    print("\n[3] 이동 강건성 실험 — 밀린 도형의 특징 맵은 얼마나 닮았나 (유사도 1=동일)")
    clean = np.zeros((16, 16), dtype="float32")
    clean[4:12, 4:12] = 1.0                            # 잡음 없는 사각형(효과가 또렷하게 보이도록)
    fm0 = np.abs(conv2d(clean, K_VERTICAL))
    print(f"      {'이동량':>6} {'풀링 전':>8} {'풀링 1회':>8} {'풀링 2회':>8}")
    for px in (1, 2, 3):
        fm1 = np.abs(conv2d(shift_right(clean, px), K_VERTICAL))
        sims = [cosine(fm0, fm1),
                cosine(maxpool2x2(fm0), maxpool2x2(fm1)),
                cosine(maxpool2x2(maxpool2x2(fm0)), maxpool2x2(maxpool2x2(fm1)))]
        print(f"      {px:>5}px {sims[0]:>8.2f} {sims[1]:>8.2f} {sims[2]:>8.2f}")
    moved = shift_right(circle, 1)
    fm_moved = np.abs(conv2d(moved, K_VERTICAL))
    print("    -> 2px 이동: 풀링 전 유사도 0.00 (완전히 다른 맵!) / 풀링 2회 후 0.71.")
    print("       풀링을 거칠수록 '조금 밀린 같은 물체'를 같은 것으로 보게 됩니다.")

    # ------------------------------------------------------------------
    print("\n[4] 특징의 계층 — conv1(저수준) vs conv2(고수준) feature map")
    conv1 = nn.Conv2d(1, 6, 3, padding=1)
    conv2 = nn.Conv2d(6, 6, 3, padding=1)
    pool = nn.MaxPool2d(2)
    relu = nn.ReLU()
    x = torch.from_numpy(circle)[None, None]           # (1,1,16,16)
    with torch.no_grad():
        f1 = relu(conv1(x))                            # (1,6,16,16) 저수준
        f2 = relu(conv2(pool(f1)))                     # (1,6,8,8)  고수준(더 넓은 시야)
    print(f"    conv1 출력 {tuple(f1.shape)} : 원본 3x3 이웃만 본 반응 (엣지 수준)")
    print(f"    conv2 출력 {tuple(f2.shape)} : 풀링 덕에 원본 기준 더 넓은 영역의 조합을 봄")
    print("    (무게는 무작위지만 '시야가 넓어지는 구조'는 그대로 관찰됩니다)")

    # ------------------------------------------------------------------
    print("\n[5] 비교판 PNG 저장")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(3, 6, figsize=(13, 7))
    top = [(circle, "input", "gray"), (fm, "edge map 16x16", "magma"),
           (fm_pooled, "pooled 8x8", "magma"), (moved, "input shifted +2px", "gray"),
           (fm_moved, "edge map (shifted)", "magma"),
           (maxpool2x2(fm_moved), "pooled (shifted)", "magma")]
    for ax, (im, title, cmap) in zip(axes[0], top):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    for j in range(6):                                  # conv1 채널 6개
        axes[1][j].imshow(f1[0, j].numpy(), cmap="viridis")
        axes[1][j].set_title(f"conv1 ch{j} (16x16)", fontsize=8)
        axes[1][j].axis("off")
    for j in range(6):                                  # conv2 채널 6개
        axes[2][j].imshow(f2[0, j].numpy(), cmap="viridis")
        axes[2][j].set_title(f"conv2 ch{j} (8x8)", fontsize=8)
        axes[2][j].axis("off")
    fig.suptitle("Max pooling: summarize location, keep evidence / feature hierarchy", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "pooling_features.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    저장 완료: {path}")

    print("\n[정리] 풀링 = 구역별 요약 보고. 위치는 버리고 증거는 남겨 이동에 강해진다.")
    print("       다음 레벨: 부품 조립 끝 — 도형 분류기를 실제로 학습시킵니다.")


if __name__ == "__main__":
    main()
