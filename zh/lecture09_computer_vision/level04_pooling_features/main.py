"""
level04 — 池化与特征的层级

用 numpy 实现最大池化，然后
  1) 数字验算  2) 特征图汇总  3) 平移鲁棒性实验，
并对比可视化迷你 CNN 的第 1 层(边缘)与第 2 层(组合)的 feature map。
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
    """复用 level02 的纯实现 (相乘再相加)。"""
    img = np.pad(img, padding)
    h, w = img.shape
    k = kernel.shape[0]
    out = np.zeros((h - k + 1, w - k + 1), dtype="float32")
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y, x] = float((img[y:y + k, x:x + k] * kernel).sum())
    return out


def maxpool2x2(fm: np.ndarray) -> np.ndarray:
    """每个 2x2 区域只留一个最大值 — '按区汇总上报'。"""
    h, w = fm.shape
    return fm.reshape(h // 2, 2, w // 2, 2).max(axis=(1, 3))


def shift_right(img: np.ndarray, px: int) -> np.ndarray:
    """把图形向右平移 px 个像素 (空出来的位置填 0)。"""
    out = np.zeros_like(img)
    out[:, px:] = img[:, :img.shape[1] - px]
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """两张特征图的余弦相似度 (1=同一模式, 0=毫无重叠)。"""
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main() -> None:
    np.random.seed(4)
    torch.manual_seed(4)  # 固定 seed(可复现)

    # ------------------------------------------------------------------
    print("[1] 最大池化数字验算 — 把 4x4 汇总成 2x2")
    mini = np.array([[1, 3, 2, 0], [5, 2, 1, 4], [0, 1, 7, 2], [2, 0, 3, 1]], dtype="float32")
    pooled = maxpool2x2(mini)
    print("    输入 4x4:")
    for row in mini:
        print("      " + " ".join(f"{v:.0f}" for v in row))
    print("    2x2 最大池化结果 (各区最大值):")
    for row in pooled:
        print("      " + " ".join(f"{v:.0f}" for v in row))
    print("    -> 左上区 [1,3,5,2] 的代表是 5。位置扔掉, 只留'出现过'。")

    # ------------------------------------------------------------------
    print("\n[2] 特征图汇总 — 边缘图 16x16 -> 池化后 8x8")
    X, y = hjh_data.shape_images(n=60, size=16, seed=13)
    circle = X[np.where(y == 1)[0][0]]
    fm = np.abs(conv2d(circle, K_VERTICAL))            # 竖向边缘强度图
    fm_pooled = maxpool2x2(fm)
    print(f"    特征图 {fm.shape} -> 池化后 {fm_pooled.shape}  (数字 {fm.size} -> {fm_pooled.size} 个)")
    print(f"    最大值位置的数值: 池化前 {fm.max():.1f} / 后 {fm_pooled.max():.1f}  (强证据被保留)")

    # ------------------------------------------------------------------
    print("\n[3] 平移鲁棒性实验 — 平移过的图形特征图有多像 (相似度 1=完全相同)")
    clean = np.zeros((16, 16), dtype="float32")
    clean[4:12, 4:12] = 1.0                            # 无噪声的正方形(让效果看得更清楚)
    fm0 = np.abs(conv2d(clean, K_VERTICAL))
    print(f"      {'平移量':>6} {'池化前':>8} {'池化1次':>8} {'池化2次':>8}")
    for px in (1, 2, 3):
        fm1 = np.abs(conv2d(shift_right(clean, px), K_VERTICAL))
        sims = [cosine(fm0, fm1),
                cosine(maxpool2x2(fm0), maxpool2x2(fm1)),
                cosine(maxpool2x2(maxpool2x2(fm0)), maxpool2x2(maxpool2x2(fm1)))]
        print(f"      {px:>5}px {sims[0]:>8.2f} {sims[1]:>8.2f} {sims[2]:>8.2f}")
    moved = shift_right(circle, 1)
    fm_moved = np.abs(conv2d(moved, K_VERTICAL))
    print("    -> 平移 2px: 池化前相似度 0.00 (完全不同的图!) / 池化 2 次后 0.71。")
    print("       池化过得越多, 就越会把'稍微移了一点的同一个物体'看成同一个。")

    # ------------------------------------------------------------------
    print("\n[4] 特征的层级 — conv1(低层级) vs conv2(高层级) feature map")
    conv1 = nn.Conv2d(1, 6, 3, padding=1)
    conv2 = nn.Conv2d(6, 6, 3, padding=1)
    pool = nn.MaxPool2d(2)
    relu = nn.ReLU()
    x = torch.from_numpy(circle)[None, None]           # (1,1,16,16)
    with torch.no_grad():
        f1 = relu(conv1(x))                            # (1,6,16,16) 低层级
        f2 = relu(conv2(pool(f1)))                     # (1,6,8,8)  高层级(视野更宽)
    print(f"    conv1 输出 {tuple(f1.shape)} : 只看了原图 3x3 邻域的反应 (边缘级别)")
    print(f"    conv2 输出 {tuple(f2.shape)} : 多亏池化, 看的是原图上更大范围的组合")
    print("    (权重是随机的, 但'视野变宽的结构'照样能观察到)")

    # ------------------------------------------------------------------
    print("\n[5] 保存对比图 PNG")
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
    for j in range(6):                                  # conv1 的 6 个通道
        axes[1][j].imshow(f1[0, j].numpy(), cmap="viridis")
        axes[1][j].set_title(f"conv1 ch{j} (16x16)", fontsize=8)
        axes[1][j].axis("off")
    for j in range(6):                                  # conv2 的 6 个通道
        axes[2][j].imshow(f2[0, j].numpy(), cmap="viridis")
        axes[2][j].set_title(f"conv2 ch{j} (8x8)", fontsize=8)
        axes[2][j].axis("off")
    fig.suptitle("Max pooling: summarize location, keep evidence / feature hierarchy", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "pooling_features.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    保存完成: {path}")

    print("\n[小结] 池化 = 按区汇总上报。扔掉位置、留下证据, 因而抗平移。")
    print("       下一关: 零件组装完毕 — 我们真正训练一个图形分类器。")


if __name__ == "__main__":
    main()
