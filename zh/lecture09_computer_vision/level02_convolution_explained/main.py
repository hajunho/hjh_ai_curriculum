"""
level02 — 理解卷积 (Convolution)

用 numpy 的循环从零实现卷积。
  1) 用 6x6 迷你例子追踪乘-加的全过程
  2) 用垂直/水平边缘核检出图形的轮廓线
  3) 确认只改核里的数字就能变成模糊/锐化 + PNG 对比图
"""

import os
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# 各式检查印章: 核 = 刻着"要找什么"的数字板
K_VERTICAL = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="float32")   # 竖向边界
K_HORIZONTAL = K_VERTICAL.T                                                     # 横向边界
K_BLUR = np.full((3, 3), 1.0 / 9.0, dtype="float32")                            # 平均 = 模糊
K_SHARPEN = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype="float32")    # 锐化


def conv2d(img: np.ndarray, kernel: np.ndarray, padding: int = 0) -> np.ndarray:
    """卷积的全部: 滑动核, '相乘再相加'。(教学用的纯实现)"""
    if padding > 0:
        img = np.pad(img, padding)                    # 在边缘围一圈 0
    h, w = img.shape
    k = kernel.shape[0]
    out = np.zeros((h - k + 1, w - k + 1), dtype="float32")
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            patch = img[y:y + k, x:x + k]             # 印章盖住的区域
            out[y, x] = float((patch * kernel).sum()) # 9 对乘积之和
    return out


def main() -> None:
    np.random.seed(2)  # 固定 seed(可复现)

    # ------------------------------------------------------------------
    print("[1] 迷你例子 — 把一次卷积用数字原样追踪一遍")
    mini = np.zeros((6, 6), dtype="float32")
    mini[:, 3:] = 1.0                                  # 左暗右亮的竖向边界
    print("    输入 6x6 (左=0, 右=1 的竖向边界):")
    for row in mini:
        print("      " + " ".join(f"{v:.0f}" for v in row))
    print("    核(垂直边缘):")
    for row in K_VERTICAL:
        print("      " + " ".join(f"{v:+.0f}" for v in row))
    patch = mini[0:3, 2:5]                             # 压在边界上的第一个印章位置
    print("    在位置 (0,2) 盖一次印章 — 9 对乘积:")
    terms = []
    for i in range(3):
        for j in range(3):
            terms.append(f"{patch[i, j]:.0f}x{K_VERTICAL[i, j]:+.0f}")
    print("      " + "  ".join(terms))
    print(f"      合计 = {(patch * K_VERTICAL).sum():+.0f}  (因为是边界, 出来一个大值)")
    flat = mini[0:3, 0:3]
    print(f"    平坦位置 (0,0) 的合计 = {(flat * K_VERTICAL).sum():+.0f}  (没有变化 -> 0)")

    # ------------------------------------------------------------------
    print("\n[2] 确认 conv2d 实现 — 输出尺寸与填充")
    out = conv2d(mini, K_VERTICAL)
    out_pad = conv2d(mini, K_VERTICAL, padding=1)
    print(f"    无填充: {mini.shape} -> {out.shape}   (缩成 n-k+1)")
    print(f"    填充 1 : {mini.shape} -> {out_pad.shape}   (尺寸保持)")
    print("    输出(无填充) — 只在边界那一列出现大值:")
    for row in out:
        print("      " + " ".join(f"{v:+4.0f}" for v in row))

    # ------------------------------------------------------------------
    print("\n[3] 给图形套上边缘核 — 每个核问的是不同的问题")
    X, y = hjh_data.shape_images(n=60, size=16, seed=13)
    names = ["square", "circle", "triangle"]
    samples = [X[np.where(y == c)[0][0]] for c in range(3)]
    sq_v = np.abs(conv2d(samples[0], K_VERTICAL, padding=1))
    sq_h = np.abs(conv2d(samples[0], K_HORIZONTAL, padding=1))
    left_right = sq_v[:, :].max(axis=0)
    print(f"    正方形 + 垂直核: 最大响应 {sq_v.max():.1f} (出现在左右两边的位置)")
    print(f"    正方形 + 水平核: 最大响应 {sq_h.max():.1f} (出现在上下两边的位置)")
    print("    -> 即使是同一张图像, 核不同'看见的东西'也不同。")
    _ = left_right  # (仅供参考的计算)

    # ------------------------------------------------------------------
    print("\n[4] 同一个运算, 不同的核 — 模糊与锐化")
    circle = samples[1]
    blurred = conv2d(circle, K_BLUR, padding=1)
    sharpened = np.clip(conv2d(circle, K_SHARPEN, padding=1), 0, 1)
    print(f"    原图标准差     = {circle.std():.3f}")
    print(f"    模糊后标准差   = {blurred.std():.3f}  (数值被揉向平均)")
    print(f"    锐化后标准差   = {sharpened.std():.3f}  (差距被进一步拉开)")

    # ------------------------------------------------------------------
    print("\n[5] 保存对比图 PNG — 3 种图形 x (原图/垂直/水平/边缘强度)")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(10, 7.5))
    for r, (name, img) in enumerate(zip(names, samples)):
        gv = conv2d(img, K_VERTICAL, padding=1)
        gh = conv2d(img, K_HORIZONTAL, padding=1)
        mag = np.sqrt(gv ** 2 + gh ** 2)               # 与方向无关的边缘强度
        panels = [(img, f"{name} (input)", "gray"),
                  (gv, "vertical edges", "coolwarm"),
                  (gh, "horizontal edges", "coolwarm"),
                  (mag, "edge magnitude", "magma")]
        for c, (im, title, cmap) in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(im, cmap=cmap)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.suptitle("Hand-made 3x3 kernels: convolution as a pattern detector", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "convolution_edges.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    保存完成: {path}")

    print("\n[小结] 卷积 = 滑动数字印章, '相乘再相加'。核本身就是问题。")
    print("       下一关: 让数据来定核里的数字, 并把它们堆成层 — 组装 CNN。")


if __name__ == "__main__":
    main()
