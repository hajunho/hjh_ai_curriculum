"""
level00 — 计算机是怎么"看"图像的

用三种表现方式确认图像只不过是一张"数字网格"。
  1) 直接打印数字表  2) 文本艺术(亮度->字符)  3) 保存为 PNG
另外通过降低分辨率，体会"瓷砖数量"就是信息量。
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
CLASS_NAMES = ["正方形", "圆形", "三角形"]
CLASS_NAMES_EN = ["square", "circle", "triangle"]  # 图片用(字体兼容)
CHARS = " .:-=+*#@"  # 从暗到亮排列的字符调色板


def to_ascii(img: np.ndarray) -> str:
    """把亮度(0~1)换成字符，做出'文本艺术'。
    渲染 = 把数字换成好看的符号的规则，仅此而已。"""
    lines = []
    for row in img:
        idx = (np.clip(row, 0.0, 1.0) * (len(CHARS) - 1)).astype(int)
        # 终端字符是竖长的，所以每个像素打两个字符比例才对
        lines.append("".join(CHARS[i] * 2 for i in idx))
    return "\n".join(lines)


def downscale_mean(img: np.ndarray, factor: int) -> np.ndarray:
    """把 factor x factor 的块压成一个平均值，从而降低分辨率。"""
    h, w = img.shape
    return img.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def main() -> None:
    np.random.seed(0)  # 固定随机数 seed(可复现)

    # ------------------------------------------------------------------
    print("[1] 生成图形图像数据 — 不下载, 用 numpy 当场画出来")
    X, y = hjh_data.shape_images(n=300, size=16, seed=13)
    print(f"    X.shape = {X.shape}  (300 张图像, 每张是 16 行 x 16 列的数字表)")
    print(f"    值范围 = {X.min():.2f} ~ {X.max():.2f}  (0=黑, 1=白)")
    counts = {CLASS_NAMES[c]: int((y == c).sum()) for c in range(3)}
    print(f"    类别构成 = {counts}")

    # 挑一张圆形(label=1)图像作为代表
    sample = X[np.where(y == 1)[0][0]]

    # ------------------------------------------------------------------
    print("\n[2] 把一张图像当成'数字表'来看 — 这才是计算机看到的原件")
    print("    (只显示小数第一位。1.0 聚集的地方就是图形)")
    for row in sample:
        print("    " + " ".join(f"{v:.1f}"[1:] for v in row))  # '0.7'->'.7' 缩写
    print("    -> 在人眼里只是一堆数字, 但这就是图像的全部。")

    # ------------------------------------------------------------------
    print("\n[3] 把同一个数组当成'文本艺术'来看 — 只多加了一条数字->字符的规则")
    print(to_ascii(sample))
    print("    -> 数组没变, 可图形(圆)出现了。图画就是数字的排列。")

    # ------------------------------------------------------------------
    print("\n[4] 分辨率实验 — 减少马赛克瓷砖数会怎样")
    for factor, name in [(1, "16x16 (原图)"), (2, "8x8"), (4, "4x4")]:
        small = sample if factor == 1 else downscale_mean(sample, factor)
        print(f"\n    --- {name}: 共 {small.size} 个数字 ---")
        for line in to_ascii(small).split("\n"):
            print("    " + line)
    print("\n    -> 到了 4x4 就很难分清是圆还是正方形了。")
    print("       分辨率既是'信息量'也是'计算成本'。任务需要多少就用多少。")

    # ------------------------------------------------------------------
    print("\n[5] 保存为 PNG — 把刚才看到的同一个数组变成图片文件")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for c in range(3):  # 每个类别 3 张
        idx = np.where(y == c)[0][:3]
        for j, i in enumerate(idx):
            ax = axes[c][j]
            ax.imshow(X[i], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{CLASS_NAMES_EN[c]} (y={c})", fontsize=9)
            ax.axis("off")
    fig.suptitle("shape_images: 16x16 grayscale, 3 classes", fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "shapes_grid.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    保存完成: {path}")
    print("    打开文件看看。这和 [2] 的数字表是完全相同的数据。")

    print("\n[小结] 图像 = 亮度数字的网格(电子表格)。")
    print("       下一关: 图像既然是数字, 编辑就是算术 — 像素、通道与图像运算。")


if __name__ == "__main__":
    main()
