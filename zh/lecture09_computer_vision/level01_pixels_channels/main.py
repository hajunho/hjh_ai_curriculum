"""
level01 — 像素、通道与图像运算

只用 numpy 亲手画出 RGB 图像，然后把
亮度(加法)、对比度(乘法)、反相、灰度转换、裁剪、缩放全部实现一遍。
结果汇总成 outputs/pixel_ops.png 这一张对比图。
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
BRIGHT_DELTA = 0.25   # 亮度调节量 (加法)
CONTRAST_GAIN = 1.8   # 对比度倍率 (乘法)


def make_scene(size: int = 48) -> np.ndarray:
    """不下载, 用 numpy 画出'天空 + 红色方块 + 黄色圆'的场景。"""
    img = np.zeros((size, size, 3), dtype="float32")
    # 天空: 上(亮蓝) -> 下(暗蓝) 的竖向渐变
    grad = np.linspace(0.9, 0.3, size)[:, None]           # (size,1) 竖直方向
    img[:, :, 2] = grad                                    # 蓝色通道
    img[:, :, 1] = grad * 0.6                              # 绿色稍微给一点
    # 红色方块 (建筑): 只让 R 通道浓
    img[26:44, 6:22] = [0.85, 0.15, 0.10]
    # 黄色圆 (太阳): R+G 浓, 没有 B
    yy, xx = np.mgrid[0:size, 0:size]
    sun = ((yy - 10) ** 2 + (xx - 36) ** 2) <= 6 ** 2
    img[sun] = [1.0, 0.9, 0.1]
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    """灰度转换: 反映人眼敏感度的加权平均(惯例系数)。"""
    return img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114


def resize_nearest(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """最近邻缩放: 每个新像素都去参考离它最近的原始像素。"""
    h, w = img.shape[:2]
    yy = np.clip(np.round(np.linspace(0, h - 1, new_h)), 0, h - 1).astype(int)
    xx = np.clip(np.round(np.linspace(0, w - 1, new_w)), 0, w - 1).astype(int)
    return img[yy][:, xx]      # 两次花式索引, 缩放就完成了


def stats(name: str, a: np.ndarray) -> None:
    print(f"    {name:<22} min={a.min():>6.2f}  mean={a.mean():>5.2f}  max={a.max():>6.2f}")


def main() -> None:
    np.random.seed(1)  # 固定随机数 seed(本关没用随机数, 但遵守规矩)

    # ------------------------------------------------------------------
    print("[1] 用 numpy 亲手画 RGB 图像 — 操作数组 = 画画")
    img = make_scene(48)
    print(f"    shape = {img.shape}  (高 48, 宽 48, 3 张通道: R/G/B)")
    print("    天空用渐变(linspace), 方块用切片, 圆用距离公式做掩码画出来的。")

    # ------------------------------------------------------------------
    print("\n[2] 通道分离 — '红'的真身就是各通道的数字差异")
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    box = (slice(30, 40), slice(10, 20))  # 红色方块内部区域
    print(f"    红色方块区域的各通道均值: R={r[box].mean():.2f}  G={g[box].mean():.2f}  B={b[box].mean():.2f}")
    print("    -> 只在 R 通道上亮 = 在我们眼里看成'红色'")

    # ------------------------------------------------------------------
    print("\n[3] 像素运算四兄弟 — 亮度=加法, 对比度=乘法, 反相=减法")
    brighter_raw = img + BRIGHT_DELTA                      # clip 之前 (故意的)
    brighter = np.clip(brighter_raw, 0.0, 1.0)
    contrast = np.clip((img - 0.5) * CONTRAST_GAIN + 0.5, 0.0, 1.0)
    inverted = 1.0 - img
    gray = to_gray(img)
    stats("原图", img)
    stats(f"亮度 +{BRIGHT_DELTA} (clip 前)", brighter_raw)
    stats(f"亮度 +{BRIGHT_DELTA} (clip 后)", brighter)
    stats(f"对比度 x{CONTRAST_GAIN}", contrast)
    stats("反相 1-x", inverted)
    stats("灰度转换", gray)
    print("    -> 注意 clip 前的 max 超过了 1.0。运算后 clip 要养成习惯。")

    # ------------------------------------------------------------------
    print("\n[4] 几何运算 — 裁剪就是切片, 缩放就是索引")
    crop = img[24:46, 4:24]                                # 只取红色方块周围
    up = resize_nearest(img, 96, 96)                       # 放大 2 倍
    down = resize_nearest(img, 16, 16)                     # 缩到 1/3
    print(f"    裁剪:     {img.shape} -> {crop.shape}   (img[24:46, 4:24])")
    print(f"    放大:     {img.shape} -> {up.shape}  (信息不会增加, 只多了台阶)")
    print(f"    缩小:     {img.shape} -> {down.shape}  ({img[:, :, 0].size} 个数字 -> {down[:, :, 0].size} 个)")

    # ------------------------------------------------------------------
    print("\n[5] 保存对比图 PNG")
    os.makedirs(OUT_DIR, exist_ok=True)
    panels = [
        ("original (RGB)", img, None), ("R channel", r, "gray"),
        ("G channel", g, "gray"), ("B channel", b, "gray"),
        (f"brightness +{BRIGHT_DELTA}", brighter, None), (f"contrast x{CONTRAST_GAIN}", contrast, None),
        ("inverted", inverted, None), ("grayscale", gray, "gray"),
        ("crop", crop, None), ("resize up 96x96", up, None),
        ("resize down 16x16", down, None), ("normalized (z-score)", (gray - gray.mean()) / gray.std(), "gray"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.5))
    for ax, (title, im, cmap) in zip(axes.ravel(), panels):
        if cmap == "gray" and title.startswith("normalized"):
            ax.imshow(im, cmap="gray")                     # z-score 范围自动
        elif cmap == "gray":
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(im)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle("Pixel & channel operations (numpy only)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "pixel_ops.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    保存完成: {path}")

    print("\n[小结] 图像编辑 = 数组算术。亮度是加法, 对比度是乘法, 裁剪是切片。")
    print("       下一关: 不是像素'单独'看, 而是'和邻居一起'看的运算 — 卷积。")


if __name__ == "__main__":
    main()
