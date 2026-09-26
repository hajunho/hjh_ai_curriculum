"""
level01 — ピクセル・チャンネル・画像演算

numpy だけで RGB 画像を自分で描いたあと、
明るさ(足し算)・コントラスト(掛け算)・反転・白黒変換・クロップ・リサイズをすべて実装します。
結果は outputs/pixel_ops.png の比較パネル 1 枚にまとめます。
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
BRIGHT_DELTA = 0.25   # 明るさの調整量 (足し算)
CONTRAST_GAIN = 1.8   # コントラストの倍率 (掛け算)


def make_scene(size: int = 48) -> np.ndarray:
    """ダウンロードなしで numpy が「空 + 赤い四角形 + 黄色い円」の風景を描く。"""
    img = np.zeros((size, size, 3), dtype="float32")
    # 空: 上(明るい青) -> 下(暗い青) の縦方向グラデーション
    grad = np.linspace(0.9, 0.3, size)[:, None]           # (size,1) 縦方向
    img[:, :, 2] = grad                                    # 青チャンネル
    img[:, :, 1] = grad * 0.6                              # 緑を少しだけ
    # 赤い四角形 (建物): R チャンネルだけを濃く
    img[26:44, 6:22] = [0.85, 0.15, 0.10]
    # 黄色い円 (太陽): R+G を濃く、B はなし
    yy, xx = np.mgrid[0:size, 0:size]
    sun = ((yy - 10) ** 2 + (xx - 36) ** 2) <= 6 ** 2
    img[sun] = [1.0, 0.9, 0.1]
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    """白黒変換: 人の目の感度を反映した加重平均(慣例の係数)。"""
    return img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114


def resize_nearest(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """最近傍リサイズ: 新しいピクセルごとに最も近い元のピクセルを参照する。"""
    h, w = img.shape[:2]
    yy = np.clip(np.round(np.linspace(0, h - 1, new_h)), 0, h - 1).astype(int)
    xx = np.clip(np.round(np.linspace(0, w - 1, new_w)), 0, w - 1).astype(int)
    return img[yy][:, xx]      # ファンシーインデックス 2 回でリサイズ完成


def stats(name: str, a: np.ndarray) -> None:
    print(f"    {name:<18} min={a.min():>6.2f}  mean={a.mean():>5.2f}  max={a.max():>6.2f}")


def main() -> None:
    np.random.seed(1)  # 乱数 seed を固定(このレベルは乱数を使わないがルールは守る)

    # ------------------------------------------------------------------
    print("[1] numpy で RGB 画像を自分で描く — 配列操作 = 絵を描くこと")
    img = make_scene(48)
    print(f"    shape = {img.shape}  (高さ 48, 幅 48, チャンネル 3 枚: R/G/B)")
    print("    空はグラデーション(linspace)、四角形はスライシング、円は距離式のマスクで描きました。")

    # ------------------------------------------------------------------
    print("\n[2] チャンネル分離 — 「赤い」の正体はチャンネルごとの数字の差")
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    box = (slice(30, 40), slice(10, 20))  # 赤い四角形の内部領域
    print(f"    赤い四角形の領域のチャンネル平均: R={r[box].mean():.2f}  G={g[box].mean():.2f}  B={b[box].mean():.2f}")
    print("    -> R チャンネルだけで明るい = 私たちの目には「赤」に見える")

    # ------------------------------------------------------------------
    print("\n[3] ピクセル演算の 4 兄弟 — 明るさ=足し算, コントラスト=掛け算, 反転=引き算")
    brighter_raw = img + BRIGHT_DELTA                      # clip 前 (わざと)
    brighter = np.clip(brighter_raw, 0.0, 1.0)
    contrast = np.clip((img - 0.5) * CONTRAST_GAIN + 0.5, 0.0, 1.0)
    inverted = 1.0 - img
    gray = to_gray(img)
    stats("原画像", img)
    stats(f"明るさ +{BRIGHT_DELTA} (clip 前)", brighter_raw)
    stats(f"明るさ +{BRIGHT_DELTA} (clip 後)", brighter)
    stats(f"コントラスト x{CONTRAST_GAIN}", contrast)
    stats("反転 1-x", inverted)
    stats("白黒変換", gray)
    print("    -> clip 前は max が 1.0 を超えていることに注目。演算後の clip は習慣です。")

    # ------------------------------------------------------------------
    print("\n[4] 幾何演算 — クロップはスライシング、リサイズはインデックス")
    crop = img[24:46, 4:24]                                # 赤い四角形の周りだけ
    up = resize_nearest(img, 96, 96)                       # 2 倍に拡大
    down = resize_nearest(img, 16, 16)                     # 1/3 に縮小
    print(f"    クロップ: {img.shape} -> {crop.shape}   (img[24:46, 4:24])")
    print(f"    拡大:     {img.shape} -> {up.shape}  (情報は増えず階段だけができる)")
    print(f"    縮小:     {img.shape} -> {down.shape}  (数字 {img[:, :, 0].size} 個 -> {down[:, :, 0].size} 個)")

    # ------------------------------------------------------------------
    print("\n[5] 比較パネルの PNG を保存")
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
            ax.imshow(im, cmap="gray")                     # z-score は範囲を自動で
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
    print(f"    保存完了: {path}")

    print("\n[まとめ] 画像編集 = 配列の算数。明るさは足し算、コントラストは掛け算、クロップはスライシング。")
    print("         次のレベル: ピクセルを「単独」ではなく「隣と一緒に」見る演算 — 畳み込み。")


if __name__ == "__main__":
    main()
