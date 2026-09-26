"""
level02 — 畳み込み(Convolution)を理解する

畳み込みを numpy のループでゼロから実装します。
  1) 6x6 のミニ例題で掛け算・足し算の全過程を追跡
  2) 垂直/水平エッジカーネルで図形の境界線を検出
  3) カーネルの数字を変えるだけでぼかし/シャープ化になることを確認 + PNG 比較パネル
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

# 検査ハンコたち: カーネル = 「何を探すか」が刻まれた数字の板
K_VERTICAL = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="float32")   # 縦の境界
K_HORIZONTAL = K_VERTICAL.T                                                     # 横の境界
K_BLUR = np.full((3, 3), 1.0 / 9.0, dtype="float32")                            # 平均 = ぼかし
K_SHARPEN = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype="float32")    # シャープ化


def conv2d(img: np.ndarray, kernel: np.ndarray, padding: int = 0) -> np.ndarray:
    """畳み込みのすべて: カーネルを滑らせて「掛けて足す」。(教育用の素朴な実装)"""
    if padding > 0:
        img = np.pad(img, padding)                    # 端に 0 の縁を巻く
    h, w = img.shape
    k = kernel.shape[0]
    out = np.zeros((h - k + 1, w - k + 1), dtype="float32")
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            patch = img[y:y + k, x:x + k]             # ハンコが覆った領域
            out[y, x] = float((patch * kernel).sum()) # 9 組の積の合計
    return out


def main() -> None:
    np.random.seed(2)  # seed 固定(再現性)

    # ------------------------------------------------------------------
    print("[1] ミニ例題 — 畳み込み 1 回を数字そのまま追跡")
    mini = np.zeros((6, 6), dtype="float32")
    mini[:, 3:] = 1.0                                  # 左が暗く右が明るい縦の境界
    print("    入力 6x6 (左=0, 右=1 の縦の境界):")
    for row in mini:
        print("      " + " ".join(f"{v:.0f}" for v in row))
    print("    カーネル(垂直エッジ):")
    for row in K_VERTICAL:
        print("      " + " ".join(f"{v:+.0f}" for v in row))
    patch = mini[0:3, 2:5]                             # 境界にまたがる最初のハンコ位置
    print("    位置 (0,2) にハンコを押すと — 9 組の積:")
    terms = []
    for i in range(3):
        for j in range(3):
            terms.append(f"{patch[i, j]:.0f}x{K_VERTICAL[i, j]:+.0f}")
    print("      " + "  ".join(terms))
    print(f"      合計 = {(patch * K_VERTICAL).sum():+.0f}  (境界なので大きな値が出る)")
    flat = mini[0:3, 0:3]
    print(f"    平らな位置 (0,0) の合計 = {(flat * K_VERTICAL).sum():+.0f}  (変化なし -> 0)")

    # ------------------------------------------------------------------
    print("\n[2] conv2d の実装確認 — 出力サイズとパディング")
    out = conv2d(mini, K_VERTICAL)
    out_pad = conv2d(mini, K_VERTICAL, padding=1)
    print(f"    パディングなし: {mini.shape} -> {out.shape}   (n-k+1 に縮む)")
    print(f"    パディング 1  : {mini.shape} -> {out_pad.shape}   (サイズ維持)")
    print("    出力(パディングなし) — 境界の列だけ大きな値:")
    for row in out:
        print("      " + " ".join(f"{v:+4.0f}" for v in row))

    # ------------------------------------------------------------------
    print("\n[3] 図形にエッジカーネルを適用 — カーネルごとに違う質問を投げる")
    X, y = hjh_data.shape_images(n=60, size=16, seed=13)
    names = ["square", "circle", "triangle"]
    samples = [X[np.where(y == c)[0][0]] for c in range(3)]
    sq_v = np.abs(conv2d(samples[0], K_VERTICAL, padding=1))
    sq_h = np.abs(conv2d(samples[0], K_HORIZONTAL, padding=1))
    left_right = sq_v[:, :].max(axis=0)
    print(f"    四角形に垂直カーネル: 応答の最大 {sq_v.max():.1f} (左右の辺の位置で)")
    print(f"    四角形に水平カーネル: 応答の最大 {sq_h.max():.1f} (上下の辺の位置で)")
    print("    -> 同じ画像でもカーネルが違えば「見えるもの」が違います。")
    _ = left_right  # (参考用の計算)

    # ------------------------------------------------------------------
    print("\n[4] 同じ演算、違うカーネル — ぼかしとシャープ化")
    circle = samples[1]
    blurred = conv2d(circle, K_BLUR, padding=1)
    sharpened = np.clip(conv2d(circle, K_SHARPEN, padding=1), 0, 1)
    print(f"    原画像の標準偏差       = {circle.std():.3f}")
    print(f"    ぼかし後の標準偏差     = {blurred.std():.3f}  (値が平均のほうへ潰れる)")
    print(f"    シャープ化後の標準偏差 = {sharpened.std():.3f}  (差がさらに開く)")

    # ------------------------------------------------------------------
    print("\n[5] 比較パネルの PNG を保存 — 図形 3 種 x (原画像/垂直/水平/エッジ強度)")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(10, 7.5))
    for r, (name, img) in enumerate(zip(names, samples)):
        gv = conv2d(img, K_VERTICAL, padding=1)
        gh = conv2d(img, K_HORIZONTAL, padding=1)
        mag = np.sqrt(gv ** 2 + gh ** 2)               # 方向に無関係なエッジ強度
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
    print(f"    保存完了: {path}")

    print("\n[まとめ] 畳み込み = 数字のハンコを滑らせて「掛けて足す」。カーネルこそが質問である。")
    print("         次のレベル: カーネルの数字をデータに決めさせ、層として積む — CNN の組み立て。")


if __name__ == "__main__":
    main()
