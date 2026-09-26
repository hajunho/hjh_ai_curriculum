"""
level00 — コンピュータが画像を見る仕組み

画像が「数字の格子」にすぎないことを、3 つの表現で確認します。
  1) 数字の表をそのまま出力  2) テキストアート(明るさ->文字)  3) PNG 保存
さらに解像度を下げていき、「タイルの枚数」が情報量であることを体験します。
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
CLASS_NAMES = ["四角形", "円", "三角形"]
CLASS_NAMES_EN = ["square", "circle", "triangle"]  # 図用(フォント互換)
CHARS = " .:-=+*#@"  # 暗い -> 明るい の順に並べた文字パレット


def to_ascii(img: np.ndarray) -> str:
    """明るさ(0〜1)を文字に変換して「テキストアート」を作る。
    レンダリング = 数字を見やすい記号に変える規則、それ以上のものではない。"""
    lines = []
    for row in img:
        idx = (np.clip(row, 0.0, 1.0) * (len(CHARS) - 1)).astype(int)
        # ターミナルの文字は縦に長いので、1 ピクセルにつき文字 2 個で比率が合う
        lines.append("".join(CHARS[i] * 2 for i in idx))
    return "\n".join(lines)


def downscale_mean(img: np.ndarray, factor: int) -> np.ndarray:
    """factor x factor のブロックを平均 1 つにまとめて解像度を下げる。"""
    h, w = img.shape
    return img.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def main() -> None:
    np.random.seed(0)  # 乱数 seed を固定(再現性)

    # ------------------------------------------------------------------
    print("[1] 図形画像データを作る — ダウンロードなしで numpy がその場で生成")
    X, y = hjh_data.shape_images(n=300, size=16, seed=13)
    print(f"    X.shape = {X.shape}  (画像 300 枚、それぞれ 16 行 x 16 列の数字の表)")
    print(f"    値の範囲 = {X.min():.2f} 〜 {X.max():.2f}  (0=黒, 1=白)")
    counts = {CLASS_NAMES[c]: int((y == c).sum()) for c in range(3)}
    print(f"    クラス構成 = {counts}")

    # 円(label=1)の画像 1 枚を代表として選ぶ
    sample = X[np.where(y == 1)[0][0]]

    # ------------------------------------------------------------------
    print("\n[2] 画像 1 枚を「数字の表」としてそのまま見る — コンピュータが見ている原本")
    print("    (小数第 1 位だけ表示。1.0 が集まっている場所が図形です)")
    for row in sample:
        print("    " + " ".join(f"{v:.1f}"[1:] for v in row))  # '0.7'->'.7' に省略
    print("    -> 人の目にはただの数字の山ですが、これが画像のすべてです。")

    # ------------------------------------------------------------------
    print("\n[3] 同じ配列を「テキストアート」で見る — 数字->文字の規則を 1 つ足すだけ")
    print(to_ascii(sample))
    print("    -> 配列はそのままなのに図形(円)が見えます。絵とは数字の並びです。")

    # ------------------------------------------------------------------
    print("\n[4] 解像度の実験 — モザイクのタイル数を減らすとどうなるか")
    for factor, name in [(1, "16x16 (原本)"), (2, "8x8"), (4, "4x4")]:
        small = sample if factor == 1 else downscale_mean(sample, factor)
        print(f"\n    --- {name}: 数字 {small.size} 個 ---")
        for line in to_ascii(small).split("\n"):
            print("    " + line)
    print("\n    -> 4x4 まで来ると円なのか四角形なのか区別が難しくなります。")
    print("       解像度は「情報量」であり「計算コスト」です。課題に必要な分だけ使います。")

    # ------------------------------------------------------------------
    print("\n[5] PNG に保存 — ここまで見てきたのと同じ配列を画像ファイルに")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for c in range(3):  # クラスごとに 3 枚ずつ
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
    print(f"    保存完了: {path}")
    print("    ファイルを開いてみてください。[2] の数字の表とまったく同じデータです。")

    print("\n[まとめ] 画像 = 明るさの数字の格子(スプレッドシート)。")
    print("         次のレベル: 画像が数字なら、編集は算数だ — ピクセル・チャンネル・画像演算。")


if __name__ == "__main__":
    main()
