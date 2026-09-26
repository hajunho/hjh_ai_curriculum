"""
level03 — CNN の構造を作る

torch でミニ CNN を定義します (学習はまだしません)。
  1) Conv2d の重みテンソルを解剖  2) 出力サイズの公式を検算
  3) 層ごとの shape 変化を追跡    4) パラメータ数の手計算 vs torch の集計
  5) MLP とのパラメータ数の比較
"""

import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data


def conv_out(size: int, kernel: int, stride: int = 1, padding: int = 0) -> int:
    """出力サイズの公式: (in + 2p - k) / s + 1 — この講義で最も頻繁に使う式。"""
    return (size + 2 * padding - kernel) // stride + 1


class TinyCNN(nn.Module):
    """16x16 の白黒図形 3 クラス用のミニ CNN。
    設計のリズム: チャンネルは増やし(1->8->16->32)、空間は減らす(16->16->8->4)。"""

    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.head = nn.Linear(32 * 4 * 4, n_classes)

    def forward_traced(self, x: torch.Tensor) -> torch.Tensor:
        """層を 1 つ通るたびに shape を出力しながら通過させる。"""
        steps = [
            ("入力 (B,C,H,W)", lambda t: t),
            ("conv1 1->8ch, s1, p1", lambda t: self.relu(self.conv1(t))),
            ("conv2 8->16ch, s2, p1", lambda t: self.relu(self.conv2(t))),
            ("conv3 16->32ch, s2, p1", lambda t: self.relu(self.conv3(t))),
            ("flatten", self.flatten),
            ("linear 512->3", self.head),
        ]
        for name, fn in steps:
            x = fn(x)
            print(f"      {name:<24} -> {tuple(x.shape)}")
        return x


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def main() -> None:
    torch.manual_seed(3)
    np.random.seed(3)  # seed 固定(再現性)

    # ------------------------------------------------------------------
    print("[1] Conv2d 層の解剖 — 「ハンコを何本も持った検査チーム」")
    conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
    w = conv.weight
    print(f"    weight.shape = {tuple(w.shape)}  (ハンコ 8 本、それぞれ 1 チャンネル x 3x3)")
    print(f"    bias.shape   = {tuple(conv.bias.shape)}  (ハンコごとにバイアス 1 個)")
    print(f"    この層のパラメータ = 8 x (1x3x3) + 8 = {8 * 9 + 8} 個  "
          f"(torch の集計: {count_params(conv)} 個)")

    # ------------------------------------------------------------------
    print("\n[2] 出力サイズの公式を検算 — out = (in + 2p - k)/s + 1")
    print("    入力 16, カーネル 3 のとき:")
    print(f"      {'stride':>7} {'padding':>8} {'公式':>6} {'実測':>6}")
    x16 = torch.zeros(1, 1, 16, 16)
    for s, p in [(1, 0), (1, 1), (2, 1), (2, 0)]:
        c = nn.Conv2d(1, 1, 3, stride=s, padding=p)
        real = c(x16).shape[-1]
        pred = conv_out(16, 3, s, p)
        print(f"      {s:>7} {p:>8} {pred:>6} {real:>6}")
    print("    -> 公式と実測はいつでも一致します。層の設計はこの式で最後まで追跡してください。")

    # ------------------------------------------------------------------
    print("\n[3] ミニ CNN に図形のバッチを通しながら shape を追跡")
    X, y = hjh_data.shape_images(n=32, size=16, seed=13)
    batch = torch.from_numpy(X).unsqueeze(1)           # (32,16,16) -> (32,1,16,16) B-C-H-W
    print(f"    numpy {X.shape} -> torch {tuple(batch.shape)}  (バッチ, チャンネル, 高さ, 幅)")
    model = TinyCNN()
    with torch.no_grad():
        scores = model.forward_traced(batch)
    print(f"    出力 = 画像 32 枚それぞれの 3 クラスのスコア。例: 1 枚目 {scores[0].numpy().round(2)}")
    print("    (まだ学習前のランダムな重みなのでスコアに意味はありません — 構造だけ確認)")

    # ------------------------------------------------------------------
    print("\n[4] 層ごとのパラメータ数 — 手計算 vs torch の集計")
    rows = [
        ("conv1", "8 x (1x3x3) + 8", 8 * 1 * 9 + 8, model.conv1),
        ("conv2", "16 x (8x3x3) + 16", 16 * 8 * 9 + 16, model.conv2),
        ("conv3", "32 x (16x3x3) + 32", 32 * 16 * 9 + 32, model.conv3),
        ("head", "3 x 512 + 3", 3 * 512 + 3, model.head),
    ]
    print(f"      {'層':<7} {'公式':<20} {'手計算':>8} {'torch':>8}")
    for name, formula, hand, mod in rows:
        print(f"      {name:<7} {formula:<20} {hand:>8,} {count_params(mod):>8,}")
    total = count_params(model)
    print(f"      {'合計':<28} {sum(r[2] for r in rows):>8,} {total:>8,}")
    print("    -> Conv のパラメータは画像サイズと無関係、Linear は Flatten のサイズに比例します。")

    # ------------------------------------------------------------------
    print("\n[5] 同じ入力を MLP で処理したら? — 重み共有の威力")
    mlp = nn.Sequential(nn.Flatten(), nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))
    print(f"    MLP  (256->128->64->3)  パラメータ: {count_params(mlp):>8,} 個")
    print(f"    CNN  (TinyCNN)          パラメータ: {total:>8,} 個")
    print("    -> CNN は「同じハンコを画像全体で再利用」するので、ずっと軽いのです。")
    print("       入力が 16x16 ではなく 160x160 なら MLP は 100 倍に膨らみますが、")
    print("       CNN の Conv 部分はパラメータがそのままです (Linear だけが大きくなります)。")

    print("\n[まとめ] 構造読解 = shape の追跡 + パラメータの計算。この 2 つでどんな CNN も見積れる。")
    print("         次のレベル: 空間を要約する専用の部品、プーリング — そして特徴の階層。")


if __name__ == "__main__":
    main()
