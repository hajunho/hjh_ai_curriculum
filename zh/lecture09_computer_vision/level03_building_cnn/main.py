"""
level03 — 搭建 CNN 结构

用 torch 定义一个迷你 CNN (还不做训练)，
  1) 解剖 Conv2d 的权重张量  2) 验算输出尺寸公式
  3) 追踪各层 shape 变化      4) 参数量手算 vs torch 统计
  5) 与 MLP 的参数量对比
"""

import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data


def conv_out(size: int, kernel: int, stride: int = 1, padding: int = 0) -> int:
    """输出尺寸公式: (in + 2p - k) / s + 1 — 本课程用得最多的一个式子。"""
    return (size + 2 * padding - kernel) // stride + 1


class TinyCNN(nn.Module):
    """给 16x16 黑白图形 3 分类用的迷你 CNN。
    设计节奏: 通道加多(1->8->16->32), 空间缩小(16->16->8->4)。"""

    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.head = nn.Linear(32 * 4 * 4, n_classes)

    def forward_traced(self, x: torch.Tensor) -> torch.Tensor:
        """每过一层就打印一次 shape, 让数据走完全程。"""
        steps = [
            ("输入 (B,C,H,W)", lambda t: t),
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
    np.random.seed(3)  # 固定 seed(可复现)

    # ------------------------------------------------------------------
    print("[1] 解剖 Conv2d 层 — '拿着好几个印章的审核小组'")
    conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
    w = conv.weight
    print(f"    weight.shape = {tuple(w.shape)}  (8 个印章, 每个 1 通道 x 3x3)")
    print(f"    bias.shape   = {tuple(conv.bias.shape)}  (每个印章 1 个偏置)")
    print(f"    这一层的参数 = 8 x (1x3x3) + 8 = {8 * 9 + 8} 个  "
          f"(torch 统计: {count_params(conv)} 个)")

    # ------------------------------------------------------------------
    print("\n[2] 验算输出尺寸公式 — out = (in + 2p - k)/s + 1")
    print("    输入 16, 核 3 时:")
    print(f"      {'stride':>7} {'padding':>8} {'公式':>6} {'实测':>6}")
    x16 = torch.zeros(1, 1, 16, 16)
    for s, p in [(1, 0), (1, 1), (2, 1), (2, 0)]:
        c = nn.Conv2d(1, 1, 3, stride=s, padding=p)
        real = c(x16).shape[-1]
        pred = conv_out(16, 3, s, p)
        print(f"      {s:>7} {p:>8} {pred:>6} {real:>6}")
    print("    -> 公式和实测总是一致。设计层的时候就用这个式子一路追到底。")

    # ------------------------------------------------------------------
    print("\n[3] 把一批图形送进迷你 CNN, 追踪 shape")
    X, y = hjh_data.shape_images(n=32, size=16, seed=13)
    batch = torch.from_numpy(X).unsqueeze(1)           # (32,16,16) -> (32,1,16,16) B-C-H-W
    print(f"    numpy {X.shape} -> torch {tuple(batch.shape)}  (批, 通道, 高, 宽)")
    model = TinyCNN()
    with torch.no_grad():
        scores = model.forward_traced(batch)
    print(f"    输出 = 32 张图像各自的 3 类别得分。例: 第一张 {scores[0].numpy().round(2)}")
    print("    (还没训练, 权重是随机的, 所以得分没有意义 — 只看结构)")

    # ------------------------------------------------------------------
    print("\n[4] 各层参数量 — 手算 vs torch 统计")
    rows = [
        ("conv1", "8 x (1x3x3) + 8", 8 * 1 * 9 + 8, model.conv1),
        ("conv2", "16 x (8x3x3) + 16", 16 * 8 * 9 + 16, model.conv2),
        ("conv3", "32 x (16x3x3) + 32", 32 * 16 * 9 + 32, model.conv3),
        ("head", "3 x 512 + 3", 3 * 512 + 3, model.head),
    ]
    print(f"      {'层':<7} {'公式':<20} {'手算':>8} {'torch':>8}")
    for name, formula, hand, mod in rows:
        print(f"      {name:<7} {formula:<20} {hand:>8,} {count_params(mod):>8,}")
    total = count_params(model)
    print(f"      {'合计':<28} {sum(r[2] for r in rows):>8,} {total:>8,}")
    print("    -> Conv 的参数与图像大小无关, Linear 则与 Flatten 后的尺寸成正比。")

    # ------------------------------------------------------------------
    print("\n[5] 同样的输入若用 MLP 处理呢? — 权重共享的威力")
    mlp = nn.Sequential(nn.Flatten(), nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))
    print(f"    MLP  (256->128->64->3)  参数: {count_params(mlp):>8,} 个")
    print(f"    CNN  (TinyCNN)          参数: {total:>8,} 个")
    print("    -> CNN 是'同一个印章在整张图像上反复用', 所以轻得多。")
    print("       输入若不是 16x16 而是 160x160, MLP 会膨胀 100 倍,")
    print("       而 CNN 的 Conv 部分参数一点不变 (只有 Linear 会变大)。")

    print("\n[小结] 读结构 = 追踪 shape + 计算参数。有这两样, 任何 CNN 都能报出价来。")
    print("       下一关: 专门负责汇总空间的零件 —— 池化, 以及特征的层级。")


if __name__ == "__main__":
    main()
