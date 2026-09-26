"""
level03 — Building a CNN Architecture

We define a mini CNN with torch (no training yet), and:
  1) dissect Conv2d's weight tensor   2) verify the output-size formula
  3) trace shape changes per layer    4) hand-count parameters vs torch's tally
  5) compare parameter counts with an MLP
"""

import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data


def conv_out(size: int, kernel: int, stride: int = 1, padding: int = 0) -> int:
    """Output-size formula: (in + 2p - k) / s + 1 — the equation you'll use most here."""
    return (size + 2 * padding - kernel) // stride + 1


class TinyCNN(nn.Module):
    """Mini CNN for 16x16 grayscale shapes, 3 classes.
    Design rhythm: grow channels (1->8->16->32), shrink space (16->16->8->4)."""

    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.head = nn.Linear(32 * 4 * 4, n_classes)

    def forward_traced(self, x: torch.Tensor) -> torch.Tensor:
        """Pass the tensor through, printing the shape after every layer."""
        steps = [
            ("input (B,C,H,W)", lambda t: t),
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
    np.random.seed(3)  # fix the seed (reproducibility)

    # ------------------------------------------------------------------
    print("[1] Dissecting a Conv2d layer — 'an inspection team holding several stamps'")
    conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
    w = conv.weight
    print(f"    weight.shape = {tuple(w.shape)}  (8 stamps, each 1 channel x 3x3)")
    print(f"    bias.shape   = {tuple(conv.bias.shape)}  (one bias per stamp)")
    print(f"    parameters in this layer = 8 x (1x3x3) + 8 = {8 * 9 + 8}  "
          f"(torch tally: {count_params(conv)})")

    # ------------------------------------------------------------------
    print("\n[2] Verifying the output-size formula — out = (in + 2p - k)/s + 1")
    print("    with input 16, kernel 3:")
    print(f"      {'stride':>7} {'padding':>8} {'formula':>8} {'actual':>7}")
    x16 = torch.zeros(1, 1, 16, 16)
    for s, p in [(1, 0), (1, 1), (2, 1), (2, 0)]:
        c = nn.Conv2d(1, 1, 3, stride=s, padding=p)
        real = c(x16).shape[-1]
        pred = conv_out(16, 3, s, p)
        print(f"      {s:>7} {p:>8} {pred:>8} {real:>7}")
    print("    -> Formula and measurement always agree. Trace your layer designs with this equation.")

    # ------------------------------------------------------------------
    print("\n[3] Tracing shapes as a batch of shapes passes the mini CNN")
    X, y = hjh_data.shape_images(n=32, size=16, seed=13)
    batch = torch.from_numpy(X).unsqueeze(1)           # (32,16,16) -> (32,1,16,16) B-C-H-W
    print(f"    numpy {X.shape} -> torch {tuple(batch.shape)}  (batch, channels, height, width)")
    model = TinyCNN()
    with torch.no_grad():
        scores = model.forward_traced(batch)
    print(f"    output = 3-class scores for each of the 32 images. e.g. first image {scores[0].numpy().round(2)}")
    print("    (weights are still random — the scores mean nothing yet; we're checking structure only)")

    # ------------------------------------------------------------------
    print("\n[4] Parameters per layer — hand count vs torch tally")
    rows = [
        ("conv1", "8 x (1x3x3) + 8", 8 * 1 * 9 + 8, model.conv1),
        ("conv2", "16 x (8x3x3) + 16", 16 * 8 * 9 + 16, model.conv2),
        ("conv3", "32 x (16x3x3) + 32", 32 * 16 * 9 + 32, model.conv3),
        ("head", "3 x 512 + 3", 3 * 512 + 3, model.head),
    ]
    print(f"      {'layer':<7} {'formula':<20} {'by hand':>8} {'torch':>8}")
    for name, formula, hand, mod in rows:
        print(f"      {name:<7} {formula:<20} {hand:>8,} {count_params(mod):>8,}")
    total = count_params(model)
    print(f"      {'total':<28} {sum(r[2] for r in rows):>8,} {total:>8,}")
    print("    -> Conv parameters are independent of image size; Linear scales with the Flatten size.")

    # ------------------------------------------------------------------
    print("\n[5] What if an MLP processed the same input? — the power of weight sharing")
    mlp = nn.Sequential(nn.Flatten(), nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))
    print(f"    MLP  (256->128->64->3)  parameters: {count_params(mlp):>8,}")
    print(f"    CNN  (TinyCNN)          parameters: {total:>8,}")
    print("    -> The CNN 'reuses the same stamp across the whole image', so it is far lighter.")
    print("       Make the input 160x160 instead of 16x16 and the MLP grows 100x,")
    print("       while the CNN's Conv part keeps the same parameters (only the Linear grows).")

    print("\n[Recap] Reading an architecture = tracing shapes + counting parameters. With these two, any CNN can be priced.")
    print("        Next level: the dedicated component that summarizes space, pooling — and the hierarchy of features.")


if __name__ == "__main__":
    main()
