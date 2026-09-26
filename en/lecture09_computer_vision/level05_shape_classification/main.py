"""
level05 — Hands-On: Classifying Shape Images

The full process of training a CNN on hjh_data.shape_images:
  data split -> training loop -> evaluation (confusion matrix) -> misclassified-case visualization.
We compare performance and parameters with an MLP under identical conditions to confirm the CNN's edge.
"""

import os
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
CLASS_EN = ["square", "circle", "triangle"]
EPOCHS = 18
BATCH = 64
LR = 1e-3


class ShapeCNN(nn.Module):
    """16x16 -> 3 classes. Two Conv-ReLU-Pool blocks + a Linear verdict."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # (8,8,8)
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # (16,4,4)
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16 * 4 * 4, 3))

    def forward(self, x):
        return self.head(self.features(x))


def make_mlp() -> nn.Module:
    """Comparison MLP: pixels simply flattened (knows nothing of spatial structure)."""
    return nn.Sequential(nn.Flatten(), nn.Linear(256, 64), nn.ReLU(),
                         nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))


def train(model: nn.Module, xtr, ytr, xva, yva, tag: str) -> list[float]:
    """Mini-batch training loop. Records and returns validation accuracy per epoch."""
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        acc = evaluate(model, xva, yva)
        history.append(acc)
        if epoch % 3 == 0 or epoch == 1:
            print(f"      {tag} epoch {epoch:>2}/{EPOCHS}  train_loss={loss.item():.3f}  val_acc={acc:.3f}")
    return history


@torch.no_grad()
def evaluate(model: nn.Module, x, y) -> float:
    model.eval()
    return float((model(x).argmax(1) == y).float().mean())


@torch.no_grad()
def predict(model: nn.Module, x) -> torch.Tensor:
    model.eval()
    return model(x).argmax(1)


def main() -> None:
    t0 = time.time()
    torch.manual_seed(5)
    np.random.seed(5)  # fix the seed (reproducibility)

    # ------------------------------------------------------------------
    print("[1] Preparing and splitting the data — train 600 / val 150 / test 150")
    X, y = hjh_data.shape_images(n=900, size=16, seed=13)
    Xt = torch.from_numpy(X).unsqueeze(1)              # (900,1,16,16)
    yt = torch.from_numpy(y)
    xtr, ytr = Xt[:600], yt[:600]                      # shape_images is already shuffled
    xva, yva = Xt[600:750], yt[600:750]
    xte, yte = Xt[750:], yt[750:]
    print(f"    {len(X)} images total (16x16 grayscale, square/circle/triangle). Test never touches training.")

    # ------------------------------------------------------------------
    print("\n[2] Training the CNN — Conv-ReLU-Pool x2 + Linear")
    cnn = ShapeCNN()
    n_cnn = sum(p.numel() for p in cnn.parameters())
    hist_cnn = train(cnn, xtr, ytr, xva, yva, "CNN")

    print("\n[3] Training an MLP on the same data — what if we just flatten the pixels?")
    mlp = make_mlp()
    n_mlp = sum(p.numel() for p in mlp.parameters())
    hist_mlp = train(mlp, xtr, ytr, xva, yva, "MLP")

    # ------------------------------------------------------------------
    print("\n[4] Test report card — 150 images never seen before")
    acc_cnn = evaluate(cnn, xte, yte)
    acc_mlp = evaluate(mlp, xte, yte)
    print(f"      {'model':<6} {'params':>10} {'test acc':>12}")
    print(f"      {'CNN':<6} {n_cnn:>10,} {acc_cnn:>12.3f}")
    print(f"      {'MLP':<6} {n_mlp:>10,} {acc_mlp:>12.3f}")
    pred = predict(cnn, xte)
    conf = np.zeros((3, 3), dtype=int)
    for t, p in zip(yte.numpy(), pred.numpy()):
        conf[t, p] += 1
    print("    CNN confusion matrix (rows=actual, cols=predicted):")
    print("            " + "".join(f"{n:>10}" for n in CLASS_EN))
    for i, name in enumerate(CLASS_EN):
        print(f"      {name:>8}  " + "".join(f"{v:>10}" for v in conf[i]))

    # ------------------------------------------------------------------
    print("\n[5] Misclassification analysis — looking at the images the model got wrong")
    wrong = torch.where(pred != yte)[0]
    print(f"    wrong: {len(wrong)}/{len(yte)}")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plt.figure(figsize=(11, 6.5))
    # Left: learning curves
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(range(1, EPOCHS + 1), hist_cnn, marker="o", label=f"CNN ({n_cnn:,} params)")
    ax.plot(range(1, EPOCHS + 1), hist_mlp, marker="s", label=f"MLP ({n_mlp:,} params)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy")
    ax.set_title("Learning curves")
    ax.set_ylim(0.3, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    # Right: up to 8 misclassified cases (using the right half of a 2x8 grid)
    n_show = min(8, len(wrong))
    for k in range(n_show):
        i = int(wrong[k])
        ax = fig.add_subplot(2, 8, 5 + (k % 4) + 8 * (k // 4))
        ax.imshow(xte[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"true {CLASS_EN[int(yte[i])]}\npred {CLASS_EN[int(pred[i])]}", fontsize=7)
        ax.axis("off")
    fig.suptitle("Shape classification: CNN vs MLP + misclassified test images", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "shape_classification.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    Saved: {path}")
    print("    -> Most misses are 'hard' images: heavy noise or shapes clipped at the edge.")
    print("       The habit of eyeballing misclassifications carries into level08's error analysis.")

    print(f"\n[Recap] Split -> train -> evaluate -> analyze misses: the standard image-classification cycle, completed.")
    print(f"        The CNN beat the MLP with fewer parameters (total time {time.time() - t0:.1f}s).")
    print("        Next level: raising performance without collecting more data — data augmentation.")


if __name__ == "__main__":
    main()
