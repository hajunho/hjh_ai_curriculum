"""
level11 — Vision Transformers (ViT) and Multimodal Models

We implement, by hand, the process of reading an image as a sentence of 'patch words'.
  1) slice a 16x16 image into sixteen 4x4 patches (tokenization)
  2) patch embedding + positional encoding
  3) self-attention implemented directly as matrix operations (Q,K,V)
  4) train a mini ViT on shape classification and visualize 'where it looks' (attention maps)
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
PATCH = 4              # patch side (16x16 -> sixteen 4x4 patches)
N_PATCH = (16 // PATCH) ** 2
DIM = 32               # embedding dimension
EPOCHS = 80


def patchify(img: np.ndarray) -> np.ndarray:
    """Slice the image into patch words: (16,16) -> (16, 4*4)."""
    h = img.shape[0] // PATCH
    out = img.reshape(h, PATCH, h, PATCH).transpose(0, 2, 1, 3)
    return out.reshape(N_PATCH, PATCH * PATCH)


class MiniViT(nn.Module):
    """Toy ViT: patch embedding -> 1 self-attention layer -> mean pool -> classify.
    Attention implemented directly with matrix ops, no nn.MultiheadAttention."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(PATCH * PATCH, DIM)         # patch -> embedding vector
        self.pos = nn.Parameter(torch.zeros(N_PATCH, DIM))  # positional encoding (learnable)
        self.wq = nn.Linear(DIM, DIM, bias=False)           # Query: 'what I'm looking for'
        self.wk = nn.Linear(DIM, DIM, bias=False)           # Key:   'what I have'
        self.wv = nn.Linear(DIM, DIM, bias=False)           # Value: 'what I'll pass along'
        self.ff = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU())
        self.head = nn.Linear(DIM, 3)

    def forward(self, x, return_attn: bool = False):
        # x: (B, 16 patches, 16 pixel values) — a sentence of patch words
        t = self.embed(x) + self.pos                        # word embedding + position
        q, k, v = self.wq(t), self.wk(t), self.wv(t)
        scores = q @ k.transpose(-2, -1) / (DIM ** 0.5)     # relevance of every word pair
        attn = torch.softmax(scores, dim=-1)                # each row sums to 1 (attention allocation)
        t = t + attn @ v                                    # mix in information by relevance
        t = t + self.ff(t)
        logits = self.head(t.mean(dim=1))                   # summarize the sentence -> 3 classes
        return (logits, attn) if return_attn else logits


def to_tokens(X: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.stack([patchify(im) for im in X]))


def main() -> None:
    t0 = time.time()
    torch.manual_seed(11)
    np.random.seed(11)  # fix the seed (reproducibility)

    # ------------------------------------------------------------------
    print("[1] Tokenizing an image into 'patch words' — preparing to read it like a sentence")
    X, y = hjh_data.shape_images(n=750, size=16, seed=13)
    sample = X[0]
    tokens = patchify(sample)
    print(f"    16x16 image -> {N_PATCH} patches of {PATCH}x{PATCH}, each patch {PATCH * PATCH} numbers")
    print(f"    tokens.shape = {tokens.shape}  (the same structure as a 16-word sentence)")
    bright = tokens.mean(axis=1).round(2)
    print("    Mean brightness per patch (viewed as a 4x4 grid, the shape's rough position survives):")
    for r in range(4):
        print("      " + " ".join(f"{v:.2f}" for v in bright[r * 4:(r + 1) * 4]))

    # ------------------------------------------------------------------
    print("\n[2] Patch embedding + positional encoding")
    print(f"    Linear({PATCH * PATCH} -> {DIM}): one patch becomes a {DIM}-dim vector (a 'word embedding').")
    print("    Shuffling the order would leave the same bag of words, so each patch gets a")
    print("    position vector added, stamping 'which slot this word occupies' (same reason as lecture10's positional encoding).")

    # ------------------------------------------------------------------
    print("\n[3] Self-attention — relevance of every patch pair, all at once")
    print("    scores = Q @ K^T / sqrt(d)  ->  softmax  ->  attn @ V")
    print("    Each patch decides for itself 'which of the other 15 patches to consult, and how much'.")
    print("    If the CNN is 'next door first, step by step', attention is 'an all-hands meeting from layer one'.")

    # ------------------------------------------------------------------
    print("\n[4] Training the mini ViT — does attention actually focus on the shape?")
    xt, yt = to_tokens(X[:600]), torch.from_numpy(y[:600])
    xe, ye = to_tokens(X[600:]), torch.from_numpy(y[600:])
    model = MiniViT()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()
        if epoch % 16 == 0:
            with torch.no_grad():
                acc = float((model(xe).argmax(1) == ye).float().mean())
            print(f"      epoch {epoch:>3}/{EPOCHS}  test_acc={acc:.3f}")
    with torch.no_grad():
        acc = float((model(xe).argmax(1) == ye).float().mean())
    print(f"    Mini ViT ({n_params:,} parameters) test accuracy: {acc:.3f}")
    print("    (comparable to a CNN on this task, but ViT's real strength emerges at scale)")

    # ------------------------------------------------------------------
    print("\n[5] 'Where does it look?' — visualizing the learned attention maps")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for row, c in enumerate(range(3)):
        i = int(np.where(y[600:] == c)[0][0]) + 600
        img = X[i]
        with torch.no_grad():
            _, attn = model(to_tokens(img[None]), return_attn=True)
        attn = attn[0]                                     # (16,16): attention allocation per patch
        # Where did all patches look on average -> 4x4 -> upscale to 16x16
        avg_attn = attn.mean(dim=0).reshape(4, 4).numpy()
        heat = np.kron(avg_attn, np.ones((PATCH, PATCH)))
        axes[row][0].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[row][0].set_title(f"{CLASS_EN[c]} (input)", fontsize=9)
        axes[row][1].imshow(heat, cmap="magma")
        axes[row][1].set_title("mean attention", fontsize=9)
        axes[row][2].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[row][2].imshow(heat, cmap="magma", alpha=0.55)
        axes[row][2].set_title("overlay", fontsize=9)
        for ax in axes[row]:
            ax.axis("off")
        top = int(avg_attn.argmax())
        print(f"    {CLASS_EN[c]:>9}: most-attended patch = ({top // 4},{top % 4}) "
              f"(4x4 grid coordinates — check in the PNG whether that's where the shape sits)")
    fig.suptitle("Mini ViT: where do the patches attend?", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "vit_attention.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    Saved: {path}")
    print("    -> Bright regions = patches the model consulted most. They cluster around the shape.")

    print(f"\n[Recap] ViT = patch tokenization + embedding + self-attention (took {time.time() - t0:.1f}s).")
    print("        Multimodal (CLIP-style) goes one step further: image embeddings and text embeddings")
    print("        share one space, trained so that 'close means same meaning' (see the README).")
    print("        Congratulations on completing lecture09 — from pixels all the way to transformers.")


if __name__ == "__main__":
    main()
