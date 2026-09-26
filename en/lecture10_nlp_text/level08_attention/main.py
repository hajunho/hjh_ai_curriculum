"""
level08 — The Attention Mechanism

Implements scaled dot-product attention in plain numpy.
A toy example exposes every intermediate value, then we compute self-attention
for two sentences containing the ambiguous word 'bank' (money/river) and save
a heatmap PNG showing "the same word attending to different places depending
on the sentence".
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SEED = 7


def set_korean_font() -> None:
    """The Korean edition needed a CJK font here; English needs only the minus fix."""
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# [1] Implementing attention — the core is 5 lines
# ---------------------------------------------------------------------------

def softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax. Subtract the max to prevent overflow."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    """Scaled dot-product attention. Returns: (output, weight matrix)."""
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)     # 1) query-key relevance  2) scaling
    weights = softmax(scores)           # 3) the attention budget (rows sum to 1)
    return weights @ V, weights         # 4) weighted average of the values


# ---------------------------------------------------------------------------
# [2] Toy example — every number checked by eye
# ---------------------------------------------------------------------------

def demo_toy() -> None:
    print("[1]-[2] Attention in 5 lines + a 3-token toy example")
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(3, 4))                    # 3 tokens, 4-dim embeddings
    Wq, Wk, Wv = (rng.normal(size=(4, 4)) for _ in range(3))
    Q, K, V = X @ Wq, X @ Wk, X @ Wv               # Q/K/V are each one matrix multiply of X
    scores = Q @ K.T
    scaled = scores / np.sqrt(K.shape[-1])
    weights = softmax(scaled)
    out = weights @ V
    np.set_printoptions(precision=2, suppress=True)
    print(f"    scores QK^T:\n{scores}")
    print(f"    after scaling (divided by sqrt(4)):\n{scaled}")
    print(f"    softmax weights (row sums = {weights.sum(axis=1)}):\n{weights}")
    print(f"    output (weights @ V) shape: {out.shape}")
    print("    -> 'Relevance -> probabilities -> weighted average' — that's all of it.\n")


# ---------------------------------------------------------------------------
# [3] Self-attention over the ambiguous 'bank' sentences
#     Instead of trained embeddings, we use handmade embeddings with planted
#     topic components. 'bank' is designed as an ambiguous word carrying
#     half a money component and half a river component.
# ---------------------------------------------------------------------------

SENT_EYE = ["bank", "approved", "my", "loan"]
SENT_SNOW = ["bank", "of", "the", "river", "flooded"]


def build_embeddings() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    dim = 16
    eye_topic = rng.normal(size=dim)               # the 'money/finance' topic axis
    snow_topic = rng.normal(size=dim)              # the 'river/nature' topic axis
    def vec(topic_mix, noise_scale=0.45):
        return topic_mix + rng.normal(size=dim) * noise_scale
    return {
        "bank": vec(0.5 * eye_topic + 0.5 * snow_topic),   # ambiguous: half each topic
        "approved": vec(eye_topic), "loan": vec(eye_topic),
        "my": vec(np.zeros(dim), 0.8),
        "river": vec(snow_topic), "flooded": vec(snow_topic),
        "of": vec(np.zeros(dim), 0.8), "the": vec(np.zeros(dim), 0.8),
    }


def self_attention_of(tokens: list[str], emb: dict) -> np.ndarray:
    """The simplest self-attention with Q=K=V=X, to keep interpretation clear."""
    X = np.stack([emb[t] for t in tokens])
    _, weights = attention(X, X, X)
    return weights


def demo_ambiguity(emb: dict) -> tuple[np.ndarray, np.ndarray]:
    print("[3] The same 'bank' attends to different places per sentence (self-attention)")
    results = []
    for tokens in (SENT_EYE, SENT_SNOW):
        W = self_attention_of(tokens, emb)
        results.append(W)
        row = W[0]                                  # the attention of the 'bank' row
        pairs = ", ".join(f"{t}({w:.2f})" for t, w in zip(tokens, row))
        print(f"    sentence: {' '.join(tokens)}")
        print(f"      attention of 'bank': {pairs}")
    print("    (Note: the large self-attention comes from setting Q=K, which makes")
    print("     self-similarity maximal. Real models learn W_Q, W_K to adjust this.)")
    print("    -> In the loan sentence, 'bank' attends to approved/loan; in the river")
    print("       sentence, to river/flooded. The representation of 'bank' is rebuilt")
    print("       from its context words — this is how attention handles the")
    print("       ambiguity problem from level00.\n")
    return results[0], results[1]


def save_heatmaps(W_eye: np.ndarray, W_snow: np.ndarray) -> None:
    print("[4] Saving the attention-weight heatmaps")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, W, tokens, title in (
        (axes[0], W_eye, SENT_EYE, "money: bank approved my loan"),
        (axes[1], W_snow, SENT_SNOW, "river: bank of the river flooded"),
    ):
        im = ax.imshow(W, cmap="Blues", vmin=0, vmax=W.max())
        ax.set_xticks(range(len(tokens)), tokens, rotation=45)
        ax.set_yticks(range(len(tokens)), tokens)
        ax.set_title(title, fontsize=10)
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                ax.text(j, i, f"{W[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if W[i, j] < 0.5 * W.max() else "white")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Self-attention weights (row = attending token)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "attention_heatmap.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    saved: {path}")
    print("    How to read: row = attending token, column = attended token, rows sum to 1.\n")


def demo_cost() -> None:
    print("[5] The cost of attention — every token looks at every token (n^2)")
    for n in (10, 100, 1000, 10000):
        cells = n * n
        mb = cells * 4 / 1e6                        # float32
        print(f"    {n:6,d} tokens -> weight matrix {cells:12,d} cells (~{mb:8.1f} MB/layer/head)")
    print("    -> The root cause of LLM context-length limits and long-input pricing.")
    print("\nBottom line: one line — 'weighted average by relevance' — is the heart of modern AI.")


if __name__ == "__main__":
    print("=" * 70)
    print("Attention — giving each speaker in the meeting a different share of your ear")
    print("=" * 70 + "\n")
    np.random.seed(SEED)
    set_korean_font()
    demo_toy()
    emb = build_embeddings()
    W_eye, W_snow = demo_ambiguity(emb)
    save_heatmaps(W_eye, W_snow)
    demo_cost()
