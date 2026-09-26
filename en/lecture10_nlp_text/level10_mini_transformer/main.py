"""
level10 — Implementing a Mini Transformer from Scratch

Builds a small GPT (decoder-only transformer) in torch and trains
next-character prediction on 30,000 characters of tiny_corpus.
Demonstrates the falling loss and changing generations, perplexity,
the causal structure of the attention, and saves a loss-curve PNG.
Finishes within 60 seconds on CPU.
"""

import os
import sys
import time
import math
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SEED = 42
BLOCK = 32          # context length (characters the model sees at once)
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
BATCH = 48
STEPS = 400
LR = 3e-3


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask (the torch version of level09's part)."""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        mask = torch.triu(torch.ones(BLOCK, BLOCK), diagonal=1).bool()
        self.register_buffer("mask", mask)
        self.last_weights = None                     # for the demo: store attention

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(D_MODEL, dim=2)
        def heads(t):                                # (B, T, C) -> (B, H, T, C/H)
            return t.view(B, T, N_HEADS, C // N_HEADS).transpose(1, 2)
        q, k, v = heads(q), heads(k), heads(v)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // N_HEADS)
        att = att.masked_fill(self.mask[:T, :T], float("-inf"))   # no cheating off the future
        att = F.softmax(att, dim=-1)
        self.last_weights = att.detach()
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    """Attention -> residual+norm -> FF -> residual+norm (Pre-LN style)."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(),
            nn.Linear(4 * D_MODEL, D_MODEL))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, D_MODEL)
        self.pos_embed = nn.Embedding(BLOCK, D_MODEL)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYERS)])
        self.ln_final = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_embed(idx) + self.pos_embed(pos)   # tokens + seat-number tags
        x = self.blocks(x)
        return self.head(self.ln_final(x))              # (B, T, vocab)


def get_batch(ids: torch.Tensor, generator: torch.Generator):
    """Cut BATCH random windows from the corpus into (input, one-char-shifted target) pairs."""
    starts = torch.randint(0, len(ids) - BLOCK - 1, (BATCH,), generator=generator)
    x = torch.stack([ids[s: s + BLOCK] for s in starts])
    y = torch.stack([ids[s + 1: s + BLOCK + 1] for s in starts])
    return x, y


@torch.no_grad()
def generate(model, stoi, itos, prompt: str, length: int = 60,
             temperature: float = 0.7) -> str:
    model.eval()
    ids = torch.tensor([[stoi.get(ch, 0) for ch in prompt]], dtype=torch.long)
    for _ in range(length):
        logits = model(ids[:, -BLOCK:])                 # sees only the latest BLOCK chars
        probs = F.softmax(logits[0, -1] / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=None)
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    model.train()
    return "".join(itos[int(i)] for i in ids[0])


if __name__ == "__main__":
    t0 = time.time()
    torch.manual_seed(SEED)
    print("=" * 70)
    print("A mini transformer (GPT architecture) — assemble, start the engine, road test")
    print("=" * 70 + "\n")

    # [1] Data -----------------------------------------------------------
    text = hjh_data.tiny_corpus()[:30000]
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    ids = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    print(f"[1] Data: {len(text):,} chars, character vocabulary {len(chars)} types")
    print(f"    The task: in a length-{BLOCK} sequence, predict 'the next char' at every position at once")
    print(f"    (one sequence = {BLOCK} problems — the power of parallel transformer training)\n")

    # [2] Model -------------------------------------------------------------
    model = MiniGPT(len(chars))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[2] Model: {D_MODEL}-dim, {N_LAYERS}-layer, {N_HEADS}-head decoder")
    print(f"    {n_params:,} parameters — GPT's architecture, a toy's size\n")

    # [3] Training -------------------------------------------------------------
    print(f"[3] Training: {STEPS} steps (watch the loss and the generations)")
    prompt = "Yesterday "
    print(f"    step   0 | loss  -   | generation: {generate(model, stoi, itos, prompt, 40)!r}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(SEED)
    losses = []
    for step in range(1, STEPS + 1):
        x, y = get_batch(ids, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(chars)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
        if step in (50, 150, STEPS):
            ppl = math.exp(losses[-1])
            print(f"    step {step:3d} | loss {losses[-1]:.3f} (perplexity {ppl:5.1f}) "
                  f"| generation: {generate(model, stoi, itos, prompt, 40)!r}")
    print(f"    -> The loss starts near ln({len(chars)})={math.log(len(chars)):.2f} (pure guessing) and,")
    print(f"       as it falls, the model gets less 'surprised' by the next character.\n")

    # save the loss curve
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, linewidth=1, color="#3b6db4")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Mini transformer training loss")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "loss_curve.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    loss curve saved: {path}\n")

    # [4] Generation test --------------------------------------------------------
    print("[4] Generation test — continue from a prompt")
    for p in ["Yesterday a student ", "Over the weekend a chef ", "This morning "]:
        print(f"    {p!r} -> {generate(model, stoi, itos, p, 50)!r}")
    print()

    # [5] Peeking at the attention -------------------------------------------------
    print("[5] Checking the causal structure of the learned attention (layer 1 head 0, top-left 5x5)")
    _ = model(ids[:BLOCK].unsqueeze(0))
    att = model.blocks[0].attn.last_weights[0, 0, :5, :5]
    for row in att:
        print("    " + " ".join(f"{v:.2f}" for v in row))
    print("    -> The upper right (the future) is all zeros — the causal mask holds.")
    print(f"\nTotal runtime: {time.time() - t0:.1f}s (within the 90s budget)")
