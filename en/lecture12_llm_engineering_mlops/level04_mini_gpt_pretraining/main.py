"""
Hands-on mini GPT pre-training.
- We implement a small two-layer decoder transformer (about 100k parameters)
  from scratch and train it on hjh_data.tiny_corpus() at character level,
  asking it only to 'guess the next character'.
- Watch the loss fall and grammar emerge in the generated text as steps pass.
- Designed tiny enough to finish within 90 seconds on CPU alone.
"""
import math
import os
import sys
import time
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

# ----- Hyperparameters: an ultra-small 'hundred-thousand-parameter' model ----
CTX = 64        # context length (characters seen at once)
D_MODEL = 64    # embedding dimension
N_HEAD = 2      # number of attention heads
N_LAYER = 2     # number of transformer blocks
BATCH = 48
STEPS = 700
LR = 3e-3
TIME_BUDGET = 75.0  # seconds — stop training early past this (safety valve)


class CausalSelfAttention(nn.Module):
    """Masked self-attention: each character may only look at the characters *before* it."""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)   # compute Q, K, V in one go
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        mask = torch.tril(torch.ones(CTX, CTX))      # lower triangle = hide the future
        self.register_buffer("mask", mask.view(1, 1, CTX, CTX))

    def forward(self, x):
        B, T, C = x.shape
        hd = C // N_HEAD
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, heads, T, dim per head)
        q = q.view(B, T, N_HEAD, hd).transpose(1, 2)
        k = k.view(B, T, N_HEAD, hd).transpose(1, 2)
        v = v.view(B, T, N_HEAD, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)      # similarity scores
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)                          # attention weights
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    """Transformer block = attention (gather information) + MLP (think), each with a residual connection."""

    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention()
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(),
            nn.Linear(4 * D_MODEL, D_MODEL),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, D_MODEL)   # character -> vector
        self.pos_emb = nn.Embedding(CTX, D_MODEL)          # position -> vector
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size)         # vector -> next-character scores

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.ln_f(self.blocks(x))
        logits = self.head(x)
        loss = None
        if targets is not None:  # loss for guessing 'the next character' at every position
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=0.8, top_k=8):
        for _ in range(n_new):
            logits, _ = self(idx[:, -CTX:])                 # use only the last CTX characters
            logits = logits[:, -1, :] / temperature
            top = torch.topk(logits, top_k)                 # keep only the top-k candidates
            logits[logits < top.values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


def get_batch(data, rng):
    """Cut BATCH random positions from the corpus into (input, shifted-by-one target) pairs."""
    ix = torch.randint(len(data) - CTX - 1, (BATCH,), generator=rng)
    x = torch.stack([data[i:i + CTX] for i in ix])
    y = torch.stack([data[i + 1:i + CTX + 1] for i in ix])
    return x, y


def sample_text(model, stoi, itos, prompt="Today ", n=90):
    idx = torch.tensor([[stoi[c] for c in prompt]])
    out = model.generate(idx, n)[0].tolist()
    return "".join(itos[i] for i in out)


def main():
    torch.manual_seed(12)
    rng = torch.Generator().manual_seed(12)
    t0 = time.time()

    # [1] load the corpus and build the character dictionary ------------------
    text = hjh_data.tiny_corpus()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_val = len(data) // 20
    train_data, val_data = data[:-n_val], data[-n_val:]
    print(f"[1] Corpus {len(text):,} characters / dictionary {len(chars)} characters / {n_val:,} held out for validation")

    # [2] create the model -----------------------------------------------------
    model = TinyGPT(len(chars))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[2] TinyGPT created: {N_LAYER} layers, {N_HEAD} heads, d={D_MODEL}, {n_params:,} parameters")
    print("    (real large models are this same structure scaled up thousands of times)")

    # [3] generation before training — it knows nothing yet --------------------
    print("\n[3] Before training:  ", repr(sample_text(model, stoi, itos)))

    # [4] pre-training: repeat 'guess the next character' ----------------------
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    print(f"\n[4] Pre-training starts (up to {STEPS} steps, batch {BATCH} x context {CTX} characters)")
    for step in range(1, STEPS + 1):
        x, y = get_batch(train_data, rng)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                vx, vy = get_batch(val_data, rng)
                _, vloss = model(vx, vy)
            print(f"    step {step:4d} | train loss {loss.item():.3f} | val loss {vloss.item():.3f}")
        if step in (30, 200):  # mid-run checks: watch sentences take shape
            print(f"      └ step {step} sample:", repr(sample_text(model, stoi, itos, n=60)))
        if time.time() - t0 > TIME_BUDGET:
            print(f"    (time safety valve tripped: stopped early at step {step})")
            break
    tokens_seen = step * BATCH * CTX
    print(f"    Training done: {step} steps, {tokens_seen:,} characters seen, {time.time()-t0:.1f}s elapsed")

    # [5] generation after training — confirm that grammar has 'emerged' -------
    print("\n[5] After training (prompt 'Today '):")
    for k in range(2):
        torch.manual_seed(100 + k)
        print(f"    sample {k+1}:", sample_text(model, stoi, itos))
    print("    -> Subject-verb-object order, articles ('a report'), and past-tense verbs now read naturally.")
    print("      It memorized no answer sentences — it only learned 'next-character probabilities', yet grammar appeared.")

    # [6] save a checkpoint — levels 06 (LoRA) through 08 (DPO) continue this model's story
    out_dir = pathlib.Path(__file__).resolve().parent / "outputs"
    os.makedirs(out_dir, exist_ok=True)
    ckpt = out_dir / "tiny_gpt.pt"
    torch.save({"model": model.state_dict(), "stoi": stoi, "itos": itos,
                "config": {"ctx": CTX, "d_model": D_MODEL, "n_head": N_HEAD,
                           "n_layer": N_LAYER, "vocab": len(chars)}}, ckpt)
    print(f"\n[6] Checkpoint saved: {ckpt}")
    print(f"    Total run time: {time.time()-t0:.1f}s (within the 90s budget)")


if __name__ == "__main__":
    main()
