"""
Hands-on LoRA (Low-Rank Adaptation) fine-tuning, implemented from scratch.
We briefly pre-train a mini GPT on the English corpus, then adapt it to a
new speaking style (an archaic 'hath made' register). Instead of retraining
all the weights, we attach low-rank adapters (W + scale·B@A) to the attention
linear layers and train only those.
We compare how far the trainable parameter count falls versus full fine-tuning.
"""
import pathlib
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

SEED = 42
CTX, D, H, LAYERS = 64, 64, 2, 2


class CausalSelfAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3 * d)   # LoRA attachment point 1
        self.proj = nn.Linear(d, d)      # LoRA attachment point 2

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        s = lambda t: t.view(B, T, self.h, C // self.h).transpose(1, 2)
        y = F.scaled_dot_product_attention(s(q), s(k), s(v), is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, h)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, D)
        self.pos = nn.Embedding(CTX, D)
        self.blocks = nn.ModuleList(Block(D, H) for _ in range(LAYERS))
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab, bias=False)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T))
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


class LoRALinear(nn.Module):
    """Freeze the existing linear layer W and train only the low-rank correction scale·B@A."""

    def __init__(self, base, r=4, alpha=8):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        # A starts as small random values, B as zeros -> right after attachment
        # the output is exactly identical to the original
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.02)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)


def make_style_corpus(n=900, seed=7):
    # New-domain data with the same sentence frames as the pre-training corpus,
    # but the verbs in an archaic 'hath ...' register
    rng = random.Random(seed)
    times = ["Yesterday", "Today", "This morning", "This evening", "Over the weekend"]
    subs = ["a student", "an office worker", "a chef", "a developer", "a teacher"]
    verbs = ["hath made", "hath fixed", "hath checked", "hath organized", "hath prepared"]
    objs = ["a report", "a pot of stew", "a program", "a letter", "a proposal"]
    return " ".join(f"{rng.choice(times)} {rng.choice(subs)} "
                    f"{rng.choice(verbs)} {rng.choice(objs)}." for _ in range(n))


def encode(text, stoi):
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def get_batch(data, bs, gen):
    ix = torch.randint(len(data) - CTX - 1, (bs,), generator=gen)
    x = torch.stack([data[i:i + CTX] for i in ix])
    y = torch.stack([data[i + 1:i + CTX + 1] for i in ix])
    return x, y


def train(model, data, steps, lr, gen, tag):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    for step in range(1, steps + 1):
        x, y = get_batch(data, 24, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % (steps // 4) == 0:
            print(f"    {tag} step {step:4d}/{steps}  loss {loss.item():.3f}")


@torch.no_grad()
def eval_loss(model, data, gen, iters=20):
    total = 0.0
    for _ in range(iters):
        x, y = get_batch(data, 24, gen)
        logits = model(x)
        total += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    return total / iters


@torch.no_grad()
def sample(model, stoi, itos, prompt, n=48, gen=None):
    idx = encode(prompt, stoi)[None]
    for _ in range(n):
        logits = model(idx[:, -CTX:])[:, -1] / 0.7
        nxt = torch.multinomial(logits.softmax(-1), 1, generator=gen)
        idx = torch.cat([idx, nxt], 1)
    return "".join(itos[i] for i in idx[0].tolist())


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    print("=" * 62)
    print("Level 06 — Fine-tuning and LoRA (adapters implemented by hand)")
    print("=" * 62)

    # [1] data: pre-training corpus (plain past tense) + new-style corpus ('hath' register)
    base_text = hjh_data.tiny_corpus()
    style_text = make_style_corpus()
    chars = sorted(set(base_text + style_text))
    stoi = {c: i for i, c in enumerate(chars)}
    base_data, style_data = encode(base_text, stoi), encode(style_text, stoi)
    print(f"[1] Corpora: pre-training {len(base_data):,} chars / new style {len(style_data):,} chars / vocab {len(chars)}")

    # [2] pre-train the base model — it will know only the plain 'made/fixed' register
    model = TinyGPT(len(chars))
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[2] Pre-training the base model ({n_total:,} parameters)")
    train(model, base_data, steps=400, lr=3e-3, gen=gen, tag="pretrain")
    print(f"    Sample: {sample(model, stoi, chars, 'This evening a chef ', gen=gen)!r}")

    # [3] performance on the new-style data before fine-tuning
    before = eval_loss(model, style_data, gen)
    print(f"[3] New-style loss before fine-tuning = {before:.3f} (it does not know the 'hath' register)")

    # [4] freeze the whole base and attach LoRA only to the attention layers
    for p in model.parameters():
        p.requires_grad_(False)
    for blk in model.blocks:
        blk.attn.qkv = LoRALinear(blk.attn.qkv, r=4, alpha=8)
        blk.attn.proj = LoRALinear(blk.attn.proj, r=4, alpha=8)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[4] Full fine-tuning would retrain all {n_total:,} parameters, but")
    print(f"    LoRA (r=4) trains only {n_train:,} -> {100 * n_train / n_total:.2f}% of the total")

    # [5] LoRA fine-tuning — only the adapters (A, B) get updated
    print("[5] LoRA fine-tuning (not one bit of the base weights changes)")
    train(model, style_data, steps=300, lr=3e-3, gen=gen, tag="LoRA")
    after = eval_loss(model, style_data, gen)
    print(f"    New-style loss {before:.3f} -> {after:.3f}")
    print(f"    Sample: {sample(model, stoi, chars, 'This evening a chef ', gen=gen)!r}")

    # [6] merging: fold into W' = W + scale·B@A and inference costs nothing extra
    lora = model.blocks[0].attn.qkv
    merged = nn.Linear(lora.base.in_features, lora.base.out_features)
    with torch.no_grad():
        merged.weight.copy_(lora.base.weight + lora.scale * (lora.B @ lora.A))
        merged.bias.copy_(lora.base.bias)
        x = torch.randn(2, 5, D, generator=gen)
        same = torch.allclose(lora(x), merged(x), atol=1e-5)
    print(f"[6] Merge check: adapter output == merged-weight output ? {same}")
    print("    For deployment, merge into a single file; for experiments, just swap adapters.")


if __name__ == "__main__":
    main()
