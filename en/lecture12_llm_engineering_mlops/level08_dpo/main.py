"""
We implement DPO (Direct Preference Optimization) alignment exactly as the equation says.
Of two responses to the same question (chosen/rejected), raise the probability
of the one a human picked and lower the other — while staying tethered to a
frozen reference model so we never drift too far from it.
No reward model, no reinforcement learning: watch alignment happen through a
single classification loss, tracked in tables of chosen/rejected log-probs and margins.
"""
import copy
import pathlib
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
CTX, D, H, LAYERS = 96, 64, 2, 2
U, A, E = "<|user|>", "<|assistant|>", "<|end|>"
BETA = 0.1

# Customer-service preference data: (question, chosen=polite response, rejected=blame-shifting response)
PAIRS = [
    ("Why is my delivery so late", "So sorry. Let me check it right away.",
     "Ask the courier company, not us."),
    ("I would like a refund", "Sure. May I have your order number?",
     "Read the refund policy first."),
    ("My product arrived broken", "So sorry. We will send a new one now.",
     "You probably used it wrong."),
    ("Nobody answered my inquiry", "Sorry for the delay. I will reply now.",
     "We are busy too, please wait."),
    ("My points disappeared", "Sorry. We will restore them right away.",
     "Points do expire sometimes."),
    ("I entered the wrong address", "No problem. We will fix the address now.",
     "That was your mistake, not ours."),
]


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        s = lambda t: t.view(B, T, self.h, C // self.h).transpose(1, 2)
        y = F.scaled_dot_product_attention(s(q), s(k), s(v), is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, D), nn.Embedding(CTX, D)
        self.blocks = nn.ModuleList(Block(D, H) for _ in range(LAYERS))
        self.lnf, self.head = nn.LayerNorm(D), nn.Linear(D, vocab, bias=False)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


def build_tok():
    text = "".join(q + w + l for q, w, l in PAIRS)
    itos = [U, A, E] + sorted(set(text))
    return itos, {s: i for i, s in enumerate(itos)}


def prompt_ids(stoi, q):
    return [stoi[U]] + [stoi[c] for c in q] + [stoi[A]]


def resp_ids(stoi, r):
    return [stoi[c] for c in r] + [stoi[E]]


def resp_logp(model, p_ids, r_ids):
    """Log-probability of the whole response given the prompt: log π(y|x) (sum of token logp)."""
    idx = torch.tensor(p_ids + r_ids)[None]
    logps = F.log_softmax(model(idx[:, :-1]), dim=-1)[0]
    tgt = idx[0, 1:]
    start = len(p_ids) - 1  # predictions from this position onward guess the response tokens
    pos = torch.arange(start, len(tgt))
    return logps[pos, tgt[start:]].sum()


@torch.no_grad()
def greedy_answer(model, stoi, itos, q, max_new=40):
    idx = torch.tensor(prompt_ids(stoi, q))[None]
    out = []
    for _ in range(max_new):
        nxt = int(model(idx[:, -CTX:])[:, -1].argmax())
        if nxt == stoi[E]:
            break
        out.append(nxt)
        idx = torch.cat([idx, torch.tensor([[nxt]])], 1)
    return "".join(itos[i] for i in out)


@torch.no_grad()
def pref_table(policy, ref, enc, title):
    print(title)
    print("    Question                     logπ(chosen)  logπ(rejected)  margin(vs ref)")
    n_correct = 0
    for q, p, w, l in enc:
        pw, pl = resp_logp(policy, p, w).item(), resp_logp(policy, p, l).item()
        rw, rl = resp_logp(ref, p, w).item(), resp_logp(ref, p, l).item()
        margin = (pw - rw) - (pl - rl)
        n_correct += pw > pl
        print(f"    {q:<28s}{pw:12.1f}{pl:14.1f}{margin:14.2f}")
    print(f"    -> pairs where chosen is more likely: {n_correct}/{len(enc)}")


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    print("=" * 66)
    print("Level 08 — Preference alignment with DPO (the loss equation, verbatim)")
    print("=" * 66)

    # [1] tokenizer and data encoding
    itos, stoi = build_tok()
    enc = [(q, prompt_ids(stoi, q), resp_ids(stoi, w), resp_ids(stoi, l))
           for q, w, l in PAIRS]
    print(f"[1] {len(enc)} preference pairs / vocab {len(itos)}")

    # [2] warm-up training (SFT): teach chosen and rejected *equally*, so both are plausible
    #     (in reality too, DPO starts from a model that has finished SFT)
    policy = TinyGPT(len(itos))
    opt = torch.optim.AdamW(policy.parameters(), lr=2e-3)
    print("[2] Warm-up SFT: both styles trained with equal weight (no preference yet)")
    for epoch in range(1, 81):
        total = torch.tensor(0.0)
        for _, p, w, l in enc:
            for r in (w, l):
                idx = torch.tensor(p + r)[None]
                logits = policy(idx[:, :-1])
                y = idx[0, 1:].clone()
                y[:len(p) - 1] = -100  # loss on the response part only (same as level07)
                total = total + F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y, ignore_index=-100)
        total = total / (2 * len(enc))
        opt.zero_grad(); total.backward(); opt.step()
        if epoch in (1, 40, 80):
            print(f"    epoch {epoch:2d}  loss {total.item():.3f}")

    # [3] freeze the reference model: a copy of our pre-alignment self, frozen
    ref = copy.deepcopy(policy)
    for prm in ref.parameters():
        prm.requires_grad_(False)
    print("[3] reference = a frozen copy of policy. At the start every margin is exactly 0.")
    pref_table(policy, ref, enc, "    --- before DPO training ---")
    q0 = PAIRS[0][0]
    print(f"    greedy answer: {greedy_answer(policy, stoi, itos, q0)!r}")

    # [4] DPO training: loss = -log σ( β·[(logπ(yw)-logπref(yw)) - (logπ(yl)-logπref(yl))] )
    with torch.no_grad():  # the reference model's log-probs never change, so compute them once
        ref_lp = [(resp_logp(ref, p, w), resp_logp(ref, p, l)) for _, p, w, l in enc]
    opt = torch.optim.AdamW(policy.parameters(), lr=2e-4)
    print(f"[4] DPO training (beta={BETA})")
    for epoch in range(1, 31):
        loss = torch.tensor(0.0)
        for (_, p, w, l), (rw, rl) in zip(enc, ref_lp):
            pw, pl = resp_logp(policy, p, w), resp_logp(policy, p, l)
            logits = BETA * ((pw - rw) - (pl - rl))  # the bracketed part of the DPO equation
            loss = loss - F.logsigmoid(logits)
        loss = loss / len(enc)
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch in (1, 10, 20, 30):
            print(f"    epoch {epoch:2d}  dpo_loss {loss.item():.4f}")

    # [5] the alignment result: chosen ↑ / rejected ↓, the margin opens up
    pref_table(policy, ref, enc, "[5] --- after DPO training ---")
    print(f"    greedy answer: {greedy_answer(policy, stoi, itos, q0)!r}")
    print("[6] RLHF needs a reward model + PPO with 4 models in memory; DPO reaches")
    print("    the same goal with the single loss line above. That is why DPO is loved.")


if __name__ == "__main__":
    main()
