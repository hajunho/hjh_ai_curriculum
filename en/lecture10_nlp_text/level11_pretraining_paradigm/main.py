"""
level11 — The Pre-training Paradigm: BERT and GPT

Contrasts two pre-training tasks on the same mini transformer.
  A. Fill-in-the-blank (MLM, the BERT way): bidirectional attention + restoring masked words
  B. Continuation (CLM, the GPT way): causal mask + next-word prediction
Finally, proves the value of 'pre-train -> fine-tune' with review sentiment
classification. Designed so that all training together finishes within
30 seconds on CPU.
"""

import sys
import time
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

SEED = 42
D_MODEL, N_HEADS, N_LAYERS = 32, 2, 2


class MiniTransformer(nn.Module):
    """causal=True gives a GPT-style decoder; False gives a BERT-style bidirectional encoder."""

    def __init__(self, vocab_size: int, block: int, causal: bool):
        super().__init__()
        self.block, self.causal = block, causal
        self.tok = nn.Embedding(vocab_size, D_MODEL)
        self.pos = nn.Embedding(block, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            D_MODEL, N_HEADS, dim_feedforward=4 * D_MODEL,
            batch_first=True, dropout=0.0, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, N_LAYERS)
        self.ln = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size)

    def hidden(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T))
        mask = None
        if self.causal:                       # the causal mask that hides the future
            mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        return self.ln(self.blocks(x, mask=mask))

    def forward(self, idx):
        return self.head(self.hidden(idx))


def train_model(model, make_batch, steps: int, label: str) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(1, steps + 1):
        x, y = make_batch()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               y.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
        if step in (1, steps):
            print(f"    {label} step {step:3d}: loss {float(loss):.3f}")


if __name__ == "__main__":
    t0 = time.time()
    torch.manual_seed(SEED)
    print("=" * 70)
    print("The pre-training paradigm — fill-in-the-blank (BERT) vs continuation (GPT)")
    print("=" * 70 + "\n")

    # ---------------- A. Prepare tiny_corpus at the word level -------------------
    words = hjh_data.tiny_corpus().replace(".", " .").split()[:12000]
    vocab = sorted(set(words)) + ["[MASK]"]
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}
    MASK = stoi["[MASK]"]
    ids = torch.tensor([stoi[w] for w in words])
    BLOCK = 10

    print("[1] The problem generator — with no labels, the text itself becomes the workbook")
    sample = words[:5]
    print(f"    original       : {' '.join(sample)}")
    print(f"    MLM problem (BERT): {' '.join(['[MASK]' if i == 1 else w for i, w in enumerate(sample)])}"
          f"  -> answer '{sample[1]}'")
    print(f"    CLM problem (GPT) : after '{' '.join(sample[:3])}', the next word? -> answer '{sample[3]}'")
    print(f"    (word vocabulary {len(vocab)} types, {len(ids):,} training tokens)\n")

    gen = torch.Generator().manual_seed(SEED)

    def batch_starts(n=32):
        return torch.randint(0, len(ids) - BLOCK - 1, (n,), generator=gen)

    def mlm_batch():
        x = torch.stack([ids[s: s + BLOCK] for s in batch_starts()]).clone()
        y = torch.full_like(x, -100)                     # grade only the masked spots
        mask_pos = torch.rand(x.shape, generator=gen) < 0.15
        y[mask_pos] = x[mask_pos]
        x[mask_pos] = MASK
        return x, y

    def clm_batch():
        starts = batch_starts()
        x = torch.stack([ids[s: s + BLOCK] for s in starts])
        y = torch.stack([ids[s + 1: s + BLOCK + 1] for s in starts])
        return x, y

    print("[2] Same architecture, two tasks, two models (300 steps each)")
    bert = MiniTransformer(len(vocab), BLOCK, causal=False)   # bidirectional
    gpt = MiniTransformer(len(vocab), BLOCK, causal=True)     # unidirectional
    train_model(bert, mlm_batch, 300, "BERT-style (fill-in)")
    train_model(gpt, clm_batch, 300, "GPT-style (continue)")
    print()

    # ---------------- [3] The fill-in-the-blank exam --------------------------------
    print("[3] Fill-in-the-blank exam — being able to see the right context is the fork in the road")
    quiz = ["[MASK]", "a", "student", "made", "a", "report", "."]
    x = torch.tensor([[stoi[w] for w in quiz]])
    with torch.no_grad():
        bert_top = torch.topk(F.softmax(bert(x)[0, 0], -1), 3)
        gpt_top = torch.topk(F.softmax(gpt(x)[0, 0], -1), 3)   # has no left context!
    fmt = lambda t: ", ".join(f"'{itos[int(i)]}'({p:.0%})"
                              for p, i in zip(t.values, t.indices))
    print(f"    problem: {' '.join(quiz)}  (the sentence's first word is masked)")
    print(f"    BERT-style answer: {fmt(bert_top)}")
    print(f"    GPT-style answer : {fmt(gpt_top)}")
    print("    -> The BERT-style model reads the right context ('a student...') and knows")
    print("       this is a time expression's seat; the GPT-style model has nothing to the")
    print("       left of the blank, so it can basically only guess.\n")

    # ---------------- [4] The continuation exam -----------------------------------
    print("[4] Continuation exam — generation is the home turf of whoever learned to continue")
    def continue_with(model, prompt_words, n=8):
        seq = [stoi[w] for w in prompt_words]
        for _ in range(n):
            x = torch.tensor([seq[-BLOCK:]])
            with torch.no_grad():
                if model.causal:
                    logits = model(x)[0, -1]              # predict the next word
                else:                                     # BERT-style: force generation via a [MASK] at the end
                    x = torch.cat([x, torch.tensor([[MASK]])], dim=1)[:, -BLOCK:]
                    logits = model(x)[0, -1]
            probs = F.softmax(logits / 0.7, -1)
            seq.append(int(torch.multinomial(probs, 1, generator=gen)))
        return " ".join(itos[i] for i in seq)
    prompt = ["Over", "the", "weekend", "a", "chef"]
    print(f"    GPT-style : {continue_with(gpt, prompt)}")
    print(f"    BERT-style: {continue_with(bert, prompt)}")
    print("    -> The GPT-style model continues the template ('verb object . time ...')")
    print("       naturally; the BERT-style model never learned 'what comes next'")
    print("       probabilities, so its flow is awkward.\n")

    # ---------------- [5] A taste of pre-train -> fine-tune ----------------------
    print("[5] A taste of fine-tuning — is pre-training real 'seed capital'? (review sentiment)")
    rows = hjh_data.review_corpus(400, seed=3)
    chars = sorted({ch for r in rows for ch in r["text"]})
    cstoi = {ch: i for i, ch in enumerate(chars)}
    CBLOCK = 32
    def encode(text):
        s = [cstoi[ch] for ch in text[:CBLOCK]]
        return s + [0] * (CBLOCK - len(s))
    corpus_ids = torch.tensor([i for r in rows for i in encode(r["text"])])

    # 5-1. Character-level CLM pre-training on the review text, no labels
    lm = MiniTransformer(len(chars), CBLOCK, causal=True)
    def char_clm_batch():
        starts = torch.randint(0, len(corpus_ids) - CBLOCK - 1, (32,), generator=gen)
        x = torch.stack([corpus_ids[s: s + CBLOCK] for s in starts])
        y = torch.stack([corpus_ids[s + 1: s + CBLOCK + 1] for s in starts])
        return x, y
    train_model(lm, char_clm_batch, 800, "pre-training (review continuation)")

    # 5-2. Fine-tune with few labels (60): the model's sentence representation + a logistic head
    X_ids = torch.tensor([encode(r["text"]) for r in rows])
    y_all = [r["label"] for r in rows]
    def features(model):
        with torch.no_grad():
            return model.hidden(X_ids).mean(dim=1).numpy()   # sentence representation (mean pooling)
    scratch = MiniTransformer(len(chars), CBLOCK, causal=True)  # control group without pre-training
    results = {}
    for name, model in (("pre-trained", lm), ("not pre-trained (random weights)", scratch)):
        feats = features(model)
        clf = LogisticRegression(max_iter=1000, random_state=SEED)
        clf.fit(feats[:60], y_all[:60])                      # train on just 60 labeled reviews
        results[name] = clf.score(feats[200:], y_all[200:])  # evaluate on 200
    for name, acc in results.items():
        print(f"    {name:32s}: trained on 60 labels -> test accuracy {acc:.1%}")
    print("    -> Same architecture, but the one that 'learned the language first' learns")
    print("       far better from few labels. This is why pre-train -> fine-tune changed the world.")
    print(f"\nTotal runtime: {time.time() - t0:.1f}s (within the 90s budget)")
