"""
Hands-on miniature SFT (Supervised Fine-Tuning), a.k.a. instruction tuning.
A base model fresh out of pre-training only 'continues' text — it does not
answer questions. We build instruction-response data with a chat template
(<|user|>/<|assistant|>/<|end|>), teach 'the shape of following instructions'
via loss masking (computing the loss only on the response part), and compare
generations before and after training.
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
PAD, U, A, E = "<pad>", "<|user|>", "<|assistant|>", "<|end|>"


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


class CharTok:
    """Special tokens (template separators) are single whole tokens; everything else is character-level."""

    def __init__(self, text):
        self.itos = [PAD, U, A, E] + sorted(set(text))
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


NAMES = ["Alex", "Grace", "Chris", "Emily", "Susan", "Daniel", "Haley", "Nora"]
ITEMS = [("pencil", 12), ("notebook", 7), ("eraser", 30), ("scissors", 5), ("glue", 21), ("ruler", 9)]


def build_sft_data():
    data = [(f"Say hello to {n}", f"Hello {n}, have a great day!") for n in NAMES]
    for a in range(1, 6):
        for b in range(1, 6):
            if a != b:  # combinations with a == b are held out as 'unseen instruction' tests
                data.append((f"What is {a} plus {b}?", f"{a} plus {b} is {a + b}."))
    data += [(f"Check the stock of {it}", f"The stock of {it} is {cnt} units.") for it, cnt in ITEMS]
    return data


def render(tok, instr, resp=None):
    # chat template: <|user|>instruction<|assistant|>response<|end|>
    ids = [tok.stoi[U]] + tok.encode(instr) + [tok.stoi[A]]
    if resp is not None:
        ids += tok.encode(resp) + [tok.stoi[E]]
    return ids


def make_sft_tensors(tok, data):
    # Build next-token prediction with x=ids[:-1], y=ids[1:],
    # but mask the prompt (instruction) span's targets with -100 to exclude it from the loss.
    rows = [(render(tok, i, r), len(render(tok, i))) for i, r in data]
    L = max(len(ids) for ids, _ in rows) - 1
    X = torch.full((len(rows), L), tok.stoi[PAD], dtype=torch.long)
    Y = torch.full((len(rows), L), -100, dtype=torch.long)
    for k, (ids, plen) in enumerate(rows):
        x, y = ids[:-1], ids[1:]
        X[k, :len(x)] = torch.tensor(x)
        for i in range(plen - 1, len(y)):  # keep only the response tokens (and <|end|>) as targets
            Y[k, i] = y[i]
    return X, Y


def get_batch(data, bs, gen):
    ix = torch.randint(len(data) - CTX - 1, (bs,), generator=gen)
    return (torch.stack([data[i:i + CTX] for i in ix]),
            torch.stack([data[i + 1:i + CTX + 1] for i in ix]))


@torch.no_grad()
def continue_text(model, tok, prompt, n=40, gen=None):
    idx = torch.tensor(tok.encode(prompt))[None]
    for _ in range(n):
        logits = model(idx[:, -CTX:])[:, -1] / 0.7
        nxt = torch.multinomial(logits.softmax(-1), 1, generator=gen)
        idx = torch.cat([idx, nxt], 1)
    return tok.decode(idx[0].tolist())


@torch.no_grad()
def answer(model, tok, instr, max_new=44):
    # fill in the template and generate greedily until <|end|> appears
    idx = torch.tensor(render(tok, instr))[None]
    out = []
    for _ in range(max_new):
        nxt = int(model(idx[:, -CTX:])[:, -1].argmax())
        if nxt == tok.stoi[E]:
            break
        out.append(nxt)
        idx = torch.cat([idx, torch.tensor([[nxt]])], 1)
    return tok.decode(out)


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    print("=" * 62)
    print("Level 07 — Instruction tuning (SFT): turning a continuation machine into an assistant")
    print("=" * 62)

    # [1] tokenizer: every character in corpus + instruction data + 4 special tokens
    sft_data = build_sft_data()
    corpus = hjh_data.tiny_corpus()
    all_text = corpus + "".join(i + r for i, r in sft_data) + "0123456789 plus?"
    tok = CharTok(all_text)
    print(f"[1] {len(sft_data)} instruction-response pairs / vocab {len(tok.itos)} (incl. 4 special tokens)")

    # [2] pre-train the base model — it learns only 'continue the next character'
    model = TinyGPT(len(tok.itos))
    corpus_ids = torch.tensor(tok.encode(corpus))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    print("[2] Pre-training the base model (300 steps)")
    for step in range(1, 301):
        x, y = get_batch(corpus_ids, 24, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step in (1, 150, 300):
            print(f"    step {step:3d}  loss {loss.item():.3f}")

    # [3] the base model's limits: great at continuing, hopeless at following instructions
    print("[3] The base model only 'continues'")
    print(f"    Continuation : {continue_text(model, tok, 'Today an office worker ', gen=gen)!r}")
    print(f"    Given an instruction: {answer(model, tok, 'What is 3 plus 5?')!r}  <- not an answer, just characters")

    # [4] visualize the chat template and loss masking
    instr, resp = sft_data[10]
    ids, plen = render(tok, instr, resp), len(render(tok, instr))
    marks = "".join("-" if i < plen else "#" for i in range(len(ids)))
    print("[4] Chat template and loss masking (-: excluded from loss / #: loss computed)")
    print(f"    Tokens: {U}{instr}{A}{resp}{E}")
    print(f"    Mask  : {marks}  (prompt {plen} tokens excluded, only the {len(ids) - plen} response tokens trained)")

    # [5] SFT: cross entropy on response tokens only (ignore_index=-100)
    X, Y = make_sft_tensors(tok, sft_data)
    print(f"[5] SFT training (batch {X.shape}, 700 steps)")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(1, 701):
        ix = torch.randint(len(X), (16,), generator=gen)
        logits = model(X[ix])
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y[ix].view(-1),
                               ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
        if step in (1, 350, 700):
            print(f"    step {step:3d}  loss {loss.item():.3f}")

    # [6] after SFT: compare responses to the same instructions
    print("[6] Responses after SFT (greedy)")
    for q, note in [("What is 3 plus 5?", "an instruction seen in training"),
                    ("Check the stock of notebook", "an instruction seen in training"),
                    ("What is 4 plus 4?", "unseen combination (a=b) — watch whether the format holds")]:
        print(f"    Q: {q:<27s} A: {answer(model, tok, q)!r}  ({note})")
    print("    The format (template, polite phrasing, <|end|>) is learned solidly, but content")
    print("    generalization takes bigger models and data. This is why scale matters.")


if __name__ == "__main__":
    main()
