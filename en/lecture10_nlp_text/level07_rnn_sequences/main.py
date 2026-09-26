"""
level07 — RNNs and Sequence Models

Trains a character-level next-character mini language model on part of
tiny_corpus using a torch RNN. Demonstrates how generated text changes
before/during/after training, next-character probabilities, and the effect
of temperature on generation.
Designed tiny enough to finish in a few seconds on CPU.
"""

import sys
import time
import pathlib

import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

SEED = 42
SEQ_LEN = 40          # characters read at a time
HIDDEN = 64           # size of the relay memo (hidden state)
EMBED = 32
EPOCHS = 12
BATCH = 64


class CharRNN(nn.Module):
    """Embedding -> RNN (relay memo) -> next-character scores."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, EMBED)
        self.rnn = nn.RNN(EMBED, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, vocab_size)

    def forward(self, x, h=None):
        emb = self.embed(x)                 # (B, T, EMBED)
        out, h = self.rnn(emb, h)           # out: the memo at every timestep (B, T, HIDDEN)
        return self.head(out), h            # next-character scores per timestep


def make_dataset(text: str, stoi: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Input = a character sequence; target = the same sequence shifted by one."""
    ids = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    n_chunks = (len(ids) - 1) // SEQ_LEN
    x = ids[: n_chunks * SEQ_LEN].view(n_chunks, SEQ_LEN)
    y = ids[1: n_chunks * SEQ_LEN + 1].view(n_chunks, SEQ_LEN)
    return x, y


@torch.no_grad()
def generate(model, stoi, itos, prompt: str, length: int = 60,
             temperature: float = 0.8) -> str:
    """Continue the prompt by length characters."""
    model.eval()
    ids = [stoi.get(ch, 0) for ch in prompt]
    x = torch.tensor([ids], dtype=torch.long)
    logits, h = model(x)                              # read the whole prompt to build the memo
    out = list(prompt)
    last = logits[0, -1]
    for _ in range(length):
        probs = torch.softmax(last / temperature, dim=-1)
        idx = int(torch.multinomial(probs, 1))
        out.append(itos[idx])
        logits, h = model(torch.tensor([[idx]]), h)   # pass the memo along, one char at a time
        last = logits[0, -1]
    model.train()
    return "".join(out)


@torch.no_grad()
def next_char_topk(model, stoi, itos, prompt: str, k: int = 3):
    model.eval()
    x = torch.tensor([[stoi.get(ch, 0) for ch in prompt]], dtype=torch.long)
    logits, _ = model(x)
    probs = torch.softmax(logits[0, -1], dim=-1)
    top = torch.topk(probs, k)
    model.train()
    return [(itos[int(i)], float(p)) for p, i in zip(top.values, top.indices)]


if __name__ == "__main__":
    t0 = time.time()
    torch.manual_seed(SEED)
    print("=" * 70)
    print("An RNN language model — guessing the next character with a relay memo")
    print("=" * 70 + "\n")

    # [1] Data prep ------------------------------------------------------
    text = hjh_data.tiny_corpus()[:20000]
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    x, y = make_dataset(text, stoi)
    print(f"[1] Data: {len(text):,} chars, character vocabulary {len(chars)} types, "
          f"{x.shape[0]} training sequences (length {SEQ_LEN})")
    sample = text[:20]
    print(f"    input example : {sample!r}")
    print(f"    target example: {text[1:21]!r}  <- the input shifted by one char\n")

    # [2] Assembling the model ---------------------------------------------
    model = CharRNN(len(chars))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[2] Model: embedding({EMBED}) -> RNN(memo of {HIDDEN} slots) -> output layer")
    print(f"    {n_params:,} parameters (a billionth-ish of today's LLMs)\n")

    # [3] Training — watching the generations start to make sense -----------
    print("[3] Training: falling loss and changing generations")
    prompt = "Yesterday a student "
    print(f"    before training: {generate(model, stoi, itos, prompt)!r}")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(x.shape[0])
        total = 0.0
        for i in range(0, x.shape[0], BATCH):
            idx = perm[i:i + BATCH]
            logits, _ = model(x[idx])                 # the memo restarts at 0 for each sequence
            loss = loss_fn(logits.reshape(-1, len(chars)), y[idx].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(idx)
        avg = total / x.shape[0]
        if epoch in (1, EPOCHS // 2, EPOCHS):
            print(f"    epoch {epoch}: loss {avg:.3f} | generation: "
                  f"{generate(model, stoi, itos, prompt, 40)!r}")
    print("    -> As the loss falls, the 'subject + verb + object.' skeleton takes shape.\n")

    # [4] Next-character prediction ---------------------------------------------
    print("[4] Next-character probabilities, top 3")
    for p in ["a student checked a re", "Today an office worker ", "a pot of st"]:
        top = ", ".join(f"'{ch}'({prob:.0%})" for ch, prob in
                        next_char_topk(model, stoi, itos, p))
        print(f"    after {p!r}: {top}")
    print()

    # [5] Temperature experiment --------------------------------------------------
    print("[5] The temperature dial — THE parameter from the LLM APIs")
    for temp in (0.3, 1.5):
        text_out = generate(model, stoi, itos, "Over the weekend ", 50, temperature=temp)
        print(f"    temperature {temp}: {text_out!r}")
    print("    -> Low = safe and monotonous; high = varied but erratic.")
    print("\n    Limitation memo: an RNN summarizes the past into one relay memo, so in")
    print("    long sentences the early content blurs (long-term dependency). The fix")
    print("    is next level's attention.")
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")
