"""
Hands-on quantization.
We implement symmetric quantization in numpy — compressing fp32 weights to
int8/int4 integers and restoring them — then apply it to the weights of a
character-level mini language model briefly trained on tiny_corpus, measuring
the memory savings and the change in loss.
We also confirm, with numbers, why block-wise quantization is the heart of
deployment formats like GGUF.
"""

import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

torch.manual_seed(42)
np.random.seed(42)

CONTEXT = 8      # how many characters to look at when guessing the next one
EMB, HID = 24, 128


def quantize(w: np.ndarray, bits: int):
    """Symmetric quantization: scale = max|w| / qmax, q = round(w / scale)."""
    qmax = 2 ** (bits - 1) - 1            # int8 -> 127, int4 -> 7
    scale = float(np.abs(w).max()) / qmax + 1e-12
    q = np.clip(np.round(w / scale), -qmax, qmax)
    return q.astype(np.int8), scale


def dequantize(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float32) * scale


def quantize_blockwise(w: np.ndarray, bits: int, block: int = 64):
    """Flatten to 1-D and give every block of `block` values its own scale
    (the core idea of the GGUF family)."""
    flat = w.reshape(-1)
    out = np.empty_like(flat, dtype=np.float32)
    for i in range(0, flat.size, block):
        chunk = flat[i:i + block]
        q, s = quantize(chunk, bits)
        out[i:i + block] = dequantize(q, s)
    return out.reshape(w.shape)


class CharMLP(nn.Module):
    """A micro language model: concatenate CONTEXT character embeddings and predict the next character."""

    def __init__(self, vocab: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, EMB)
        self.fc1 = nn.Linear(CONTEXT * EMB, HID)
        self.fc2 = nn.Linear(HID, vocab)

    def forward(self, x):
        h = self.emb(x).reshape(x.shape[0], -1)
        return self.fc2(torch.tanh(self.fc1(h)))


def make_batches(ids: torch.Tensor, n: int, batch: int, gen: torch.Generator):
    for _ in range(n):
        idx = torch.randint(0, len(ids) - CONTEXT - 1, (batch,), generator=gen)
        x = torch.stack([ids[i:i + CONTEXT] for i in idx])
        y = ids[idx + CONTEXT]
        yield x, y


def eval_loss(model: nn.Module, ids: torch.Tensor) -> float:
    gen = torch.Generator().manual_seed(7)  # evaluation batches are fixed too
    model.eval()
    with torch.no_grad():
        losses = [F.cross_entropy(model(x), y).item()
                  for x, y in make_batches(ids, 20, 256, gen)]
    return sum(losses) / len(losses)


def apply_quant(model: nn.Module, fn) -> nn.Module:
    """Return a copy of the model with quantize->restore applied to every weight tensor."""
    import copy
    m = copy.deepcopy(model)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.from_numpy(fn(p.numpy().copy())))
    return m


if __name__ == "__main__":
    print("[1] How symmetric quantization works — see it on a small array")
    w = np.array([0.82, -1.30, 0.05, 0.44, -0.71], dtype=np.float32)
    q8, s8 = quantize(w, 8)
    restored = dequantize(q8, s8)
    print(f"    original fp32: {np.round(w, 4)}")
    print(f"    int8 integers: {q8}  (scale={s8:.6f})")
    print(f"    restored     : {np.round(restored, 4)}")
    print(f"    max error    : {np.abs(w - restored).max():.6f}")

    print("\n[2] Memory savings — for 1 million parameters")
    n_params = 1_000_000
    for name, bytes_per in [("fp32", 4), ("fp16", 2), ("int8", 1), ("int4", 0.5)]:
        mb = n_params * bytes_per / 1e6
        print(f"    {name:5s}: {mb:6.1f} MB  ({4 / bytes_per:.0f}x smaller than fp32)")

    print("\n[3] Training the mini language model (tiny_corpus, character level)")
    text = hjh_data.tiny_corpus()[:20000]
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    model = CharMLP(len(chars))
    n_p = sum(p.numel() for p in model.parameters())
    print(f"    vocabulary {len(chars)} characters, {n_p:,} parameters")
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(0)
    model.train()
    for step, (x, y) in enumerate(make_batches(ids, 400, 128, gen), 1):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"    step {step:3d}  loss {loss.item():.3f}")

    base = eval_loss(model, ids)
    print(f"\n[4] Performance after int8 quantization (per-tensor)")
    m8 = apply_quant(model, lambda a: dequantize(*quantize(a, 8)))
    l8 = eval_loss(m8, ids)
    print(f"    fp32 loss = {base:.4f}")
    print(f"    int8 loss = {l8:.4f}  (increase {l8 - base:+.4f}) -> essentially no damage")

    print("\n[5] Pushing down to int4? Plus outliers and block quantization")
    m4 = apply_quant(model, lambda a: dequantize(*quantize(a, 4)))
    l4 = eval_loss(m4, ids)
    print(f"    int4 loss = {l4:.4f}  (increase {l4 - base:+.4f})")
    # Real large-model weights contain unusually large 'outlier' values.
    # We reproduce how a single outlier inflates the whole tensor's scale
    # and wrecks the precision of everything else.
    w_out = np.random.randn(1024).astype(np.float32) * 0.05
    w_out[100] = 3.0  # insert one outlier
    err_tensor = np.abs(w_out - dequantize(*quantize(w_out, 4))).mean()
    err_block = np.abs(w_out - quantize_blockwise(w_out, 4, block=64)).mean()
    print(f"    Mean int4 restoration error of the outlier-laden array:")
    print(f"      one scale for the whole tensor: {err_tensor:.5f}")
    print(f"      one scale per 64-value block  : {err_block:.5f}  ({err_tensor / err_block:.1f}x more accurate)")
    print("    -> With per-block scales, one large value no longer ruins everyone's precision.")
    print("    -> llama.cpp's GGUF format is exactly a file holding such block-quantized weights.")

    print("\n[6] Summary")
    print(f"    {'format':6s} {'size (rel.)':>11s} {'eval loss':>10s}")
    for name, rel, l in [("fp32", "1.00x", base), ("int8", "0.25x", l8),
                         ("int4", "0.125x", l4)]:
        print(f"    {name:6s} {rel:>11s} {l:>10.4f}")
    print("    The point: give up a little precision and save 4-8x the memory —")
    print("    and techniques like block quantization make that 'little' even smaller.")
