"""
A tour of the tensor, PyTorch's basic ingredient.
Runs creation / reshaping / broadcasting / aggregation / matrix multiplication
step by step, checks interconversion with numpy arrays (including memory
sharing) and the device concept, then prints a numpy <-> torch API mapping table.
"""

import numpy as np
import torch


def main():
    torch.manual_seed(42)                 # Reproducibility: fixed RNG seed
    np.random.seed(42)

    print("[1] Tensor = numpy array + autodiff ability + GPU-moving ability")
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    z = torch.zeros(2, 3)
    r = torch.arange(6)
    g = torch.randn(2, 3)                 # standard-normal randoms (fixed seed -> always the same)
    print(f"    tensor  : {a.tolist()}  shape={tuple(a.shape)}, dtype={a.dtype}")
    print(f"    zeros   : shape={tuple(z.shape)}")
    print(f"    arange  : {r.tolist()}")
    print(f"    randn   : first element {g[0, 0].item():+.4f} (fixed seed -> identical every run)\n")

    print("[2] Reshaping — the data stays put; only the viewing window changes")
    x = torch.arange(12, dtype=torch.float32)
    m = x.reshape(3, 4)
    print(f"    arange(12).reshape(3,4) ->\n{m}")
    print(f"    m.T shape = {tuple(m.T.shape)}, m.flatten() length = {m.flatten().numel()}\n")

    print("[3] Broadcasting — different shapes expand automatically when the rule holds")
    col = torch.tensor([[10.0], [20.0], [30.0]])      # (3,1)
    row = torch.tensor([1.0, 2.0, 3.0, 4.0])          # (4,)
    print(f"    (3,1) + (4,) -> {tuple((col + row).shape)} matrix:\n{col + row}\n")

    print("[4] Aggregation and matrix multiplication")
    print(f"    m.sum() = {m.sum().item():.0f}, m.mean(dim=0) = {m.mean(dim=0).tolist()}")
    W = torch.randn(4, 2)
    out = m @ W                            # the essence of one neural-network layer
    print(f"    (3,4) @ (4,2) = {tuple(out.shape)}  <- one network layer is exactly this operation\n")

    print("[5] numpy <-> torch conversion — they share the same memory (CPU tensors)")
    arr = np.ones(3, dtype=np.float32)
    t = torch.from_numpy(arr)              # not a copy — 'looking at the same sheet of paper together'
    arr[0] = 99.0                          # modify the numpy side and...
    print(f"    set numpy to 99 -> torch tensor is also {t.tolist()} (sharing confirmed)")
    back = t.numpy()
    t[1] = -7.0
    print(f"    set torch to -7 -> numpy array is also {back.tolist()}")
    safe = torch.tensor(arr)               # when you need a copy: torch.tensor() / clone()
    arr[2] = 0.0
    print(f"    torch.tensor(arr) is a copy -> unaffected by editing the original: {safe.tolist()}\n")

    print("[6] device — 'which calculator' the tensor sits on")
    print(f"    default device: {a.device} (CPU)")
    print(f"    cuda available? {torch.cuda.is_available()}")
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    print(f"    mps (Apple GPU) available? {mps_ok}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    moved = a.to(device)                   # moves if a GPU exists, stays put otherwise
    print(f"    a.to('{device}') -> device = {moved.device}")
    print("    # Real GPU code is this same one line: model.to('cuda'), batch.to('cuda')\n")

    print("[7] dtype — the trade-off between precision (digits) and memory")
    f32 = torch.randn(4)
    f16 = f32.to(torch.float16)
    print(f"    float32: {f32[0].item():+.8f} ({f32.element_size()} bytes/element)")
    print(f"    float16: {f16[0].item():+.8f} ({f16.element_size()} bytes/element, trailing digits lost)\n")

    print("[8] API mapping vs numpy — what you know carries straight over")
    rows = [
        ("np.array([1,2])",        "torch.tensor([1,2])"),
        ("np.zeros((2,3))",        "torch.zeros(2,3)"),
        ("np.arange(6)",           "torch.arange(6)"),
        ("np.random.randn(2,3)",   "torch.randn(2,3)"),
        ("a.reshape(3,4)",         "a.reshape(3,4) / a.view(3,4)"),
        ("a.T",                    "a.T / a.transpose(0,1)"),
        ("np.dot(a,b) / a @ b",    "torch.matmul(a,b) / a @ b"),
        ("a.sum(axis=0)",          "a.sum(dim=0)"),
        ("np.concatenate([a,b])",  "torch.cat([a,b])"),
        ("a.astype(np.float32)",   "a.to(torch.float32)"),
        ("np.maximum(a,0)",        "torch.relu(a) / a.clamp(min=0)"),
        ("(none)",                 "a.to('cuda') — move to GPU"),
        ("(none)",                 "a.requires_grad_() — autodiff"),
    ]
    print(f"    {'numpy':34s}| torch")
    print(f"    {'-'*34}|{'-'*34}")
    for np_api, th_api in rows:
        print(f"    {np_api:34s}| {th_api}")
    print("\n[9] Recap: if you know numpy, you already know 90% of torch.")
    print("    The remaining 10% (autograd, device) is what makes deep learning possible — and it's the next level's topic.")


if __name__ == "__main__":
    main()
