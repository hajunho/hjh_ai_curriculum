# Lecture 08 · Level 05 — First Steps with PyTorch — Tensors

> A tour of the tensor, the basic ingredient of the deep-learning framework PyTorch, compared side by side with numpy. Conclusion first: if you know numpy, you've already learned 90% of this.

**Difficulty** ⭐⭐⭐ / **Prerequisites** level04 (backpropagation from scratch) / **Estimated time** 40 min

## 1. Why learn this — the business view

In levels 00–04 you implemented the entire principle of neural networks in numpy. Now it's time to switch to the language of practice. PyTorch is the most widely used open-source deep-learning framework in both research and industry, and every line of code in the upcoming computer-vision, NLP, and LLM lectures is written in PyTorch. Whether you're reviewing a vendor's model code, talking with your company's data scientists, or reading job postings, PyTorch is the lingua franca. And the noun of that lingua franca is the tensor. If you can read a tensor's three properties — shape, dtype, and device — you can follow how data flows through even unfamiliar deep-learning code.

## 2. Understanding through an analogy

**A tensor = a generalized Excel sheet + two superpowers.**

- A single number (scalar) is one cell; a vector is one column; a matrix is one sheet. A tensor extends this naturally to "a stack of sheets" (3-D) and "stacks of stacks" (4-D). Example: 100 grayscale images form a (100, 16, 16) tensor.
- A numpy array does all of that too. The tensor's superpowers are two:
  1. **Automatic differentiation**: it keeps a ledger of every computation and later does the differentiation (responsibility tracing) automatically. What you did by hand in level04 becomes free.
  2. **Moving to a GPU**: one line, `.to('cuda')`, relocates the computation from CPU to GPU. One trip of the moving truck, and the same household (data) runs in a much bigger workshop.

**The device is "which desk you're working at":** the CPU desk, the NVIDIA GPU desk (cuda), the Apple-silicon GPU desk (mps). One important rule — only ingredients on the same desk can be combined. Mix a CPU tensor with a GPU tensor and you get an error.

## 3. Key concepts

### 3-1. The three tensor properties: shape, dtype, device

90% of error messages come from one of these three not matching. When you meet an unfamiliar tensor, make it a reflex to check `x.shape, x.dtype, x.device`.

### 3-2. Reshaping and broadcasting

- `reshape` / `view` don't move data; they only change the "viewing window." Twelve elements can be seen as (3,4) or as (2,6).
- Broadcasting: even with different shapes, like (3,1) + (4,), if the rule holds (from the trailing dimension, each must be 1 or equal) the arrays are expanded automatically. The rules are identical to numpy's.

### 3-3. The relationship with numpy — sharing vs copying

| Method | Behavior | Caution |
|---|---|---|
| `torch.from_numpy(arr)` | **Shares memory** (both look at the same sheet of paper) | Modify one side and the other changes too |
| `t.numpy()` | **Shares memory** (CPU tensors only) | Same as above |
| `torch.tensor(arr)` | Creates a **copy** | Safe, but double the memory |

Sharing is fast but can cause "value changed from afar" accidents. You'll verify this yourself in exercise [5].

### 3-4. dtype — trading precision for memory

float32 (4 bytes) is the default. float16 (2 bytes) halves memory and roughly doubles speed at the cost of losing trailing decimal digits. This trade leads directly to mixed-precision training in level11.

### 3-5. Why not just keep using numpy

"We built the whole network in numpy in level04 — do we really have to switch?" Good question. For small exercises numpy is plenty. But (1) with millions of weights, hand-writing the derivatives becomes close to impossible; (2) on CPU alone, training takes hundreds of times longer; and (3) you can't reuse battle-tested layers, optimizers, and data loaders. PyTorch solves these three problems with autograd, device, and the nn module respectively. If numpy is "hand calculation plus a paper ledger," PyTorch is "an accounting system." The principle is the same, but when the scale changes, you change tools.

### 3-6. A note on this exercise's limits

The tensors in this level still have "differentiation switched off." Real deep learning begins the moment you flip the `requires_grad` switch — and that's the next level.

## 4. Hands-on — main.py

Run:

```bash
cd lecture08_deep_learning_foundations/level05_pytorch_tensors
python3 main.py
```

Output, in order:

- **[1]** Four ways to create tensors (tensor/zeros/arange/randn). Thanks to `torch.manual_seed(42)`, randn's first element is +0.3367 every time — the fundamentals of reproducibility.
- **[2]–[4]** reshape, transpose, broadcasting, aggregation, matrix multiplication. `(3,4) @ (4,2) = (3,2)` is exactly one neural-network layer — the same operation you did with numpy in levels 01–04.
- **[5]** Proves experimentally that a tensor made with `from_numpy` shares memory with the numpy array. Set the numpy side to 99 and the tensor becomes 99 too.
- **[6]** Device check. This exercise runs on CPU, but it prints cuda/mps availability and shows how one line, `.to(device)`, does the move. Works without error even with no GPU.
- **[7]** Watch trailing digits get shaved off in the float32 → float16 conversion.
- **[8]** A numpy ↔ torch API mapping table. Keep this table at your side and your numpy experience ports over directly.

## 5. Try it yourself

1. **(Easy)** Reshape `torch.arange(24)` to (2,3,4), (4,6), and (24,) and print each. Also check what error (5,5) raises. *Hint: when the element counts don't match you get a RuntimeError. This is practice in reading error messages.*
2. **(Medium)** In [5], predict how the output changes if you use `torch.tensor` instead of `torch.from_numpy`, then run to confirm. *Hint: it's a copy, so the 99 edit doesn't propagate.*
3. **(Challenge)** Port level04's `forward()` to a torch-tensor version as-is (backward still forbidden). *Hint: swap np.tanh → torch.tanh and np.exp → torch.exp and it almost just runs. Consult mapping table [8].*

## 6. Common mistakes

- **Doing network math with integer tensors.** `torch.tensor([1, 2])` becomes int64. Weight computations need float, so write `[1.0, 2.0]` or specify `dtype=torch.float32`.
- **Confusing sharing with copying.** You modify a tensor made with `from_numpy` and the original numpy array changes too — a "ghost bug." When unsure, `clone()`.
- **Device mismatch.** `RuntimeError: Expected all tensors to be on the same device` — happens when the model is on GPU and the data on CPU. Move data and model to the same desk.
- **Panicking when `view` fails.** After a transpose or similar, memory is no longer contiguous and `view` refuses. Use `reshape` or `contiguous().view()` instead.

## Next level preview

Flip the `requires_grad=True` switch on a tensor and PyTorch records every computation in its ledger, then replaces all of level04's backpropagation with the single line `loss.backward()`. In level06 you'll solve XOR again with autograd and optimizers (SGD, Adam) and compare just how much shorter the code gets.
