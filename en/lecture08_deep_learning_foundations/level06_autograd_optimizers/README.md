# Lecture 08 · Level 06 — autograd and Optimizers

> Witness the moment the 30 lines of backpropagation you hand-wrote in level04 become the single line `loss.backward()`, and compare two "descent strategies": SGD and Adam.

**Difficulty** ⭐⭐⭐ / **Prerequisites** level05 (PyTorch tensors), level04 (backpropagation) / **Estimated time** 50 min

## 1. Why learn this — the business view

One of the decisive reasons deep learning spread across industry is automatic differentiation. If derivatives had to be hand-written, every change to a layer would need a math specialist to re-derive the backprop code. Thanks to autograd, "experiments that change the model architecture" became a matter of minutes — and that became companies' iteration speed. Practically, this level pays off twice. First, you'll be able to read the 5-step loop (zero_grad → forward → loss → backward → step) that appears in every piece of PyTorch training code you'll ever open. Second, you'll be able to answer the perennial workplace question "which optimizer do you use?" with actual reasons.

## 2. Understanding through an analogy

**autograd = automatic corporate-card records.** Level04's numpy backprop was a "handwritten ledger": every expense (computation) we recorded ourselves, and at reconciliation (differentiation) we paged backwards through the ledger assigning responsibility. autograd is the corporate card. Pay (compute) with a `requires_grad=True` card and the transactions are recorded automatically; the moment you call `backward()`, reconciliation — tracing responsibility (derivatives) for every parameter — finishes automatically. And as we verified with numerical differentiation in level04, this automatic reconciliation is mathematically exact.

**An optimizer = a descent strategy.** autograd tells you the gradient (which way is downhill). The strategy is what you do with that information.

- **SGD**: the basic walk. Move by gradient × learning rate. Simple and honest.
- **SGD + momentum**: a walk with inertia. Keep heading the same way and you accelerate, coasting over shallow ripples.
- **Adam**: a hiker wearing a smartwatch. On top of a moving average of direction (momentum), it records the terrain's roughness per parameter and adjusts each stride individually — cautious on steep axes, brisk on gentle ones.

## 3. Key concepts

### 3-1. requires_grad and backward

```
x = torch.tensor(2.0, requires_grad=True)   # start recording
y = x**2 + 3*x                              # a computation graph is quietly built
y.backward()                                # sweep the graph backwards, differentiating
x.grad                                      # dy/dx = 7.0
```

Every time PyTorch computes, it records a graph of "what was made from what" (the computation graph). `backward()` walks this graph from output back to input — exactly like level04's backprop — applying the chain rule.

### 3-2. Gradients accumulate — why zero_grad exists

`backward()` **adds** to `grad` (it doesn't overwrite). Call it twice and you get 7 + 7 = 14. That's why the first line of every training step is `opt.zero_grad()`. Forget it and the gradients of past steps keep piling up — a "ghost bug" where the loss thrashes around.

### 3-3. The standard 5-step training loop

```
opt.zero_grad()      # 1. wipe the ledger clean
p = model(X)         # 2. forward pass
loss = loss_fn(p, y) # 3. grade it
loss.backward()      # 4. backward pass (level04's 30 lines)
opt.step()           # 5. weight update (level03's x -= lr*grad)
```

In every level and every lecture from here on, this 5-line skeleton never changes.

### 3-4. Optimizers carry state

SGD walks looking only at the gradient, but momentum and Adam store a "record of movement so far" internally. Adam keeps two moving averages per parameter (direction and magnitude), so it uses roughly 2x the model's parameter memory on top. Negligible for small models — but for billion-parameter models this optimizer state claims a large share of GPU memory. The number resurfaces in level11's memory estimate.

### 3-5. SGD vs Adam — when to use which

| | SGD (+momentum) | Adam |
|---|---|---|
| Stride | Shared across all parameters | Auto-adjusted per parameter |
| Learning-rate sensitivity | High (tuning required) | Low (default 0.001 usually fine) |
| Typical use | Squeezing final performance from vision models | The first-try default; the NLP/LLM standard |

Practical rule of thumb: **start with Adam** to get a baseline fast, then experiment with SGD+momentum+a schedule (level10) when it's time to squeeze out performance.

## 4. Hands-on — main.py

Run:

```bash
cd lecture08_deep_learning_foundations/level06_autograd_optimizers
python3 main.py
```

Output, in order:

- **[1]** Differentiates y = x² + 3x at x=2 with autograd → 7.0. Matches the hand calculation (2x+3) exactly.
- **[2]** Reproduces the accident of calling backward twice without zero_grad: grad accumulates to 14.
- **[3]** Defines the same 2-8-1 XOR network as level04 in one `nn.Sequential` line, then trains it from the same initial weights (fixed seed) with SGD (lr=0.5) and Adam (lr=0.05).
- **[4]** Comparison table: epochs to reach loss < 0.05 — SGD 199 vs Adam 43. Adam converges roughly 4–5x faster.

The heart of the code is the 5-step loop inside `train_xor()`. Compare with level04 and see for yourself that the 15-line `backward()` function has vanished entirely.

## 5. Try it yourself

1. **(Easy)** Change Adam's learning rate 0.05 → 0.001 → 0.5 and tabulate how the epoch reaching loss<0.05 changes. *Hint: Adam isn't a cure-all either — at values as large as 0.5 it can turn unstable.*
2. **(Medium)** Turn on momentum with `torch.optim.SGD(..., momentum=0.9)`. See where it lands between vanilla SGD and Adam. *Hint: it's just one extra momentum argument.*
3. **(Challenge)** Extend [1] to take partial derivatives of z = sin(x·y) + x² at x=1.0, y=2.0. Compare autograd's result against the hand calculation (∂z/∂x = y·cos(xy) + 2x). *Hint: create both tensors with requires_grad=True, call z.backward(), then look at x.grad and y.grad.*

## 6. Common mistakes

- **Missing zero_grad.** The single most common mistake. If the loss falls then starts thrashing, check this first.
- **Calling `loss.backward()` twice.** The computation graph is freed after backward by default. Calling it again on the same graph raises a RuntimeError — in a loop, redo the forward pass each step.
- **Wasting the ledger in evaluation code.** When only inferring, turn recording off with `with torch.no_grad():` to save memory and time (used in the next level).
- **Storing loss in a list without `.item()`.** Collecting raw tensors also holds their computation graphs and leaks memory. If you only need the number, `loss.item()`.

## Next level preview

All the parts have arrived. In level07 you'll learn the standard idiom of writing your own model class with `nn.Module`, then train an MLP on real business data (the churn_table subscription-churn dataset) and pit it against logistic regression.
