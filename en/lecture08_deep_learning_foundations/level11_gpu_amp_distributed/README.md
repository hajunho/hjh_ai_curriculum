# Lecture 08 · Level 11 — GPUs, Mixed Precision, and Distributed Training Concepts

> Why deep learning runs on GPUs, why the "calculators with fewer digits" (fp16/bf16) appeared, and how a model that doesn't fit on one GPU is carved up (DDP/FSDP) — understood not through code execution but through **cost estimation**.

**Difficulty** ⭐⭐⭐⭐⭐ / **Prerequisites** level10 (learning rates and stabilization) / **Estimated time** 50 min

## 1. Why learn this — the business view

The goal of this level is not to operate a GPU cluster yourself. It's **not getting fooled in infrastructure budget meetings**. "How many GPUs do we need to fine-tune an 8B model?" "Vendor A says they trained in bf16 — what does that mean?" "Does FSDP reduce cost?" — the difference between the person who answers these with grounded numbers and the person who answers on instinct quickly becomes a difference of tens of thousands of dollars in the estimate. Fortunately, the core calculation is a single multiplication: **parameter count × bytes + activations**. In this level you do that multiplication yourself. The intuition you build here is reused as-is in lecture12 (LLM) when judging API costs and fine-tuning strategies.

## 2. Understanding through an analogy

- **CPU vs GPU = a few PhDs vs thousands of arithmetic clerks.** CPU cores are an elite handful of PhDs, good at complex branching and judgment. A GPU is thousands of clerks who only multiply and add. Over 90% of deep-learning computation is matrix multiplication — "simple multiplications, in bulk, simultaneously" — the arithmetic army's winning event.
- **fp16/bf16 = calculators with fewer digits.** fp32 is the calculator with generous digits (4 bytes); fp16 has half the digits (2 bytes). Half the price and twice the speed, but errors creep in with very large or very small numbers. bf16 is the same 2 bytes with the representable range (exponent) kept as wide as fp32 at the cost of fewer precise digits (mantissa) — robust to numbers that swing wildly in magnitude, like gradients, which made it the standard for large-model training.
- **Mixed precision (AMP) = shorthand for the arithmetic, fair copy for the master ledger.** Fast computation (multiplications) happens in half precision, while the master records that can't accumulate mistakes (the master copy of the weights, accumulations) stay in fp32. With fp16, you also apply loss scaling — deliberately inflating the loss and deflating it back — so tiny gradients don't vanish to 0.
- **DDP = everyone carries a copy of the same load, but different boxes.** When the model (the load) fits on one card, each GPU copies the model, splits only the data (the boxes), and averages the gradients. The goal is **speed**.
- **FSDP/ZeRO = splitting the load itself.** When the model doesn't fit on one card, the weights, gradients, and optimizer state are sliced into shards that the GPUs store between them, gathering the shards only at the moment computation needs them, then scattering again. The goal is **breaking through the one-card limit**.

## 3. Key concepts

### 3-1. Computation grows as a cube

An (n,n) matrix multiplication costs about 2n³ FLOPs. Make n 8x bigger and the computation grows 512x — widen a model "a little" and the math explodes. This is why GPUs are a necessity, not a choice.

### 3-2. Inference memory vs training memory

- **Inference**: only the weights are needed. 8B × 2 bytes (bf16) ≈ 15GB — one large GPU's worth.
- **Training**: weights + gradients + **Adam state (m, v)** + activations are all required. By the conventional approximation, **16 bytes per parameter** (the standard arithmetic of the ZeRO-paper lineage): fp32 is 4+4+8; mixed precision is fp16 weights 2 + grads 2 + fp32 master 4 + Adam 8, still 16. In other words, **even with AMP, the "storage" memory barely shrinks** — the gains come from compute speed (~2x), halved activations, and halved communication. The cost flagged in level06 — "Adam stores two moving averages per parameter" — arrives here as the invoice.

### 3-3. The 8B training estimate in one line

8e9 × 16 bytes ≈ 119GB (state) + activations 10–20GB ≈ **130–140GB** → doesn't fit on an 80GB GPU → distribution required. These three lines are this level's key deliverable.

### 3-4. Choosing between DDP and FSDP

| Situation | Choice | Memory/card | Effect |
|---|---|---|---|
| Model fits on 1 card, want more speed | DDP | Unchanged (full copy) | Throughput ~× card count |
| Model doesn't fit on 1 card | FSDP/ZeRO | State ÷ card count + activations | Large-model training becomes possible |

FSDP adds communication, so per-card efficiency is lower than DDP. The first-order rule: "fits → DDP, doesn't fit → FSDP."

## 4. Hands-on — main.py

Run:

```bash
cd lecture08_deep_learning_foundations/level11_gpu_amp_distributed
python3 main.py
```

Output, in order:

- **[2]** A CPU matmul benchmark (n=128–1024), tabulating time and GFLOP/s. 8x the n means 512x the computation — this CPU manages hundreds to thousands of GFLOP/s, while a modern GPU does hundreds of TFLOP/s in the same event.
- **[3]** A per-precision memory calculator: storage for 0.1B/1B/8B/70B models in fp32/fp16/int8. You'll be able to produce numbers like "70B in fp16 = 130GB" on the spot.
- **[4]** The 8B training-memory estimate: fp32 139GB vs AMP 129GB. The answer to "why doesn't AMP shrink it much?" (Adam state, 16 bytes/parameter) appears in the output.
- **[5]** The per-card burden table when sharding across N cards with FSDP: 1 card 139GB (impossible) → 2 cards 79.6GB (barely under 80GB) → 8 cards 34.9GB (comfortable). The table that grounds a GPU purchase-quantity decision.
- **[6]** Shows, in comments, how few lines AMP/DDP/FSDP take in real code (not executed here — this exercise is CPU-only).

## 5. Try it yourself

1. **(Easy)** Add model sizes you know (e.g., 0.5B, 3B, 32B) to the calculator in [3]. *Hint: just append one tuple to the list in main().*
2. **(Medium)** If the activation estimate (act) goes 20GB → 40GB (assuming long context and large batches), how does the "2 cards OK" verdict in [5] change? *Hint: the trap is that activations are a per-card burden that does not divide by card count. In practice, activation checkpointing trades time for activation memory.*
3. **(Challenge)** Add an inference-only estimator: weights (bf16) + KV cache (context length × layers × ... simplified as 20% of the parameters). Print how many GB 8B-model inference needs and how many times cheaper it is than training. *Hint: it's roughly the multiplication 8e9×2/1024³ × 1.2.*

## 6. Common mistakes

- **Budgeting training with an inference estimate.** "8B = 15GB, so one card, right?" — that's inference. Training jumps ~9x (16 bytes + activations vs 2 bytes).
- **Treating fp16 and bf16 as the same.** Same 2 bytes, but fp16's narrow range makes loss scaling mandatory, while bf16's wide range is easier to handle. Whether the hardware supports bf16 is also a checklist item.
- **Expecting GPU count = speed multiplier.** Communication costs mean 8 cards don't give 8x. FSDP in particular has heavy gather/scatter traffic, and the network (NVLink etc.) can become the bottleneck.
- **Estimates that forget activations.** Compute only the state memory and a slightly larger batch triggers OOM (out of memory). Always add "activations + 10–20% headroom" to the estimate.
- **Believing distribution = universal savings.** FSDP divides memory; it doesn't cut the total bill. Total GPU-hours can actually increase — the purpose is "turning impossible into possible."

## Next level preview

Lecture08 is complete. Starting from the perceptron, you wrote backpropagation from scratch, built PyTorch training loops and stabilization techniques, and made GPU estimates — the entire common foundation of deep learning. In the next lecture, **lecture09 Computer Vision**, we place one new part — the convolution — on top of this foundation, learn how computers "see" images, and build a shape-image classifier ourselves.
