# Lecture 12 · Level 06 — Fine-tuning and LoRA

> Leave the already-trained model's brain untouched and teach it a new job by sticking on thin "sticky-note adapters" — we implement the technique ourselves.

**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level04, level05 / **Estimated time** 60 min

## 1. Why Learn This — The Business View

As we computed in level05, training a large model from scratch costs millions of dollars. But what we actually want at work is usually a small adjustment: "answer in our company's tone," "use our product terminology." Retraining the entire model for that (full fine-tuning) is expensive, needs several times the GPU memory, and forces you to store a separate multi-hundred-GB model copy per client.

LoRA (Low-Rank Adaptation) is the technique that solved this. It leaves the original weights completely untouched and trains only two tiny "correction matrices." Trainable parameters shrink to 1–3% of the total, which is how fine-tuning multi-billion-parameter models on a single consumer GPU became possible. Most corporate LLM customization today is done with the LoRA family (including QLoRA). When someone asks "please tune the model on our data," you need this principle to judge the quote and the approach.

## 2. Grasping It Through an Analogy

Imagine a veteran with 20 years of experience (the pre-trained model) joins your company. This person is superb at general work but does not know your company's report format or sign-off style.

- **Full fine-tuning** = sending this person back to college for re-education. They would learn the house style, but the time and cost are enormous — and done badly, they may forget what they were already good at (catastrophic forgetting).
- **LoRA** = sticking a few sticky notes on their monitor that say "here's how we write it at this company." Their own knowledge (the original weights) is intact; only the sticky notes (adapters) are added. Transfer them to another department and you just peel off the notes and stick on different ones.

Why do such small sticky notes work? That is LoRA's central insight: an adjustment like "switch to the company tone" is really a handful of simple rules ("made → hath made"), so a **small, low-dimensional correction** suffices — no gigantic overhaul needed.

## 3. Core Concepts

### 3-1. Low-Rank Decomposition — a Small Edit to a Big Matrix

Full fine-tuning changes a weight matrix W (say 4096×4096, about 16.78 million numbers) into W + ΔW. LoRA's discovery: this change ΔW is effectively "low-rank" — expressible with far less information. So we replace ΔW with the product of two thin matrices.

```
y = W·x  +  (α/r)·B·A·x        # A: r×4096, B: 4096×r, r around 4–64
```

With r=8, the numbers to train are 4096×8×2 ≈ 65k — 0.4% of the original. A is initialized with small random values and B with zeros, so right after attachment B·A = 0: the model starts from *exactly* the original outputs. As training proceeds, B·A gradually learns the "correction."

### 3-2. Where to Attach It

By convention, on the attention Q·K·V and output-projection linear layers (in our exercise: qkv and proj). Attaching to the MLP layers too adds expressiveness — and parameters; it is a trade-off.

### 3-3. The Hyperparameters r and α

- **r (rank)**: the size of the sticky note. Bigger means more expressive but more parameters. Light tasks like tone correction use r=4–8; injecting new domain knowledge uses r=32–64.
- **α (alpha)**: the strength multiplier of the correction term. Usually α = 2r, keeping the effective multiplier α/r = 2.

### 3-4. Merging and Deployment

After training there are two deployment options. (1) **Merge**: precompute W' = W + (α/r)·B·A into a single weight — inference then costs nothing extra. (2) **Keep adapters separate**: serve one original model and hot-swap per-client adapters (tens of MB). An adapter file is under 1% of the original (tens of GB), so customizing for 100 clients raises no storage worries.

### 3-5. One Step Further — QLoRA

QLoRA, which you will hear about constantly in practice, is the combination "quantize the frozen original weights to 4 bits (level09 teaches quantization), then attach LoRA on top." It is what made tuning large models on laptop-class GPUs possible.

## 4. Hands-On — main.py

Run:

```bash
cd lecture12_llm_engineering_mlops/level06_finetuning_lora
python3 main.py
```

The following flow prints in about 10 seconds.

- **[1]–[2]** Pre-train a 2-layer mini GPT (107,776 parameters) for 400 steps on hjh_data.tiny_corpus() (plain past-tense sentences: "made", "fixed"…). Check that the generated sample comes out in the plain register: `'This evening a chef prepared a pot of stew. ...'`.
- **[3]** Measure the loss on new-style data (the same sentence frames, but with verbs in an archaic "hath made" register). It is about 1.20 — the base model does not know the 'hath' register.
- **[4]** Freeze the whole model (requires_grad=False), then wrap each block's qkv and proj in `LoRALinear`. Watch the key numbers in the output: full fine-tuning would retrain 107,776 parameters; LoRA trains only 3,072 (2.85%).
- **[5]** Training just the LoRA for 300 steps drops the new-style loss from 1.197 to 0.186, and the generated sample's verbs switch to the archaic register: `'This evening a chef hath made a program. Today a teacher hath prepar'`. And yet not a single base weight changed.
- **[6]** Merge check: `torch.allclose` confirms that a plain linear layer built as W + scale·B@A produces exactly the same output as the adapter-wrapped layer.

The most important part of the code is `LoRALinear.forward`: `self.base(x)` (the frozen original) plus `scale * (x @ A.T @ B.T)` (the trainable correction) — that one line is all of LoRA.

## 5. Try It Yourself

1. **Rank experiment**: change `r=4` in [4] to `r=1` and `r=16`. Does r=1 still manage the register swap? (Hint: swapping verb endings is a very simple transformation, so a very low rank may suffice. Tabulate trainable parameters versus final loss.)
2. **Attachment-point experiment**: attach only to qkv and leave proj original. How much does the resulting loss change? (Hint: comment out one of the two lines.)
3. **Check catastrophic forgetting**: after LoRA training, also measure eval_loss on the original pre-training data (base_data). Then compare with the base_data loss after full fine-tuning instead (300 steps on style_data with nothing frozen). Which preserves the original skill better? (Hint: also consider that peeling off the LoRA adapter restores the original 100%.)

## 6. Common Mistakes

- **Not initializing B to zero**: if both A and B start random, the model's output is broken at the moment of attachment and training starts from a damaged state. "Start from the original" is the core of LoRA's stability.
- **Forgetting to freeze**: without freezing the base you have just done full fine-tuning with adapters on top, and LoRA's advantage evaporates. Make it a habit to print the trainable-parameter count before training.
- **Using the pre-training learning rate as-is**: LoRA has few parameters and typically uses a larger learning rate than full fine-tuning (e.g., 1e-4 vs 1e-5). The difference is small in our mini exercise but matters in production.
- **Saving the whole model when only the adapter is needed**: LoRA's payoff is the tens-of-MB adapter file. Standard practice is to pick out just A and B from the state_dict and save those.

## Next Level Preview

LoRA changed the "tone," but the model still merely continues text — it does not answer questions. In level07, instruction tuning (SFT), we use a chat template and loss masking to build "a model that follows instructions."
