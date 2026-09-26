# Lecture 12 · Level 05 — Scaling Laws and Estimating Training Costs

> Understand the triangle of model, data, and compute — and build an estimator that answers "what would training this model cost?" with actual numbers.

**Difficulty** ⭐⭐⭐ / **Prerequisites** Level 04 (mini GPT pre-training) / **Estimated time** 45 min

## 1. Why Learn This — The Business View

In Level 04 we trained a 108k-parameter model in about 12 seconds. So why do real large models train on thousands of GPUs for weeks, and how is that cost computed? Answering that question is this level. Remarkably, LLM training cost can be estimated on a napkin. You need only two formulas — **compute ≈ 6 × parameters × tokens**, and **optimal tokens ≈ about 20 × parameters**.

This intuition turns into money immediately. Handed a "build our own LLM" proposal, you can sanity-check the GPU hours and cloud cost just by asking for the parameter count and training token count. If the quote differs from your ballpark by more than 10x, something is off. And when you explain to leadership "why we should fine-tune instead of pre-train," this level's numbers (pre-training in the millions of dollars vs. fine-tuning in the hundreds) are the most persuasive evidence there is. Your reading of AI industry news changes too — an article saying "Company X is building a 100,000-GPU cluster" becomes something whose implied model scale you can compute yourself.

## 2. Grasping It Through an Analogy

**Training a model is like a construction estimate.** When the client asks "how much?", the builder ballparks cost per square foot × floor area. Not exact, but plenty for judging scale. The LLM's cost-per-square-foot formula is 6ND — each parameter takes about 6 multiply-adds to learn from one token; multiply it all out and you get the total workload (FLOPs). Divide the workload by one worker's daily output (a GPU's operations per second) and you get the schedule; multiply schedule by labor rates and you get the estimate.

**The Chinchilla law is the balance between a student and the reading list.** Give a brilliant student (a big model) a single thin workbook (little data) and the talent is wasted. Give an average student (a small model) the whole library (lots of data) and, past a point, more reading stops raising grades — the student's own ceiling binds. The balance point researchers found through many experiments is "about 20 tokens per parameter." With a fixed budget (study hours), grades are best when student caliber and reading volume are matched at this ratio.

**GPU hours are billed like a rental car.** Cloud GPUs have a fixed hourly rate: total GPU hours × hourly rate = cloud cost. On your own servers, electricity takes its place (GPU power draw × hours × electricity rate, with cooling piled on top).

## 3. Core Concepts

### 3-1. The Compute Formula — FLOPs ≈ 6ND

Learning from one token costs each parameter about 2 floating-point operations in the forward pass and about 4 in the backward pass. So total compute is roughly:

$$\text{FLOPs} \approx 6 \times N(\text{parameters}) \times D(\text{tokens})$$

Training a 70-billion-parameter model on 1.4 trillion tokens gives 6 × 7×10¹⁰ × 1.4×10¹² ≈ 5.9×10²³ FLOPs. If a human did one operation per second, it would take over a million times the age of the universe.

### 3-2. GPU Hours and MFU

Numbers like "989 TFLOPS" on a spec sheet are theoretical peaks. In real training you harvest only 30–45% of that, thanks to data stalls, communication, and memory bottlenecks. That effective fraction is called **MFU (Model FLOPs Utilization)**.

$$\text{GPU hours} = \frac{\text{FLOPs}}{\text{GPU peak} \times \text{MFU}}$$

Engineering MFU from 30% up to 40% saves millions of dollars outright — which is why big labs pay infrastructure engineers so well.

### 3-3. Converting to Money — Electricity and Cloud Rates

| Item | How to compute |
|---|---|
| Energy (kWh) | GPU power (W) × GPU hours × PUE ÷ 1000 |
| Electricity cost | energy × rate ($/kWh) |
| Cloud cost | GPU hours × hourly rental ($/hr) |

PUE (power usage effectiveness) folds in cooling and facility overhead, typically 1.1–1.5. Cloud rental costs far more than raw electricity because it includes hardware depreciation and margin — hence companies that run GPUs year-round buy, and occasional users rent.

### 3-4. The Chinchilla Intuition — a Fixed Budget Fixes the Optimal Size

With the compute budget C fixed and C = 6ND, growing the model (N↑) forces less data (D↓). Research found the optimal balance near D ≈ 20N; substituting gives **N_opt = √(C/120)**. By this yardstick, early large models were "oversized relative to their data," and the law is why later generations shrank parameters and grew tokens. Note this optimum considers *training* cost only. Once you count inference cost after deployment, "over-training" a small model far beyond 20x can win — which is why today's open models sometimes train on hundreds of tokens per parameter.

### 3-5. Scaling Laws — Loss Is Predictable

Knowing model size N and token count D, the final loss can be predicted with the approximation L(N, D) = 1.69 + 406.4/N^0.34 + 410.7/D^0.28 — that is the power of scaling laws. Fit this curve with several small experiments, then **predict the result of the multi-million-dollar main run before training it**. This is what turns large-scale training from a gamble into engineering.

## 4. Hands-On — main.py

How to run:

```bash
cd lecture12_llm_engineering_mlops/level05_scaling_laws_costs
python3 main.py
```

- **[1] The basic formula** — prints 6ND and this estimate's assumptions (MFU 35%, electricity $0.12/kWh, PUE 1.3).
- **[2] Estimates by scale** — quotes FLOPs, GPU hours, wall-clock days, electricity, and cloud cost for our TinyGPT (0.02 GPU-seconds!), then 120M/7B/70B-parameter models. The 70B model comes out to about 9.6 days on 2,048 H200s and roughly $1.65 million in cloud rental. Add failed experiments, data preparation, and headcount, and the real project cost is several times that.
- **[3] Chinchilla arithmetic** — computes the optimal parameter and token counts for budgets of 10²¹–10²⁴ FLOPs (e.g., 10²² FLOPs → 9.1B parameters, 182.6B tokens, predicted loss ≈ 2.146).
- **[4] Scaling curves** — saved to `outputs/scaling_curves.png`. What to look for: at small budgets the 100M model (blue) reaches the lowest loss, but as the budget grows the curves **cross** and the 1B (orange), then 10B (green) models take over. The dashed line is the reachable limit at each budget (the Chinchilla frontier).

The code's core is the four lines inside `estimate()` (6ND → GPU hours → kWh → dollars), and the part of `draw_scaling_chart()` that fixes the budget C and back-solves the tokens as D = C/6N.

## 5. Try It Yourself

1. **Estimate your own scenario** — what would a 3B-parameter model trained at the Chinchilla ratio cost? Add `estimate("Our company", 3e9, 6e10, gpu="H200", n_gpus=64)` and run. (Hint: tokens = 20 × 3B = 60B.)
2. **MFU sensitivity** — change `MFU` to 0.2 and 0.45 and compare the 70B model's estimates. (Hint: cost is inversely proportional to MFU. You will feel why efficiency work is money.)
3. **Electricity by country** — try rates of $0.05 (a cheap-power country) and $0.25 (an expensive one). The numbers show why data centers get sited where power is cheap. (Hint: change the single `ELEC_USD_KWH` constant.)

## 6. Common Mistakes

- **Using the spec-sheet TFLOPS as-is** — skip MFU and your estimate comes out about 3x too optimistic. Theoretical and effective are different.
- **Mistaking training cost for total cost** — the main run is the tip of the iceberg. Failed experiments, hyperparameter searches, data-cleaning headcount, and evaluation/alignment costs stack up to several multiples.
- **Treating Chinchilla's 20x as absolute law** — it is only the *training-compute* optimum. Counting inference, over-training a small model is often the right answer.
- **Extrapolating scaling curves forever** — the approximation holds inside the experimented range. Change the data quality or the architecture and the coefficients change too.
- **Forgetting inference cost** — training happens once; inference (serving) happens every day. For a popular service, cumulative inference cost soon overtakes training cost (covered in Level 10).

## Next Level Preview

You cannot redo a multi-million-dollar pre-training run every time. So 99% of practice goes the other way: "cheaply refine a model that is already trained." In Level 06 we implement the flagship technique — **LoRA (low-rank adapters)** — and attach it to Level 04's TinyGPT ourselves, then check how far the trainable parameter count falls compared to full fine-tuning.
