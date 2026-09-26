# Lecture 12 · Level 00 — What It Means to Build a Model — The Big Picture

> A one-glance map of the production line a model like ChatGPT travels down — with each stage's inputs, outputs, and costs.

**Difficulty** ⭐ / **Prerequisites** none (lecture11 recommended) / **Estimated time** 30 min

## 1. Why Learn This — The Business View

In the lectures so far you learned how to *use* an LLM (large language model). From this lecture on, we flip the perspective and reproduce, in miniature, the process by which such a model is *made*. Why should a business professional without a CS degree go this far?

First, decision-making. When someone proposes "let's build our own company model," you cannot draw up a budget or a timeline unless you can tell whether that means a pre-training run costing tens of millions of dollars or a fine-tuning job costing a few hundred. In practice, most of what companies call "our own LLM" is a public model with a few of the later stages layered on top.

Second, collaboration. When the AI team says "we're in SFT now, DPO starts next week," knowing which station of the factory line that corresponds to — and what it produces — completely changes the quality of the conversation.

Third, the job market. Non-engineering professionals who can explain LLM engineering even at "map level" are still rare. This level is where we draw you that map.

## 2. Grasping It Through an Analogy

Making an LLM is astonishingly similar to running a car factory.

- **Pre-training** is the steel mill. You feed iron ore (trillions of tokens of raw web text) into the furnace and smelt it for months to produce steel (the fundamentals of language). Most of the total cost is spent here, and if you botch it, you start over from scratch.
- **Annealing** is the heat-treatment step. At the very end of training, you finish slowly with only textbook-grade, high-quality data — the same steel comes out much harder.
- **SFT (instruction tuning)** is the assembly line. Just as a slab of steel is not yet a car, a model fresh out of pre-training can only "continue text" — it does not follow instructions. Only after assembly according to the manual (instruction-response examples) does it start looking like a product.
- **Preference alignment (DPO)** is the quality-control (QC) line. Between two answers, you pick the one customers liked better and polish the finish quality.
- **Quantization** is the packing line. Just as a finished car must be shrunk down to fit a shipping container, a 14GB model is compressed to 4GB so it can be delivered.
- **Serving** is the distribution center. From here on you are no longer in the world of "build cost" but of "operating cost" that scales with order volume.

Memorize these six stations in order. Levels 01–11 of this lecture follow exactly this line.

## 3. Core Concepts

### 3-1. The Two Great Phases: Pre-training vs. Post-training

- **Pre-training**: the model is made to do just one thing — "guess the next token" — on internet-scale text. The resulting base model knows the statistical structure of language but has no concept of answering a question.
- **Post-training**: the umbrella term for SFT plus preference alignment. The data is tens to hundreds of thousands of examples — a ten-thousandth of pre-training — yet most of the "it understands me" feeling users experience is built here.

### 3-2. A Sense of Cost — The 6 × N × D Formula

The compute (FLOPs) needed for training is approximated as **6 × parameter count (N) × token count (D)**. Training a 7B (7-billion-parameter) model on 2 trillion tokens gives about 8.4×10²² FLOPs. Divide by the effective throughput of one H200-class GPU and you get tens of thousands of GPU-hours — a couple hundred thousand dollars at cloud rates. SFT, computed with the very same formula, comes out to a few GPU-hours and pocket change. The key takeaway of this level: **each stage differs in cost by a factor of about 10,000**.

### 3-3. Semi-finished Goods Called Checkpoints

Each station's output is a file called a checkpoint. The names run base → base-annealed → sft → chat (alignment complete) → int4 (quantized), and the `-base`, `-instruct`, `-int4` suffixes you see on model hubs like Hugging Face are exactly these tags telling you where on the line a model came from.

### 3-4. Why Most Companies Start in the Middle

Because pre-training accounts for over 95% of total cost, the industry-standard strategy is to take a public base model (free) and perform only SFT/DPO/quantization. The hands-on work in this lecture has the same structure: in level04 we build a mini base model once, and every later level is a stage layered on top of it.

| Stage | Data scale | Cost order of magnitude |
|---|---|---|
| Pre-training | trillions of tokens | hundreds of millions to hundreds of billions of KRW (hundreds of thousands to hundreds of millions of dollars) |
| Annealing | tens of billions of tokens | millions to tens of millions of KRW |
| SFT | tens to hundreds of thousands of examples | tens of thousands to millions of KRW |
| DPO | tens of thousands of pairs | tens of thousands to millions of KRW |
| Quantization | no training | thousands to tens of thousands of KRW |

## 4. Hands-On — main.py

Run it like this:

```bash
cd lecture12_llm_engineering_mlops/level00_the_big_picture
python3 main.py
```

The program simulates a hypothetical 7B model passing through the six stations of the factory line. For each stage `[1]`–`[6]` it first shows the **analogy, input, and output**, and for training stages the `train_cost()` function converts 6 × N × D into FLOPs → GPU-hours → dollars, in that order (with a secondary figure in Korean won, KRW, at an assumed rate of 1,400). It also mimics the loss dropping a little at each stage (seed fixed): pre-training falls dramatically from 10.5 to about 2.4, while later stages move only slightly — the typical shape of real training logs.

In the final `[7]` summary, confirm that pre-training makes up more than 97% of the total build cost (in this run: $206,229 of $211,406, or 97.6%). Three spots in the code deserve attention: the spec constants at the top (PARAMS, PRETRAIN_TOKENS, …) are the experiment knobs you should try turning, `train_cost()` is the cost approximation we will reuse throughout this lecture, and the stages list doubles as the table of contents for the entire lecture.

## 5. Try It Yourself

1. **(Easy)** Change PARAMS to 70e9 (70B) and run again. Watch how the total cost scales and how the pre-training share changes. (Hint: the formula is linear in N, so training cost becomes exactly 10x.)
2. **(Medium)** Change GPU_PRICE to your local cloud provider's rate, and lower the MFU (effective utilization) from 0.4 to 0.25. Same model — how far apart are the quotes now? (Hint: the MFU is the 0.40 inside the GPU_FLOPS constant. These two variables are why real-world quotes differ from company to company.)
3. **(Challenge)** Add a "Safety evaluation (red-teaming)" stage to the stages list, just before serving. Inventing the input, output, and analogy yourself is a great way to internalize the pipeline structure. (Hint: look at the stages with tokens=None to see how to add a stage with no training.)

## 6. Common Mistakes

- **Confusing "fine-tuning" with "building a model"**: when the news says a company "developed its own model," ask which station they actually started from. Usually it is only SFT and later.
- **Lumping all costs together**: "training an LLM costs a fortune" applies only to pre-training. The later stations fit a laptop-scale budget.
- **Mistaking quantization for a kind of training**: quantization merely changes the numeric representation of a finished model. Nothing new is learned, so it is done in hours.
- **Treating deployment as "done"**: the moment you deploy, the cost structure flips from fixed build cost to usage-proportional operating cost. If a quote has no line item for serving, that quote is incomplete.

## Next Level Preview

The factory's first gate is preparing the raw material. In level01 we build, from the ground up, the machine that chops text into pieces a model can eat: the **BPE tokenizer**. You will see how a simple rule — merging the most frequent adjacent character pairs — grows into a vocabulary, and why the way you split numbers can decide a model's math ability.
