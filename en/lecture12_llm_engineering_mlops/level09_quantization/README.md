# Lecture 12 · Level 09 — Quantization and Model Compression

> Save the model at "decent quality" instead of the pristine original, and it gets 4–8x lighter — we implement quantization ourselves.

**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level08_dpo / **Estimated time** 50 min

## 1. Why Learn This — The Business View

Up to the previous level we followed the process of training a model. But the moment you try to put the finished model into a real service, you hit a very practical wall: size. A 7-billion-parameter model stored in fp32 (32-bit floating point) is about 28GB. It does not fit on most single server GPUs, and a laptop is out of the question.

Quantization attacks this problem head-on. Store each parameter in 1 byte (int8) or 0.5 bytes (int4) instead of 4, and the same model shrinks to 7GB or 3.5GB. The performance cost? Astonishingly, in most cases almost none. Open-source LLMs running on laptops, API prices falling year after year, AI assistants embedded in smartphones — all of it is thanks to quantization.

From a business standpoint, quantization is simply cost. If the same service runs on 1 GPU instead of 4, the infrastructure bill drops to a quarter. When the request comes in — "we want to run the model trained on our data on an in-house server" — this is the technology that decides whether it is feasible.

## 2. Grasping It Through an Analogy

An original photo from your phone is over 10MB. Send it through a messenger and it is automatically compressed to 500KB. Peer closely and there are subtle differences, but for everyday viewing you can barely tell. That is exactly quantization: re-saving the "photo" that is your weights at slightly lower resolution.

A more precise analogy is measuring with rulers. fp32 is a precision instrument that measures down to 0.0001mm; int8 is a ruler with 1mm gradations. You do not need 0.0001mm precision to measure a desk — with the 1mm ruler, the desk is still the desk. What matters is how far the ruler's range must stretch. To measure a 3m desk with a 30cm ruler you must stretch the gradations 10x, making each tick 1cm — and the error grows. This is the concept of scale (tick spacing) you will verify in the exercise.

And if you are measuring heights in a class where one student is 7'6"? Set the range around that one student and everyone else's heights get squashed together. Hence the idea "let each group use its own ruler" — that is block quantization.

## 3. Core Concepts

### 3.1 The Symmetric Quantization Formula

The most basic int8 symmetric quantization is exactly two lines.

```
scale   = max(|w|) / 127
q       = round(w / scale)      # integers from -127 to 127
restore = q × scale
```

Set the ticks so the weight with the largest absolute value lands on 127, then round every value to an integer multiple of that tick. To restore, multiply the integer by the scale again. The rounding difference is the quantization error, and the maximum error is half a scale.

### 3.2 Bits and Memory

| Format | Size per parameter | 7B-parameter model | Representable levels |
|---|---|---|---|
| fp32 | 4 bytes | 28 GB | effectively continuous |
| fp16 | 2 bytes | 14 GB | effectively continuous |
| int8 | 1 byte | 7 GB | 255 levels |
| int4 | 0.5 bytes | 3.5 GB | 15 levels |

Down to int8, performance loss is nearly zero; from int4 on there are only 15 ticks, and finesse is required.

### 3.3 Outliers and Block Quantization

Trained model weights occasionally contain unusually large outliers. With one scale per tensor, a single outlier stretches the ticks and wrecks the precision of the other tens of thousands of values. The fix: cut the weights into blocks of 32 or 64 and give each block its own scale. Storage for scales grows a little; the error shrinks a lot.

The GGUF format used by llama.cpp is precisely a deployment file format holding such block-quantized weights. Read a name like "Q4_K_M" as roughly "4-bit, block-quantization scheme K." In this exercise we build the idea ourselves in numpy.

### 3.4 Which Precision, When

- Training: fp16/bf16 (mixed precision) — gradient computation needs a certain amount of precision.
- Server inference: int8 — under half the cost with almost no loss.
- Personal PCs and edge devices: int4 — memory first, accepting slight quality loss.

## 4. Hands-On — main.py

How to run:

```bash
cd lecture12_llm_engineering_mlops/level09_quantization
python3 main.py
```

Stage-by-stage guide to the output:

- **[1]** Watch quantize→restore on a five-element array. See how the original 0.82 becomes the integer 80 and returns as 0.8189, and that the max error (0.0037) is under half a scale.
- **[2]** Compares storage sizes of fp32/fp16/int8/int4 for 1 million parameters.
- **[3]** Trains a character-level micro language model (28-character vocabulary, 28,988 parameters) on tiny_corpus for 400 steps. It finishes in seconds (loss ends around 0.15).
- **[4]** Compares the eval loss of a model whose weights were all quantized to int8 and restored against the original: fp32 0.1624 vs. int8 0.1622. The key observation: the difference is down in the fourth decimal place (here rounding luck even nudged it a hair *lower*) — essentially no damage.
- **[5]** Shows the loss increase when shrinking further to int4 (0.1649, +0.0025), then compares restoration error on an outlier-laden array between one whole-tensor scale (0.03890) and 64-value blocks (0.00645). The block scheme is 6x more accurate.
- **[6]** Wraps up with the size/loss summary table.

The core of the code is the `quantize()` function: one line `scale = |w|.max() / 127` and one line `round(w / scale)` — that is all. Note how simple the skeleton of this seemingly enormous technology is, and how `apply_quant()` measures the performance impact by applying "quantize→restore" to every model parameter — a clean experimental method.

## 5. Try It Yourself

1. **(Basic)** Change `bits` in `quantize(w, bits)` to 2 and experiment with int2 quantization. With only 3 ticks, watch how badly the loss collapses. Hint: change `quantize(a, 4)` in stage [5] to `quantize(a, 2)`.
2. **(Intermediate)** Vary the block size across 16, 64, and 256 and tabulate how the outlier array's restoration error changes. Hint: smaller blocks are more accurate but store more scales — compute that trade-off too (one fp32 scale per block = 4 bytes).
3. **(Challenge)** We currently use the raw maximum as the scale. What happens to the error if you clip outliers at the 99.9th-percentile value before quantizing? Implement it with `np.percentile` and compare. Hint: the outliers themselves take big errors, but the other 99.9% become accurate.

## 6. Common Mistakes

- **Assuming quantization means integer arithmetic too.** Schemes like this exercise — "store as integers, compute in floats after restoring" — are widely used. Genuine integer compute needs hardware support.
- **Rounding without a scale.** Plain `round(w)` zeroes out every weight below 0.5 and destroys the model. Always set the ticks (scale) first.
- **Trusting int4 because int8 was lossless.** Every bit removed halves the number of ticks. From int4 down, corrective techniques like block quantization are effectively mandatory.
- **Skipping the post-quantization measurement.** Sensitivity differs by model and data. Build the habit of comparing before/after on the same metric, as in [4]–[5].
- **Confusing fp16 with int8.** fp16 is still floating point (tick spacing varies with the value's magnitude); int8 has fixed ticks. The compression principles differ.

## Next Level Preview

The model is light — time to send it into the world. In the next level we wrap the model file in an HTTP API server that actually receives requests and responds. We measure first-hand why p95 latency matters more than the average, and how concurrent requests change throughput.
