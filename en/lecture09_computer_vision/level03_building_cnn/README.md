# Lecture 09 · Level 03 — Building a CNN Architecture

> We read Conv2d's channels, kernel, stride, and padding like an assembly manual, build a mini CNN, and trace how the data's shape changes at every layer.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level02, lecture08 level05–07 (PyTorch tensors, MLPs) / **Estimated time** 45 min

## 1. Why This Matters — the Business View

Evaluate a vision-model rollout and you will hear things like "the backbone is such-and-such-Net, starting with 3×3 conv at 32 channels…". If you cannot read an architecture spec, you cannot estimate the model's size (cost), speed (response time), or required resolution (camera specs) yourself — you depend entirely on the vendor's word.

Reading CNN architectures comes down to exactly two skills. **(1) How does the data's shape change as it passes each layer? (2) How many parameters does each layer have?** Once you can compute these two, any architecture diagram reads like a cost estimate. Today we learn the arithmetic and verify it in code.

## 2. An Analogy

One CNN layer is **an inspection team holding several stamps**.

In level02 there was one stamp (kernel). A real Conv2d layer holds **several**: a vertical-edge stamp, a horizontal-edge stamp, a rounded-corner stamp… With 16 stamps you get 16 inspection-result maps (feature maps). That "number of maps" is **out_channels**.

The next layer's inspection team examines not the original image but **the bundle of 16 maps the previous team produced**. So the next layer's stamp is a solid stamp that sees through all 16 sheets at once (3×3×16). That is what **in_channels** means. As layers deepen, the inspections climb in abstraction: "combinations of edge maps" → "corner and curve maps" → "shape-part maps".

Finally, after all the maps have been reviewed, you need an **approval chain** that renders the verdict. Flatten the maps into one row and hand them to an ordinary network (Linear), and out comes the conclusion: "with this combination of evidence, it's a circle".

## 3. Core Concepts

### 3.1 Conv2d's five settings

`nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)`

| Setting | Analogy | Effect |
|---|---|---|
| in_channels | Number of maps in the incoming bundle | Must match the input's channel count |
| out_channels | Number of stamps (questions) | Number of output maps = expressiveness |
| kernel_size | Stamp size | Neighborhood seen at once |
| stride | Stamp step length | 2 halves the output height and width |
| padding | Margin around the document edge | 1 preserves size with a 3×3 kernel |

### 3.2 The output-size formula — the equation you will use most in this lecture

```
out = (in + 2*padding - kernel) / stride + 1
```

Example: input 16, kernel 3, padding 1, stride 1 → (16+2-3)/1+1 = **16** (preserved).
Switch to stride 2 → (16+2-3)/2+1 = **8** (halved). Every time you design layers, trace this formula all the way through — that is how you know the size just before Flatten (= the Linear layer's input size).

### 3.3 Counting parameters — pricing the model

- **Conv2d**: `out_ch × (in_ch × k × k) + out_ch` (the last term is the biases). Example: 1→8 channels, 3×3 gives 8×9+8 = **80**.
- **Linear**: `out × in + out`. Example: 256→3 gives 3×256+3 = **771**.

Note the key fact: a Conv layer's parameters are **independent of image size**. The stamp stays the same stamp no matter how big the document grows. A post-Flatten Linear, by contrast, scales directly with image size. That is why most of a CNN's parameters often pile up in the trailing Linear layers, and shrinking that (pooling, global averaging) connects directly to level04's topic.

### 3.4 Activation functions and stacking layers

Convolution is a linear operation — nothing but multiply-adds — so without a nonlinearity (ReLU) in between, any stack of layers collapses to a single layer (a review of lecture08 level02). Hence the basic block is `Conv → ReLU`, and a typical small CNN looks like this.

```
input (1, 16, 16)
Conv(1→8, 3x3, pad 1) + ReLU     → (8, 16, 16)
Conv(8→16, 3x3, pad 1, stride 2) + ReLU → (16, 8, 8)
Conv(16→32, 3x3, pad 1, stride 2) + ReLU → (32, 4, 4)
Flatten                           → (512,)
Linear(512→3)                     → (3,)  # class scores
```

Channels grow (8→16→32) while space shrinks (16→8→4) — "summarize the where, enrich the what" is the basic rhythm of CNN design.

### 3.5 The batch dimension

A PyTorch image tensor is 4-dimensional: `(batch, channels, height, width)`. Processing 32 images at once means `(32, 1, 16, 16)`. Most shape errors come from violating this 4-D convention, so drill the order "B-C-H-W" until it is second nature.

## 4. Hands-On — main.py

Run it:

```bash
python3 main.py
```

- **[1]** Builds a single Conv2d layer and dissects its settings and the weight tensor's shape `(out_ch, in_ch, k, k)`. "8 stamps, each 1×3×3" becomes visible as a shape.
- **[2]** Turns the output-size formula into a Python function, prints a table across stride/padding combinations, then pushes a tensor through a real Conv2d to verify that **formula and measurement agree**.
- **[3]** Defines a mini CNN (`TinyCNN`) and feeds a batch of shape images through it, printing **how the shape changes after every layer**. The entire journey from a (16,16) image to a (3,) score is one table.
- **[4]** Prints each layer's hand-computed parameter count next to PyTorch's tally to cross-check. See which layer the parameters pile up in.
- **[5]** Compares parameter counts against an MLP processing the same input (pixels simply flattened into Linear). Thanks to stamp reuse (weight sharing), the CNN is far lighter — confirmed in numbers.

No training yet (the weights are random initial values). Today is **structure and shape** only — training happens in level05.

## 5. Try It Yourself

1. **(Easy)** Change TinyCNN's first-layer out_channels from 8 to 16 and run it. Which layers' parameter counts change? The point is that it's not just the first layer. (Hint: the next layer's in_channels)
2. **(Medium)** Instead of two stride=2 layers, give the third Conv stride=2 as well. Predict with the formula what the Flatten size becomes, then verify in code. (Hint: 4→2, and the Linear input must change too)
3. **(Challenge)** Set `padding=0` everywhere, run it, and see where the error occurs. Explain the cause using the output-size formula. (Hint: the moment the size shrinks below the kernel)

## 6. Common Mistakes

- **Miscomputing the Flatten size**: a mismatched in_features on the Linear layer throwing `mat1 and mat2 shapes cannot be multiplied` is the number-one beginner CNN error. Always trace the formula to the end.
- **Breaking the channel chain**: previous layer's out_channels ≠ next layer's in_channels means an immediate error. Channels are a relay baton.
- **Violating the 4-D convention**: feed a raw `(16,16)` image and it fails. Use `unsqueeze` to create the batch and channel dimensions and feed `(1,1,16,16)`.
- **Confusing parameter count with compute**: Conv layers have few parameters but repeat the computation at every position, so their FLOPs can be large. "Light" does not always mean "fast".

## Next Level Preview

There is one component in the architecture we have not explained yet — the dedicated layer that summarizes space: pooling. Why use pooling rather than stride, and how features stratify from "low-level to high-level" as layers deepen — level04 shows it in pictures.
