# Lecture 09 · Level 11 — Vision Transformers and Multimodal Models

> We build a miniature ViT that reads an image as a sentence of "patch words", and wrap up with the concept of CLIP-style multimodality, where images and text meet in one space.
**Difficulty** ⭐⭐⭐⭐⭐ / **Prerequisites** level03–05 (even better if you've seen lecture10's attention first) / **Estimated time** 55 min

## 1. Why This Matters — the Business View

In today's vision-AI news and proposals, different words appear where CNN used to — "transformer-based vision model", "multimodal", "image–text search", "zero-shot classification". The features where you toss in a photo and get a caption, or search "torn packaging" and get matching photos, are rooted in the **Vision Transformer (ViT)** and **CLIP-style image–text alignment**.

Knowing these terms precisely, at the concept level, buys you two things. First, judgment calls: "does our defect inspection need multimodal?" (usually no — classification and detection suffice) versus "we want to search our unlabeled image assets by text" (yes — exactly what multimodal is for). Second, questioning power: when a vendor leads with "state-of-the-art transformer-based", you can probe what actual benefit that brings to the task. As the final level of this lecture, we connect the journey that started at pixels all the way to the newest grammar.

## 2. An Analogy

**Reading an image like a sentence.** The transformer (which you will meet in lecture10) reads a sentence as "a sequence of word tokens". ViT's idea is audaciously simple — **slice the image into pieces (patches), line the pieces up like words, and feed them straight into the sentence-reading machine.** Slice a 16×16 image into sixteen 4×4 pieces and you have "a sentence of 16 words".

**Self-attention is an all-hands meeting.** The CNN goes "next door first, step by step" — each pixel sees only its neighbors, and the field of view widens only by stacking layers (level04). Attention, from the very first layer, **puts every patch in one meeting room, where each decides for itself whose statements to weigh, and how much.** If the top-left patch judges "the one most relevant to me is the bottom-right patch", it consults it directly, regardless of distance. Capturing relationships between far-apart parts (long-range dependencies) in one step is attention's power.

**CLIP-style multimodality is a universal coordinate system.** The image department and the text department each write their reports, but agree to **submit them summarized as vectors in the same coordinate system**. Train on hundreds of millions of pairs so the vector of a cat photo lands near the vector of the sentence "a photo of a cat" — and afterwards you can search images with text, and classify without labels by asking "which sentence is this photo closest to?" (zero-shot).

## 3. Core Concepts

### 3.1 The ViT assembly order

1. **Patch tokenization**: cut the H×W image into P×P pieces. (16×16, P=4 → 16 tokens. A real ViT: 224×224, P=16 → 196 tokens)
2. **Patch embedding**: flatten each piece's pixels into a row and Linear-map it to a d-dimensional vector. The image version of "word embedding".
3. **Positional encoding**: attention is order-blind ("words in a bag"), so each token gets a position vector added, stamping its place.
4. **Self-attention layers × N**: see 3.2.
5. **Summarize → classification head**: average the tokens (or use a dedicated summary token) into class scores.

### 3.2 Self-attention — the three glasses: Q, K, V

Each token produces three vectors. **Query** ("what I'm looking for"), **Key** ("what I have"), **Value** ("what I'll pass along"). For every token pair, measure relevance by the Q·K dot product, softmax it into an "attention allocation table" (each row sums to 1), then mix and fetch the V's in those proportions.

```
attn = softmax(Q Kᵀ / √d),   output = attn V
```

Because this allocation table (the attention map) falls out as a by-product, you can peek at "where the model looked when it decided" — that is the visualization in main.py.

### 3.3 CNN vs ViT — with and without prior knowledge

The CNN bakes the priors of locality and translation invariance into its structure (we saw in level05 that this is "free data"). ViT includes almost none of that prior — instead, **given very large data, it discovers for itself how images should be seen**. Hence the rule of thumb: little data → CNN (or transfer learning from a pretrained ViT, level07); big data and compute → the ViT family wins. That our mini ViT does not crush the CNN on today's 600 images is an honest reflection of this principle.

### 3.4 CLIP-style alignment and zero-shot (concepts only)

An image encoder (e.g. a ViT) and a text encoder each emit vectors, trained jointly by contrastive learning — "paired image and sentence pull close, unpaired push apart". What the result buys in practice:

- **Text → image search**: query an unlabeled photo warehouse with a sentence.
- **Zero-shot classification**: compare the image vector's similarity to candidate sentences ("torn packaging", "intact packaging") — classification with no training for those classes.
- **The eyes of generative and understanding models**: image captioning, document understanding — the visual front-end of multimodal LLMs is built atop such aligned encoders.

And the limits: distinctions that language does not split well (like subtle defects), and domain-specific imagery, send zero-shot performance off a cliff. The working rule: "however slick the zero-shot demo, always evaluate on our own data."

## 4. Hands-On — main.py

Run it:

```bash
python3 main.py
```

- **[1]** Tokenizes a shape image into sixteen 4×4 patches. Prints each patch's mean brightness as a 4×4 grid, confirming "the shape's location survives even in pieces".
- **[2]** Explains and constructs the patch embedding (Linear 16→32) and learnable positional encoding.
- **[3]** Implements self-attention directly as matrix operations: `Q @ Kᵀ / √d → softmax → @ V` (6 lines in `MiniViT.forward`).
- **[4]** Trains the mini ViT (5,283 parameters) on shape classification for 80 epochs. Test accuracy about 0.92 — falling short of level05's CNN (0.97) is itself a lesson (section 3.3).
- **[5]** Extracts the **mean attention map** for one image per class, overlays it on the original, and saves to `outputs/vit_attention.png`. Bright patches = most consulted in the decision. See with your own eyes that they cluster on the shape — nobody taught it "look at the shape"; attention drifted there simply from trying to classify well.

## 5. Try It Yourself

1. **(Easy)** Change `PATCH` from 4 to 8 (a 4-token sentence). How do the accuracy and the attention map coarsen? Confirm that token count = representational resolution. (Hint: N_PATCH recomputes automatically)
2. **(Medium)** Turn off the positional encoding (remove `+ self.pos`). How much does accuracy drop? Think about what becomes impossible without knowing where the shape is. (Hint: patch contents alone still get you part of the way — why?)
3. **(Challenge)** Instead of the mean attention, visualize **one specific patch's attention row** — e.g. `attn[0]` for the top-left patch 0. Can you see background patches consulting shape patches? An experiment in watching "who listens to whom" in the meeting, individually. (Hint: `attn[0, 0]` is patch 0's allocation table)

## 6. Common Mistakes

- **Over-trusting the attention map as a full explanation**: attention is a clue to "what was consulted", not a causal explanation. Use it as supporting material only.
- **Forcing ViT onto small data**: training a ViT from scratch on small data without pretraining often loses to a CNN. Transfer learning (level07) is the premise.
- **"Multimodal = always newest and best"**: for fixed-class classification and detection tasks, a dedicated model is cheaper and more accurate. Multimodality earns its keep when you need a language interface.
- **Dropping the √d division**: skip dividing attention scores by the square root of the dimension and softmax collapses onto one spot, destabilizing training. A staple bug in hand-rolled implementations.

## Next Level Preview

Congratulations on completing lecture09. Starting from level00's confirmation that pixels are numbers, through convolution, CNNs, augmentation, transfer, detection, and segmentation, we arrived at transformers that read images like sentences. The attention, tokens, and embeddings that appeared here are the protagonists of the next lecture (lecture10 — natural language processing). When you meet in language the concepts you met in images, you will see that the two worlds speak the same grammar.
