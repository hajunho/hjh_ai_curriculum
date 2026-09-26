# Lecture 12 · Level 02 — Training Data Pipelines and Packing

> We join documents with `<s>...</s>`, slice them to fixed length, and build a miniature of "the conveyor belt that streams trillions of tokens" — with uint16 binary shards and memmap.

**Difficulty** ⭐⭐⭐ / **Prerequisites** level01 / **Estimated time** 50 min

## 1. Why Learn This — The Business View

During LLM pre-training, every GPU runs with a taxi meter attached — dollars per hour, thousands of them. Every moment this expensive machine sits idle waiting for data is money burned. That is why the real skill gap in large-scale training often lies not in the model code but in **the pipeline that feeds data without stalls and without waste**.

From a business standpoint, this level answers three questions. First, "why is training data stored as weird binary instead of text files?" — because re-tokenizing on every pass makes the CPU the bottleneck, we tokenize once and store the result in a form the machine can eat directly. Second, "how do you fit hundreds of gigabytes into memory?" — you don't. A technique called memmap reads only the slices you need, on demand. Third, "documents come in all lengths, so why is the GPU always full?" — packing. These three concepts appear verbatim in AI infrastructure quotes and bottleneck discussions.

## 2. Grasping It Through an Analogy

Picture a sandwich factory. Orders (documents) come in every size: one customer wants two sliders, another wants ten footlongs.

**The bad way**: assign each order its own lunchbox (a fixed-length batch) and fill the empty space with foam packing peanuts (`<pad>` tokens). Put two sliders in a jumbo box and the box is mostly foam — most of the GPU's compute is wasted crunching meaningless padding.

**The packing way**: roll every order into one endless super-sandwich (the token stream), inserting divider picks (`<s>`, `</s>` tokens) between orders to mark the boundaries. Then chop it into box-sized pieces (block_size). Some boxes straddle two orders, but the dividers let the eater (the model) tell "a new order starts here." No box has empty space.

Storage works like big catering tubs (binary shards). Instead of individual wrappers (millions of per-document files), you pack everything into a few thousand large tubs — fewer trips to the fridge door (file-open cost), faster logistics.

## 3. Core Concepts

### 3-1. Packing: `<s> doc1 </s> <s> doc2 </s> ...`

Tokenize every document, wrap it in special tokens, join everything into one long stream, and cut it into block_size pieces (64 in this exercise; 4096–128k in real systems) as training samples. Two advantages: (1) zero padding, so all GPU compute goes to real learning; (2) uniform sample lengths make batching trivial. The trade-off is that one sample can contain the tail of one document and the head of another — but the `<s>` token acts as the "new document starts" signal, and the model learns to break context along with everything else.

### 3-2. uint16 Shards

Token ids are integers. If the vocab is smaller than 65,536, each id fits in 2 bytes (uint16). At a fixed 2 bytes per token, file sizes are instantly predictable (100 million tokens = 200MB) and the file reads straight into a numeric array with no parsing. Splitting the whole dataset into hundreds or thousands of shard files lets multiple GPUs read different shards in parallel, and if one is corrupted you rebuild only that shard.

### 3-3. memmap — Opening Without Opening

`np.memmap` does not copy the file into RAM; it merely tells the operating system "pretend this file is in memory." Actual reads happen only when you touch a particular range of the array — just that part comes up from disk. That is why a 200GB shard opens in near-zero time, and drawing a batch causes only a few KB of I/O. As a bonus, the numpy slicing syntax you learned in lecture03 works unchanged.

### 3-4. The (x, y) Batch — Shift the Answer Sheet by One

A language model's answer is "the next token," so if the input x is the stream's `[i : i+64]`, the target y is `[i+1 : i+65]`. No separate label file needed. The data is simultaneously the exam and the answer key — this is why pre-training data can be "just text."

## 4. Hands-On — main.py

```bash
cd lecture12_llm_engineering_mlops/level02_data_pipeline_packing
python3 main.py
```

The seven pipeline stages run in order. `[1]` bundles tiny_corpus into 200 documents of 5 sentences each (42,027 characters), and `[2]` builds a 32-entry vocab from 4 special tokens (`<s>`, `</s>`, `<pad>`, `<unk>`) plus the 28 characters (character-level — swap in level01's BPE and the stream gets shorter). `[3]` `tokenize_and_pack()` wraps every document in `<s>...</s>` and joins them into a single stream of 42,427 tokens, and `[4]` saves it with `stream.tofile()` to `outputs/shard_000.bin`. At 2 bytes per token the shard (84,854 bytes) is about twice the UTF-8 original (42,027 bytes) — English is 1 byte per character in UTF-8, so character-level uint16 doubles it; with a real BPE vocab (fewer, longer tokens) or a multi-byte script like Korean, the binary comes out *smaller* instead. The win here is not compression but zero re-tokenization and instant numeric loading. `[5]` loads it with `np.memmap` instantly, and `[6]` `get_batch()` cuts length-64 (x, y) pairs at 8 random start points. Confirm in the output that y is x shifted by exactly one. In `[7]`, decoding one batch row back to characters shows it starting mid-sentence with `</s><s>` boundaries mixed in (3 of the 8 rows contain a boundary) — this is what packed training data actually looks like.

## 5. Try It Yourself

1. **(Easy)** Change `block_size` from 64 to 256 and run again. How do the batch shape and the tokens consumed per batch change? (Hint: tokens per batch = block_size × batch_size. In real training this number is the basic unit of "throughput per step.")
2. **(Medium)** Using `[7]` as a guide, add code that counts `<s>` tokens across all 8 batch rows. Check how the probability of containing a boundary changes as block_size grows. (Hint: one line — `(x == stoi[BOS]).sum()`.)
3. **(Challenge)** Save the documents as two shards (`shard_000.bin`, `shard_001.bin`) and modify batching to pick a shard at random. (Hint: cut the stream array in half, `tofile` each, and choose with `rng.integers(0, 2)` before `get_batch`. This is the skeleton of a multi-GPU data loader.)

## 6. Common Mistakes

- **Packing without `<s>`**: with no boundary markers, the end of one document becomes a "false signal" predicting the start of the next, and the model learns confusion. Don't skip the divider picks.
- **Overflowing uint16**: if the vocab exceeds 65,536 but you store uint16, token ids get silently truncated into garbage data. Make `vocab_size < 65536` a habitual pre-save check. (Real large-model vocabs exceed 100k and sometimes use uint32.)
- **Copying the whole memmap**: converting everything with `np.array(mm)` throws away the memmap's advantage and blows up RAM. Always slice out only the range you need.
- **Tokenizer version mismatch**: decode a shard with a different tokenizer version than the one that made it, and the data is wholesale corrupted. Always store the vocab file and version info right next to the shard.

## Next Level Preview

The conveyor belt is built, but what you put on the belt still matters. "Garbage in, garbage out" — in level03 we build n-gram-based **deduplication** and a **benchmark contamination checker** that catches exam questions leaking into the textbook, learning data quality management hands-on.
