"""
A miniature implementation of a pre-training data pipeline.
We join documents into one long token stream with <s>...</s> (packing),
save it as a uint16 binary shard, load it instantly with numpy memmap,
and produce fixed-length (block_size) training batches — the whole flow.
It is a scale model of how real large-scale training handles terabytes.
"""

import os
import sys
import pathlib
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"

# Special tokens — pinned to the lowest ids.
BOS, EOS, PAD, UNK = "<s>", "</s>", "<pad>", "<unk>"
SPECIALS = [BOS, EOS, PAD, UNK]


def make_documents(n_docs: int = 200, sents_per_doc: int = 5):
    """Split tiny_corpus into sentences, then bundle 5 sentences per 'document'."""
    corpus = hjh_data.tiny_corpus()
    sents = [s.strip() + "." for s in corpus.split(".") if s.strip()]
    docs = []
    for i in range(n_docs):
        chunk = sents[i * sents_per_doc:(i + 1) * sents_per_doc]
        docs.append(" ".join(chunk))
    return docs


def build_vocab(docs):
    """Character-level vocab: 4 special tokens + every character in the corpus."""
    chars = sorted(set("".join(docs)))
    itos = SPECIALS + chars
    stoi = {ch: i for i, ch in enumerate(itos)}
    return stoi, itos


def tokenize_and_pack(docs, stoi):
    """Wrap each document as <s> tokens </s> and join into one long stream."""
    stream = []
    for doc in docs:
        stream.append(stoi[BOS])
        stream.extend(stoi.get(ch, stoi[UNK]) for ch in doc)
        stream.append(stoi[EOS])
    return np.array(stream, dtype=np.uint16)  # vocab < 65536, so 2 bytes suffice


def get_batch(mm, block_size: int, batch_size: int, rng):
    """Pick random positions in the memmap stream and build an (x, y) batch.
    y is x shifted by one — the answer sheet for 'guess the next token'."""
    starts = rng.integers(0, len(mm) - block_size - 1, size=batch_size)
    x = np.stack([np.asarray(mm[s:s + block_size]) for s in starts])
    y = np.stack([np.asarray(mm[s + 1:s + block_size + 1]) for s in starts])
    return x, y


if __name__ == "__main__":
    rng = np.random.default_rng(42)  # fixed seed

    # [1] raw text -> list of documents
    docs = make_documents()
    total_chars = sum(len(d) for d in docs)
    print(f"[1] Documents ready: {len(docs)} documents, {total_chars:,} characters total")
    print(f"    Example document: \"{docs[0][:40]}...\"")

    # [2] build the vocab (character-level — level01's BPE would make it shorter)
    stoi, itos = build_vocab(docs)
    print(f"\n[2] Vocab built: {len(itos)} entries"
          f" ({len(SPECIALS)} special tokens + {len(itos) - len(SPECIALS)} characters)")
    print(f"    Special token ids: " +
          ", ".join(f"{t}={stoi[t]}" for t in SPECIALS))

    # [3] tokenize + <s>...</s> packing
    stream = tokenize_and_pack(docs, stoi)
    print(f"\n[3] Packing done: {len(docs)} documents -> a stream of {len(stream):,} tokens")
    print("    Cutting documents individually wastes <pad> on every short one,")
    print("    so joining everything and slicing to fixed length keeps the GPU full.")

    # [4] save as a uint16 binary shard
    os.makedirs(OUT_DIR, exist_ok=True)
    shard_path = OUT_DIR / "shard_000.bin"
    stream.tofile(shard_path)
    disk = os.path.getsize(shard_path)
    text_bytes = sum(len(d.encode("utf-8")) for d in docs)
    print(f"\n[4] Shard saved: {shard_path}")
    print(f"    Size {disk:,} bytes (2 bytes per token)"
          f" / original UTF-8 {text_bytes:,} bytes")
    print("    Real pipelines create thousands of such shards and read them in parallel.")

    # [5] memmap loading — use the file instantly without loading it all into RAM
    mm = np.memmap(shard_path, dtype=np.uint16, mode="r")
    print(f"\n[5] memmap loaded: {len(mm):,} tokens — only the needed slices are read, on demand")
    print("    This is why even multi-hundred-GB shards open in near-zero time.")

    # [6] build fixed-length batches
    block_size, batch_size = 64, 8
    x, y = get_batch(mm, block_size, batch_size, rng)
    print(f"\n[6] Batch created: x{x.shape}, y{y.shape}"
          f" (block_size={block_size}, batch_size={batch_size})")
    print(f"    x[0][:8] = {x[0][:8].tolist()}")
    print(f"    y[0][:8] = {y[0][:8].tolist()}  <- x shifted by one: the answer sheet")

    # [7] verify by decoding — document-boundary tokens end up inside batches.
    def decode(ids):
        return "".join(itos[i] for i in ids)

    row = decode(x[0])
    print(f"\n[7] First batch row decoded: \"{row[:60]}\"")
    boundary = [i for i in range(batch_size)
                if stoi[BOS] in x[i] or stoi[EOS] in x[i]]
    print(f"    Rows containing a document boundary (<s>/</s>): {len(boundary)}/{batch_size}")
    print("    Pieces that straddle a boundary are trained on as-is — <s> announces")
    print("    'a new document starts here', so the model learns to break context on its own.")
