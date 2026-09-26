"""
level09 — Dissecting the Transformer Architecture

Builds positional encoding, multi-head attention, residual connections,
layer normalization, and the feed-forward network one by one in numpy,
then assembles a transformer block. No training (seed-fixed random
weights) — we verify each part's input/output shapes and roles, and the
difference between encoder and decoder (the causal mask).
"""

import numpy as np

SEED = 11
N_TOKENS = 6         # sentence length (token count)
D_MODEL = 32         # model dimension
N_HEADS = 4          # number of attention heads
D_FF = 64            # feed-forward expansion dimension

TOKENS = ["Yesterday", "a", "student", "made", "a", "report"]
rng = np.random.default_rng(SEED)


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# [1] Positional encoding — seat-number tags
# ---------------------------------------------------------------------------

def positional_encoding(n_pos: int, d: int) -> np.ndarray:
    """Gives each position a unique pattern using sine/cosine waves."""
    pos = np.arange(n_pos)[:, None]
    i = np.arange(d // 2)[None, :]
    angle = pos / (10000 ** (2 * i / d))
    pe = np.zeros((n_pos, d))
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle)
    return pe


# ---------------------------------------------------------------------------
# [2] Multi-head attention — splitting the meeting into working groups
# ---------------------------------------------------------------------------

class MultiHeadAttention:
    def __init__(self, d_model: int, n_heads: int):
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        s = 1 / np.sqrt(d_model)
        self.Wq = rng.normal(0, s, (d_model, d_model))
        self.Wk = rng.normal(0, s, (d_model, d_model))
        self.Wv = rng.normal(0, s, (d_model, d_model))
        self.Wo = rng.normal(0, s, (d_model, d_model))

    def __call__(self, X: np.ndarray, causal: bool = False, verbose: bool = False):
        n = X.shape[0]
        Q, K, V = X @ self.Wq, X @ self.Wk, X @ self.Wv       # (n, d_model)
        # Split the dimensions across heads: (n, d_model) -> (heads, n, d_head)
        def split(M):
            return M.reshape(n, self.n_heads, self.d_head).transpose(1, 0, 2)
        Qh, Kh, Vh = split(Q), split(K), split(V)
        scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(self.d_head)  # (heads, n, n)
        if causal:                                             # decoder: no peeking ahead
            mask = np.triu(np.ones((n, n)), k=1).astype(bool)
            scores = np.where(mask, -1e9, scores)
        weights = softmax(scores)
        heads_out = weights @ Vh                               # (heads, n, d_head)
        concat = heads_out.transpose(1, 0, 2).reshape(n, -1)   # concatenate
        if verbose:
            print(f"    Q/K/V shape: {Q.shape} -> split into heads {Qh.shape} "
                  f"({self.n_heads} heads x {self.d_head} dims)")
            print(f"    per-head weight matrices: {weights.shape}, after concat: {concat.shape}")
        return concat @ self.Wo, weights


# ---------------------------------------------------------------------------
# [3] Layer normalization — keeping every teammate's speaking volume in range
# ---------------------------------------------------------------------------

def layer_norm(X: np.ndarray) -> np.ndarray:
    """Standardize each token (row) to mean 0, variance 1. (Scale/shift fixed at 1, 0.)"""
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True)
    return (X - mean) / (std + 1e-6)


# ---------------------------------------------------------------------------
# [4] Feed-forward — private digestion time
# ---------------------------------------------------------------------------

class FeedForward:
    def __init__(self, d_model: int, d_ff: int):
        s = 1 / np.sqrt(d_model)
        self.W1 = rng.normal(0, s, (d_model, d_ff))
        self.W2 = rng.normal(0, s, (d_ff, d_model))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        hidden = np.maximum(X @ self.W1, 0)     # expand + ReLU
        return hidden @ self.W2                 # contract


# ---------------------------------------------------------------------------
# [5] Assembling the block
# ---------------------------------------------------------------------------

class TransformerBlock:
    """Meeting (attention) -> residual+norm -> private digestion (FF) -> residual+norm"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)

    def __call__(self, X: np.ndarray, causal: bool = False) -> np.ndarray:
        attn_out, _ = self.attn(X, causal=causal)
        X = layer_norm(X + attn_out)            # residual connection + layer norm
        X = layer_norm(X + self.ff(X))
        return X

    def n_params(self) -> int:
        mats = [self.attn.Wq, self.attn.Wk, self.attn.Wv, self.attn.Wo,
                self.ff.W1, self.ff.W2]
        return sum(m.size for m in mats)


if __name__ == "__main__":
    print("=" * 70)
    print("Transformer anatomy — build the parts, follow the shapes, assemble")
    print("=" * 70 + "\n")
    np.set_printoptions(precision=2, suppress=True)

    # [1] Embedding + positional encoding ----------------------------------------
    embed_table = rng.normal(0, 1, (100, D_MODEL))        # a fake embedding table
    X = embed_table[: N_TOKENS]                            # pretend these are our 6 tokens
    pe = positional_encoding(N_TOKENS, D_MODEL)
    print(f"[1] Positional encoding — attention is order-blind, so add seat-number tags")
    print(f"    token embeddings X: {X.shape} ({N_TOKENS} tokens x {D_MODEL} dims)")
    print(f"    PE table: {pe.shape}, first 4 cells at pos 0 {pe[0, :4]}, at pos 3 {pe[3, :4]}")
    same_word_diff = np.linalg.norm((X[0] + pe[0]) - (X[0] + pe[3]))
    print(f"    same word at position 0 vs 3 — input difference (norm): {same_word_diff:.2f}")
    print("    -> The same word in a different seat becomes a different input.\n")
    X = X + pe

    # [2] Multi-head attention --------------------------------------------------
    print(f"[2] Multi-head attention — split {D_MODEL} dims into {N_HEADS} working groups")
    mha = MultiHeadAttention(D_MODEL, N_HEADS)
    attn_out, weights = mha(X, verbose=True)
    print(f"    output: {attn_out.shape} (same shape as the input — that's what makes stacking possible)")
    head0_focus = TOKENS[int(np.argmax(weights[0][2]))]
    head1_focus = TOKENS[int(np.argmax(weights[2][2]))]
    print(f"    where the 'student' token attends most: head0='{head0_focus}', head2='{head1_focus}'")
    print("    -> Each head has its own attention pattern (division of perspectives).\n")

    # [3] Residual connection + layer norm ---------------------------------------
    print("[3] Residual + layer norm — preserving the original and managing condition")
    added = X + attn_out
    normed = layer_norm(added)
    print(f"    per-token std after residual sum: {added.std(axis=1)[:4]} ...")
    print(f"    per-token std after layer norm  : {normed.std(axis=1)[:4]} ... (all 1)")
    print("    -> However many layers you stack, the scale of the values stays steady.\n")

    # [4] Feed-forward -------------------------------------------------------
    print(f"[4] Feed-forward — each token digests on its own")
    ff = FeedForward(D_MODEL, D_FF)
    hidden = np.maximum(normed @ ff.W1, 0)
    ff_out = hidden @ ff.W2
    print(f"    expand: {normed.shape} -> {hidden.shape} (ReLU) -> contract: {ff_out.shape}")
    print("    -> No exchange between tokens here. Exchange happens only in attention.\n")

    # [5] Assembling and stacking blocks -------------------------------------------
    print("[5] Assembling the block — stacking 'meeting + digestion' sets")
    block1 = TransformerBlock(D_MODEL, N_HEADS, D_FF)
    block2 = TransformerBlock(D_MODEL, N_HEADS, D_FF)
    out1 = block1(X)
    out2 = block2(out1)
    print(f"    input {X.shape} -> block1 -> {out1.shape} -> block2 -> {out2.shape}")
    per_block = block1.n_params()
    print(f"    parameters per block: {per_block:,}")
    print(f"    Note: a large LLM is this block scaled to dozens-to-hundreds of layers and thousands of dims.\n")

    # [6] Encoder vs decoder -------------------------------------------------
    print("[6] Encoder vs decoder — same parts, one mask of difference")
    _, w_enc = mha(X, causal=False)
    _, w_dec = mha(X, causal=True)
    print("    encoder (bidirectional) head0 weights:")
    print("    " + str(w_enc[0]).replace("\n", "\n    "))
    print("    decoder (causal mask) head0 weights — upper right all zero:")
    print("    " + str(w_dec[0]).replace("\n", "\n    "))
    print("    -> The decoder cannot see 'future tokens that haven't appeared yet'. It is")
    print("       the anti-cheating device of next-token prediction — the mask that makes")
    print("       the G in GPT possible.")
    print("\nBottom line: transformer = (seat tags + group meetings + private digestion + safety gear) x N layers.")
