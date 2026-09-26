"""
level09 — トランスフォーマー構造の解剖

位置エンコーディング、マルチヘッドアテンション、残差接続、レイヤー正規化、
フィードフォワードを numpy で 1 つずつ作り、トランスフォーマーブロックを組み立てます。
学習はせず (seed 固定の乱数重み)、各部品の入出力 shape と
役割、エンコーダ vs デコーダ (因果マスク) の違いを出力で確認します。
"""

import numpy as np

SEED = 11
N_TOKENS = 6         # 文の長さ (トークン数)
D_MODEL = 32         # モデル次元
N_HEADS = 4          # アテンションヘッド数
D_FF = 64            # フィードフォワードの拡張次元

TOKENS = ["昨日", "学生", "が", "報告書", "を", "作った"]
rng = np.random.default_rng(SEED)


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# [1] 位置エンコーディング — 座席番号札
# ---------------------------------------------------------------------------

def positional_encoding(n_pos: int, d: int) -> np.ndarray:
    """サイン/コサインの波で各位置に固有のパターンを与えます。"""
    pos = np.arange(n_pos)[:, None]
    i = np.arange(d // 2)[None, :]
    angle = pos / (10000 ** (2 * i / d))
    pe = np.zeros((n_pos, d))
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle)
    return pe


# ---------------------------------------------------------------------------
# [2] マルチヘッドアテンション — 観点を分けて会議する
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
        # ヘッド数だけ次元を分割します: (n, d_model) -> (heads, n, d_head)
        def split(M):
            return M.reshape(n, self.n_heads, self.d_head).transpose(1, 0, 2)
        Qh, Kh, Vh = split(Q), split(K), split(V)
        scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(self.d_head)  # (heads, n, n)
        if causal:                                             # デコーダ: 後ろを見えなくする
            mask = np.triu(np.ones((n, n)), k=1).astype(bool)
            scores = np.where(mask, -1e9, scores)
        weights = softmax(scores)
        heads_out = weights @ Vh                               # (heads, n, d_head)
        concat = heads_out.transpose(1, 0, 2).reshape(n, -1)   # つなぎ合わせ
        if verbose:
            print(f"    Q/K/V の shape: {Q.shape} -> ヘッド分割 {Qh.shape} "
                  f"(ヘッド {self.n_heads}個 x {self.d_head}次元)")
            print(f"    ヘッドごとの重み行列: {weights.shape}、結合後: {concat.shape}")
        return concat @ self.Wo, weights


# ---------------------------------------------------------------------------
# [3] レイヤー正規化 — チームメンバーの声のトーンを揃える
# ---------------------------------------------------------------------------

def layer_norm(X: np.ndarray) -> np.ndarray:
    """トークン (行) ごとに平均 0、分散 1 に標準化。(倍率・シフトのパラメータは 1, 0 に固定)"""
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True)
    return (X - mean) / (std + 1e-6)


# ---------------------------------------------------------------------------
# [4] フィードフォワード — 一人で整理する時間
# ---------------------------------------------------------------------------

class FeedForward:
    def __init__(self, d_model: int, d_ff: int):
        s = 1 / np.sqrt(d_model)
        self.W1 = rng.normal(0, s, (d_model, d_ff))
        self.W2 = rng.normal(0, s, (d_ff, d_model))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        hidden = np.maximum(X @ self.W1, 0)     # 拡張 + ReLU
        return hidden @ self.W2                 # 縮小


# ---------------------------------------------------------------------------
# [5] ブロックの組み立て
# ---------------------------------------------------------------------------

class TransformerBlock:
    """会議 (アテンション) -> 残差+ノルム -> 個人整理 (FF) -> 残差+ノルム"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)

    def __call__(self, X: np.ndarray, causal: bool = False) -> np.ndarray:
        attn_out, _ = self.attn(X, causal=causal)
        X = layer_norm(X + attn_out)            # 残差接続 + レイヤーノルム
        X = layer_norm(X + self.ff(X))
        return X

    def n_params(self) -> int:
        mats = [self.attn.Wq, self.attn.Wk, self.attn.Wv, self.attn.Wo,
                self.ff.W1, self.ff.W2]
        return sum(m.size for m in mats)


if __name__ == "__main__":
    print("=" * 70)
    print("トランスフォーマー解剖 — 部品を作り、shape を追いながら組み立てる")
    print("=" * 70 + "\n")
    np.set_printoptions(precision=2, suppress=True)

    # [1] 埋め込み + 位置エンコーディング -----------------------------------
    embed_table = rng.normal(0, 1, (100, D_MODEL))        # 疑似の埋め込みテーブル
    X = embed_table[: N_TOKENS]                            # トークン 6 個と仮定
    pe = positional_encoding(N_TOKENS, D_MODEL)
    print(f"[1] 位置エンコーディング — アテンションは順序を知らないので、座席番号札を足す")
    print(f"    トークン埋め込み X: {X.shape} (トークン {N_TOKENS}個 x {D_MODEL}次元)")
    print(f"    PE 表: {pe.shape}、位置0 の先頭 4 マス {pe[0, :4]}、位置3 の先頭 4 マス {pe[3, :4]}")
    same_word_diff = np.linalg.norm((X[0] + pe[0]) - (X[0] + pe[3]))
    print(f"    同じ単語が位置 0 vs 3 にあるときの入力差 (ノルム): {same_word_diff:.2f}")
    print("    -> 同じ単語でも座席が違えば違う入力になります。\n")
    X = X + pe

    # [2] マルチヘッドアテンション -------------------------------------------
    print(f"[2] マルチヘッドアテンション — {D_MODEL}次元を {N_HEADS}個の分科会に分けて同時会議")
    mha = MultiHeadAttention(D_MODEL, N_HEADS)
    attn_out, weights = mha(X, verbose=True)
    print(f"    出力: {attn_out.shape} (入力と同じ shape — だから積み上げられる)")
    head0_focus = TOKENS[int(np.argmax(weights[0][1]))]
    head1_focus = TOKENS[int(np.argmax(weights[1][1]))]
    print(f"    「学生」トークンが最も注目した先: ヘッド0='{head0_focus}'、ヘッド1='{head1_focus}'")
    print("    -> ヘッドごとに注目パターンが違います (観点の分担)。\n")

    # [3] 残差接続 + レイヤーノルム -----------------------------------------
    print("[3] 残差接続 + レイヤーノルム — 原本の保存とコンディション管理")
    added = X + attn_out
    normed = layer_norm(added)
    print(f"    残差合算後のトークン別標準偏差: {added.std(axis=1)[:4]} ...")
    print(f"    レイヤーノルム後のトークン別標準偏差: {normed.std(axis=1)[:4]} ... (すべて 1)")
    print("    -> 層をいくら積んでも値のスケールが一定に保たれます。\n")

    # [4] フィードフォワード -------------------------------------------------
    print(f"[4] フィードフォワード — 各トークンが一人で消化する時間")
    ff = FeedForward(D_MODEL, D_FF)
    hidden = np.maximum(normed @ ff.W1, 0)
    ff_out = hidden @ ff.W2
    print(f"    拡張: {normed.shape} -> {hidden.shape} (ReLU) -> 縮小: {ff_out.shape}")
    print("    -> トークン間の交流はありません。交流はアテンションの中でだけ起きます。\n")

    # [5] ブロックの組み立てと積層 -------------------------------------------
    print("[5] ブロックの組み立て — 「会議+整理」のセットを積む")
    block1 = TransformerBlock(D_MODEL, N_HEADS, D_FF)
    block2 = TransformerBlock(D_MODEL, N_HEADS, D_FF)
    out1 = block1(X)
    out2 = block2(out1)
    print(f"    入力 {X.shape} -> ブロック1 -> {out1.shape} -> ブロック2 -> {out2.shape}")
    per_block = block1.n_params()
    print(f"    ブロック 1 個のパラメータ: {per_block:,}個")
    print(f"    参考: 大型 LLM は、このブロックを数十〜百余りの層、次元数千に育てたものです。\n")

    # [6] エンコーダ vs デコーダ ---------------------------------------------
    print("[6] エンコーダ vs デコーダ — 同じ部品、マスク 1 枚の違い")
    _, w_enc = mha(X, causal=False)
    _, w_dec = mha(X, causal=True)
    print("    エンコーダ (双方向) ヘッド0 の重み:")
    print("    " + str(w_enc[0]).replace("\n", "\n    "))
    print("    デコーダ (因果マスク) ヘッド0 の重み — 右上が全部 0:")
    print("    " + str(w_dec[0]).replace("\n", "\n    "))
    print("    -> デコーダは「まだ出てきていない未来のトークン」を見られません。次のトークン予測の")
    print("       カンニング防止装置であり、GPT の G を可能にするマスクです。")
    print("\n結論: トランスフォーマー = (座席札 + 分科会議 + 個人整理 + 安全装置) x N層。")
