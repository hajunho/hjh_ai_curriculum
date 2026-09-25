"""
level09 — 트랜스포머 구조 해부

포지셔널 인코딩, 멀티헤드 어텐션, 잔차 연결, 레이어 정규화,
피드포워드를 numpy 로 하나씩 만들어 트랜스포머 블록을 조립합니다.
학습은 하지 않고(seed 고정 난수 가중치) 각 부품의 입출력 shape 와
역할, 인코더 vs 디코더(인과 마스크)의 차이를 출력으로 확인합니다.
"""

import numpy as np

SEED = 11
N_TOKENS = 6         # 문장 길이 (토큰 수)
D_MODEL = 32         # 모델 차원
N_HEADS = 4          # 어텐션 헤드 수
D_FF = 64            # 피드포워드 확장 차원

TOKENS = ["어제", "학생", "이", "보고서", "를", "만들었다"]
rng = np.random.default_rng(SEED)


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# [1] 포지셔널 인코딩 — 자리 번호표
# ---------------------------------------------------------------------------

def positional_encoding(n_pos: int, d: int) -> np.ndarray:
    """사인/코사인 파동으로 각 위치에 고유 패턴을 부여합니다."""
    pos = np.arange(n_pos)[:, None]
    i = np.arange(d // 2)[None, :]
    angle = pos / (10000 ** (2 * i / d))
    pe = np.zeros((n_pos, d))
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle)
    return pe


# ---------------------------------------------------------------------------
# [2] 멀티헤드 어텐션 — 관점을 나눠서 회의하기
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
        # 헤드 수만큼 차원을 쪼갭니다: (n, d_model) -> (heads, n, d_head)
        def split(M):
            return M.reshape(n, self.n_heads, self.d_head).transpose(1, 0, 2)
        Qh, Kh, Vh = split(Q), split(K), split(V)
        scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(self.d_head)  # (heads, n, n)
        if causal:                                             # 디코더: 뒤를 못 보게
            mask = np.triu(np.ones((n, n)), k=1).astype(bool)
            scores = np.where(mask, -1e9, scores)
        weights = softmax(scores)
        heads_out = weights @ Vh                               # (heads, n, d_head)
        concat = heads_out.transpose(1, 0, 2).reshape(n, -1)   # 이어붙이기
        if verbose:
            print(f"    Q/K/V shape: {Q.shape} -> 헤드 분할 {Qh.shape} "
                  f"(헤드 {self.n_heads}개 x {self.d_head}차원)")
            print(f"    헤드별 가중치 행렬: {weights.shape}, 결합 후: {concat.shape}")
        return concat @ self.Wo, weights


# ---------------------------------------------------------------------------
# [3] 레이어 정규화 — 팀원들의 발언 톤 맞추기
# ---------------------------------------------------------------------------

def layer_norm(X: np.ndarray) -> np.ndarray:
    """토큰(행)별로 평균 0, 분산 1 로 표준화. (배율·이동 파라미터는 1, 0 으로 고정)"""
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True)
    return (X - mean) / (std + 1e-6)


# ---------------------------------------------------------------------------
# [4] 피드포워드 — 개인 정리 시간
# ---------------------------------------------------------------------------

class FeedForward:
    def __init__(self, d_model: int, d_ff: int):
        s = 1 / np.sqrt(d_model)
        self.W1 = rng.normal(0, s, (d_model, d_ff))
        self.W2 = rng.normal(0, s, (d_ff, d_model))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        hidden = np.maximum(X @ self.W1, 0)     # 확장 + ReLU
        return hidden @ self.W2                 # 축소


# ---------------------------------------------------------------------------
# [5] 블록 조립
# ---------------------------------------------------------------------------

class TransformerBlock:
    """회의(어텐션) -> 잔차+놈 -> 개인정리(FF) -> 잔차+놈"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)

    def __call__(self, X: np.ndarray, causal: bool = False) -> np.ndarray:
        attn_out, _ = self.attn(X, causal=causal)
        X = layer_norm(X + attn_out)            # 잔차 연결 + 레이어놈
        X = layer_norm(X + self.ff(X))
        return X

    def n_params(self) -> int:
        mats = [self.attn.Wq, self.attn.Wk, self.attn.Wv, self.attn.Wo,
                self.ff.W1, self.ff.W2]
        return sum(m.size for m in mats)


if __name__ == "__main__":
    print("=" * 70)
    print("트랜스포머 해부 — 부품을 만들고 shape 를 따라가며 조립")
    print("=" * 70 + "\n")
    np.set_printoptions(precision=2, suppress=True)

    # [1] 임베딩 + 포지셔널 인코딩 ----------------------------------------
    embed_table = rng.normal(0, 1, (100, D_MODEL))        # 가짜 임베딩 테이블
    X = embed_table[: N_TOKENS]                            # 토큰 6개라고 가정
    pe = positional_encoding(N_TOKENS, D_MODEL)
    print(f"[1] 포지셔널 인코딩 — 어텐션은 순서를 모르므로 자리 번호표를 더한다")
    print(f"    토큰 임베딩 X: {X.shape} (토큰 {N_TOKENS}개 x {D_MODEL}차원)")
    print(f"    PE 표: {pe.shape}, 위치0 앞 4칸 {pe[0, :4]}, 위치3 앞 4칸 {pe[3, :4]}")
    same_word_diff = np.linalg.norm((X[0] + pe[0]) - (X[0] + pe[3]))
    print(f"    같은 단어가 위치 0 vs 3 에 있을 때 입력 차이(노름): {same_word_diff:.2f}")
    print("    -> 같은 단어라도 자리가 다르면 다른 입력이 됩니다.\n")
    X = X + pe

    # [2] 멀티헤드 어텐션 --------------------------------------------------
    print(f"[2] 멀티헤드 어텐션 — {D_MODEL}차원을 {N_HEADS}개 분과로 나눠 동시 회의")
    mha = MultiHeadAttention(D_MODEL, N_HEADS)
    attn_out, weights = mha(X, verbose=True)
    print(f"    출력: {attn_out.shape} (입력과 같은 shape — 그래야 쌓을 수 있음)")
    head0_focus = TOKENS[int(np.argmax(weights[0][1]))]
    head1_focus = TOKENS[int(np.argmax(weights[1][1]))]
    print(f"    '학생' 토큰이 가장 주목한 곳: 헤드0='{head0_focus}', 헤드1='{head1_focus}'")
    print("    -> 헤드마다 주목 패턴이 다릅니다(관점 분담).\n")

    # [3] 잔차 연결 + 레이어놈 ---------------------------------------------
    print("[3] 잔차 연결 + 레이어놈 — 원본 보존과 컨디션 관리")
    added = X + attn_out
    normed = layer_norm(added)
    print(f"    잔차 합산 후 토큰별 표준편차: {added.std(axis=1)[:4]} ...")
    print(f"    레이어놈 후 토큰별 표준편차: {normed.std(axis=1)[:4]} ... (전부 1)")
    print("    -> 층을 아무리 쌓아도 값의 스케일이 일정하게 유지됩니다.\n")

    # [4] 피드포워드 -------------------------------------------------------
    print(f"[4] 피드포워드 — 각 토큰이 혼자 소화하는 시간")
    ff = FeedForward(D_MODEL, D_FF)
    hidden = np.maximum(normed @ ff.W1, 0)
    ff_out = hidden @ ff.W2
    print(f"    확장: {normed.shape} -> {hidden.shape} (ReLU) -> 축소: {ff_out.shape}")
    print("    -> 토큰 간 교류는 없습니다. 교류는 오직 어텐션에서만 일어납니다.\n")

    # [5] 블록 조립과 적층 -------------------------------------------------
    print("[5] 블록 조립 — '회의+정리' 세트를 쌓는다")
    block1 = TransformerBlock(D_MODEL, N_HEADS, D_FF)
    block2 = TransformerBlock(D_MODEL, N_HEADS, D_FF)
    out1 = block1(X)
    out2 = block2(out1)
    print(f"    입력 {X.shape} -> 블록1 -> {out1.shape} -> 블록2 -> {out2.shape}")
    per_block = block1.n_params()
    print(f"    블록 1개 파라미터: {per_block:,}개")
    print(f"    참고: 대형 LLM 은 이런 블록을 수십~백여 층, 차원 수천으로 키운 것입니다.\n")

    # [6] 인코더 vs 디코더 -------------------------------------------------
    print("[6] 인코더 vs 디코더 — 같은 부품, 마스크 하나의 차이")
    _, w_enc = mha(X, causal=False)
    _, w_dec = mha(X, causal=True)
    print("    인코더(양방향) 헤드0 가중치:")
    print("    " + str(w_enc[0]).replace("\n", "\n    "))
    print("    디코더(인과 마스크) 헤드0 가중치 — 오른쪽 위가 전부 0:")
    print("    " + str(w_dec[0]).replace("\n", "\n    "))
    print("    -> 디코더는 '아직 안 나온 미래 토큰'을 못 봅니다. 다음 토큰 예측의")
    print("       컨닝 방지 장치이며, GPT 의 G 를 가능하게 하는 마스크입니다.")
    print("\n결론: 트랜스포머 = (자리표 + 분과회의 + 개인정리 + 안전장치) x N층.")
