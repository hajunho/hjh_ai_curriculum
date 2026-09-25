"""
level08 — 어텐션 메커니즘

scaled dot-product attention 을 numpy 로 직접 구현합니다.
장난감 예제로 중간 계산을 전부 확인하고, '눈'(eye/snow) 중의성 문장
두 개의 셀프 어텐션을 계산해 "같은 단어가 문장에 따라 다른 곳을
주목하는" 모습을 히트맵 PNG 로 저장합니다.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SEED = 7


def set_korean_font() -> None:
    candidates = ["AppleGothic", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# [1] 어텐션 구현 — 핵심은 5줄
# ---------------------------------------------------------------------------

def softmax(x: np.ndarray) -> np.ndarray:
    """행 방향 softmax. 큰 값을 빼서 오버플로 방지."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    """scaled dot-product attention. 반환: (출력, 가중치 행렬)."""
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)     # 1) 질의-키 관련도  2) 스케일
    weights = softmax(scores)           # 3) 주목도 배분표 (행 합 = 1)
    return weights @ V, weights         # 4) 값의 가중 평균


# ---------------------------------------------------------------------------
# [2] 장난감 예제 — 숫자를 전부 눈으로 확인
# ---------------------------------------------------------------------------

def demo_toy() -> None:
    print("[1]-[2] 어텐션 5줄 구현 + 토큰 3개 장난감 예제")
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(3, 4))                    # 토큰 3개, 4차원 임베딩
    Wq, Wk, Wv = (rng.normal(size=(4, 4)) for _ in range(3))
    Q, K, V = X @ Wq, X @ Wk, X @ Wv               # Q/K/V 는 X 에 행렬 곱 한 번씩
    scores = Q @ K.T
    scaled = scores / np.sqrt(K.shape[-1])
    weights = softmax(scaled)
    out = weights @ V
    np.set_printoptions(precision=2, suppress=True)
    print(f"    점수 QK^T:\n{scores}")
    print(f"    스케일 후 (÷√4):\n{scaled}")
    print(f"    softmax 가중치 (각 행 합 = {weights.sum(axis=1)}):\n{weights}")
    print(f"    출력 (가중치 @ V) shape: {out.shape}")
    print("    -> '관련도 계산 -> 확률화 -> 가중 평균' 이 전부입니다.\n")


# ---------------------------------------------------------------------------
# [3] '눈' 중의성 문장의 셀프 어텐션
#     학습된 임베딩 대신, 주제(topic) 성분을 심은 수제 임베딩을 사용합니다.
#     눈 = eye 성분 + snow 성분을 반씩 가진 중의어로 설계.
# ---------------------------------------------------------------------------

SENT_EYE = ["눈이", "침침해서", "안과에", "갔다"]
SENT_SNOW = ["눈이", "펑펑", "내려서", "길이", "막혔다"]


def build_embeddings() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    dim = 16
    eye_topic = rng.normal(size=dim)               # '신체' 주제 축
    snow_topic = rng.normal(size=dim)              # '날씨' 주제 축
    def vec(topic_mix, noise_scale=0.45):
        return topic_mix + rng.normal(size=dim) * noise_scale
    return {
        "눈이": vec(0.5 * eye_topic + 0.5 * snow_topic),   # 중의어: 두 주제 반반
        "침침해서": vec(eye_topic), "안과에": vec(eye_topic),
        "갔다": vec(np.zeros(dim), 0.8),
        "펑펑": vec(snow_topic), "내려서": vec(snow_topic),
        "길이": vec(np.zeros(dim), 0.8), "막혔다": vec(np.zeros(dim), 0.8),
    }


def self_attention_of(tokens: list[str], emb: dict) -> np.ndarray:
    """해석을 명확히 하기 위해 Q=K=V=X 인 가장 단순한 셀프 어텐션."""
    X = np.stack([emb[t] for t in tokens])
    _, weights = attention(X, X, X)
    return weights


def demo_ambiguity(emb: dict) -> tuple[np.ndarray, np.ndarray]:
    print("[3] 같은 '눈이', 문장에 따라 다른 곳을 주목한다 (셀프 어텐션)")
    results = []
    for tokens in (SENT_EYE, SENT_SNOW):
        W = self_attention_of(tokens, emb)
        results.append(W)
        row = W[0]                                  # '눈이' 행의 주목도
        pairs = ", ".join(f"{t}({w:.2f})" for t, w in zip(tokens, row))
        print(f"    문장: {' '.join(tokens)}")
        print(f"      '눈이'의 주목도: {pairs}")
    print("    (참고: 자기 자신의 주목도가 큰 것은 Q=K 로 두어 자기 유사도가")
    print("     최대가 되기 때문입니다. 실제 모델은 W_Q, W_K 를 학습해 조절합니다.)")
    print("    -> 안과 문장에서는 침침/안과에, 폭설 문장에서는 펑펑/내려서에")
    print("       주목합니다. '눈'의 표현이 문맥 단어들로 다시 조립되는 것 —")
    print("       level00 의 중의성 문제를 어텐션이 이렇게 다룹니다.\n")
    return results[0], results[1]


def save_heatmaps(W_eye: np.ndarray, W_snow: np.ndarray) -> None:
    print("[4] 어텐션 가중치 히트맵 저장")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, W, tokens, title in (
        (axes[0], W_eye, SENT_EYE, "eye: 눈이 침침해서 안과에 갔다"),
        (axes[1], W_snow, SENT_SNOW, "snow: 눈이 펑펑 내려서 길이 막혔다"),
    ):
        im = ax.imshow(W, cmap="Blues", vmin=0, vmax=W.max())
        ax.set_xticks(range(len(tokens)), tokens, rotation=45)
        ax.set_yticks(range(len(tokens)), tokens)
        ax.set_title(title, fontsize=10)
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                ax.text(j, i, f"{W[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if W[i, j] < 0.5 * W.max() else "white")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Self-attention weights (row = attending token)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "attention_heatmap.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    저장: {path}")
    print("    읽는 법: 행 = 주목하는 토큰, 열 = 주목받는 토큰, 행 합 = 1.\n")


def demo_cost() -> None:
    print("[5] 어텐션의 비용 — 모든 토큰이 모든 토큰을 본다 (n^2)")
    for n in (10, 100, 1000, 10000):
        cells = n * n
        mb = cells * 4 / 1e6                        # float32 기준
        print(f"    토큰 {n:6,d}개 -> 가중치 행렬 {cells:12,d}칸 (~{mb:8.1f} MB/층/헤드)")
    print("    -> LLM 컨텍스트 길이 제한과 긴 입력 요금의 근본 원인입니다.")
    print("\n결론: '관련도만큼 가중 평균' 한 줄이 현대 AI 의 심장입니다.")


if __name__ == "__main__":
    print("=" * 70)
    print("어텐션 — 회의에서 발언자마다 주목도를 다르게")
    print("=" * 70 + "\n")
    np.random.seed(SEED)
    set_korean_font()
    demo_toy()
    emb = build_embeddings()
    W_eye, W_snow = demo_ambiguity(emb)
    save_heatmaps(W_eye, W_snow)
    demo_cost()
