"""
level11 — 비전 트랜스포머(ViT)와 멀티모달

이미지를 '패치 단어'의 문장으로 읽는 과정을 직접 구현합니다.
  1) 16x16 이미지를 4x4 패치 16개로 쪼개기 (토큰화)
  2) 패치 임베딩 + 위치 인코딩
  3) 셀프 어텐션을 행렬 연산으로 직접 구현 (Q,K,V)
  4) 미니 ViT 를 도형 분류로 학습시켜 '어디를 보는가' 어텐션 맵 시각화
"""

import os
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
CLASS_EN = ["square", "circle", "triangle"]
PATCH = 4              # 패치 한 변 (16x16 -> 4x4 패치 16개)
N_PATCH = (16 // PATCH) ** 2
DIM = 32               # 임베딩 차원
EPOCHS = 80


def patchify(img: np.ndarray) -> np.ndarray:
    """이미지를 패치 단어들로 쪼갠다: (16,16) -> (16, 4*4)."""
    h = img.shape[0] // PATCH
    out = img.reshape(h, PATCH, h, PATCH).transpose(0, 2, 1, 3)
    return out.reshape(N_PATCH, PATCH * PATCH)


class MiniViT(nn.Module):
    """장난감 ViT: 패치 임베딩 -> 셀프 어텐션 1층 -> 평균 풀 -> 분류.
    어텐션은 nn.MultiheadAttention 없이 행렬 연산으로 직접 구현합니다."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(PATCH * PATCH, DIM)         # 패치 -> 임베딩 벡터
        self.pos = nn.Parameter(torch.zeros(N_PATCH, DIM))  # 위치 인코딩(학습형)
        self.wq = nn.Linear(DIM, DIM, bias=False)           # Query: '내가 찾는 것'
        self.wk = nn.Linear(DIM, DIM, bias=False)           # Key:   '내가 가진 것'
        self.wv = nn.Linear(DIM, DIM, bias=False)           # Value: '내가 전달할 내용'
        self.ff = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU())
        self.head = nn.Linear(DIM, 3)

    def forward(self, x, return_attn: bool = False):
        # x: (B, 16패치, 16픽셀값) — 패치 단어들의 문장
        t = self.embed(x) + self.pos                        # 단어 임베딩 + 위치
        q, k, v = self.wq(t), self.wk(t), self.wv(t)
        scores = q @ k.transpose(-2, -1) / (DIM ** 0.5)     # 모든 단어쌍의 관련도
        attn = torch.softmax(scores, dim=-1)                # 행마다 합 1 (주목 배분)
        t = t + attn @ v                                    # 관련도만큼 정보 섞기
        t = t + self.ff(t)
        logits = self.head(t.mean(dim=1))                   # 문장 요약 -> 3클래스
        return (logits, attn) if return_attn else logits


def to_tokens(X: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.stack([patchify(im) for im in X]))


def main() -> None:
    t0 = time.time()
    torch.manual_seed(11)
    np.random.seed(11)  # seed 고정(재현성)

    # ------------------------------------------------------------------
    print("[1] 이미지를 '패치 단어'로 토큰화 — 문장처럼 읽기 위한 준비")
    X, y = hjh_data.shape_images(n=750, size=16, seed=13)
    sample = X[0]
    tokens = patchify(sample)
    print(f"    16x16 이미지 -> {PATCH}x{PATCH} 패치 {N_PATCH}개, 각 패치는 숫자 {PATCH * PATCH}개")
    print(f"    tokens.shape = {tokens.shape}  (단어 16개짜리 문장과 같은 구조)")
    bright = tokens.mean(axis=1).round(2)
    print("    패치별 평균 밝기 (4x4 배열로 보면 도형의 대략 위치가 남아 있음):")
    for r in range(4):
        print("      " + " ".join(f"{v:.2f}" for v in bright[r * 4:(r + 1) * 4]))

    # ------------------------------------------------------------------
    print("\n[2] 패치 임베딩 + 위치 인코딩")
    print(f"    Linear({PATCH * PATCH} -> {DIM}): 패치 하나를 {DIM}차원 벡터('단어 임베딩')로.")
    print("    순서를 섞으면 같은 단어 집합이 되므로, 패치마다 위치 벡터를 더해")
    print("    '몇 번째 자리의 단어인지'를 새깁니다 (lecture10 의 위치 인코딩과 같은 이유).")

    # ------------------------------------------------------------------
    print("\n[3] 셀프 어텐션 — 모든 패치 쌍의 관련도를 한 번에")
    print("    scores = Q @ K^T / sqrt(d)  ->  softmax  ->  attn @ V")
    print("    각 패치가 '나머지 15개 패치 중 누구를 얼마나 참고할지'를 스스로 정합니다.")
    print("    CNN 이 '옆집부터 차근차근'이라면, 어텐션은 '첫 층부터 전 지역 회의'입니다.")

    # ------------------------------------------------------------------
    print("\n[4] 미니 ViT 학습 — 어텐션이 실제로 도형을 주목하는지 확인")
    xt, yt = to_tokens(X[:600]), torch.from_numpy(y[:600])
    xe, ye = to_tokens(X[600:]), torch.from_numpy(y[600:])
    model = MiniViT()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()
        if epoch % 16 == 0:
            with torch.no_grad():
                acc = float((model(xe).argmax(1) == ye).float().mean())
            print(f"      epoch {epoch:>3}/{EPOCHS}  test_acc={acc:.3f}")
    with torch.no_grad():
        acc = float((model(xe).argmax(1) == ye).float().mean())
    print(f"    미니 ViT ({n_params:,} 파라미터) 테스트 정확도: {acc:.3f}")
    print("    (같은 과제의 CNN 과 견줄 만하지만, ViT 의 진가는 대규모 데이터에서 나옵니다)")

    # ------------------------------------------------------------------
    print("\n[5] '어디를 보는가' — 학습된 어텐션 맵 시각화")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for row, c in enumerate(range(3)):
        i = int(np.where(y[600:] == c)[0][0]) + 600
        img = X[i]
        with torch.no_grad():
            _, attn = model(to_tokens(img[None]), return_attn=True)
        attn = attn[0]                                     # (16,16): 패치별 주목 배분
        # 모든 패치가 평균적으로 어디를 참고했나 -> 4x4 -> 16x16 확대
        avg_attn = attn.mean(dim=0).reshape(4, 4).numpy()
        heat = np.kron(avg_attn, np.ones((PATCH, PATCH)))
        axes[row][0].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[row][0].set_title(f"{CLASS_EN[c]} (input)", fontsize=9)
        axes[row][1].imshow(heat, cmap="magma")
        axes[row][1].set_title("mean attention", fontsize=9)
        axes[row][2].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[row][2].imshow(heat, cmap="magma", alpha=0.55)
        axes[row][2].set_title("overlay", fontsize=9)
        for ax in axes[row]:
            ax.axis("off")
        top = int(avg_attn.argmax())
        print(f"    {CLASS_EN[c]:>9}: 가장 주목받은 패치 = ({top // 4},{top % 4}) "
              f"(4x4 격자 좌표, 도형이 있는 자리인지 PNG 로 확인)")
    fig.suptitle("Mini ViT: where do the patches attend?", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "vit_attention.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    저장 완료: {path}")
    print("    -> 밝은 영역 = 모델이 판단에 많이 참고한 패치. 도형 주변에 몰려 있습니다.")

    print(f"\n[정리] ViT = 패치 토큰화 + 임베딩 + 셀프 어텐션 (소요 {time.time() - t0:.1f}초).")
    print("       멀티모달(CLIP 식)은 여기서 한 걸음 더: 이미지 임베딩과 텍스트 임베딩을")
    print("       같은 공간에 두고 '가까우면 같은 뜻'이 되도록 학습합니다 (README 참고).")
    print("       lecture09 완주를 축하합니다 — 픽셀에서 트랜스포머까지 왔습니다.")


if __name__ == "__main__":
    main()
