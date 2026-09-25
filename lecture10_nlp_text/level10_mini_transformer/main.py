"""
level10 — 미니 트랜스포머 직접 구현

torch 로 소형 GPT(디코더 전용 트랜스포머)를 만들어
tiny_corpus 3만 자로 '다음 글자 예측'을 학습시킵니다.
loss 하강과 생성문의 변화, 퍼플렉시티, 어텐션의 인과 구조,
loss 곡선 PNG 저장까지 시연합니다. CPU 에서 60초 이내로 끝납니다.
"""

import os
import sys
import time
import math
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SEED = 42
BLOCK = 32          # 컨텍스트 길이 (모델이 한 번에 보는 글자 수)
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
BATCH = 48
STEPS = 400
LR = 3e-3


class CausalSelfAttention(nn.Module):
    """인과 마스크가 달린 멀티헤드 셀프 어텐션 (level09 부품의 torch 판)."""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        mask = torch.triu(torch.ones(BLOCK, BLOCK), diagonal=1).bool()
        self.register_buffer("mask", mask)
        self.last_weights = None                     # 시연용: 어텐션 저장

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(D_MODEL, dim=2)
        def heads(t):                                # (B, T, C) -> (B, H, T, C/H)
            return t.view(B, T, N_HEADS, C // N_HEADS).transpose(1, 2)
        q, k, v = heads(q), heads(k), heads(v)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // N_HEADS)
        att = att.masked_fill(self.mask[:T, :T], float("-inf"))   # 미래 컨닝 방지
        att = F.softmax(att, dim=-1)
        self.last_weights = att.detach()
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    """어텐션 -> 잔차+놈 -> FF -> 잔차+놈 (Pre-LN 방식)."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(),
            nn.Linear(4 * D_MODEL, D_MODEL))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, D_MODEL)
        self.pos_embed = nn.Embedding(BLOCK, D_MODEL)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYERS)])
        self.ln_final = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_embed(idx) + self.pos_embed(pos)   # 토큰 + 자리 번호표
        x = self.blocks(x)
        return self.head(self.ln_final(x))              # (B, T, vocab)


def get_batch(ids: torch.Tensor, generator: torch.Generator):
    """말뭉치에서 임의 위치 BATCH 개를 잘라 (입력, 한 글자 민 정답) 쌍으로."""
    starts = torch.randint(0, len(ids) - BLOCK - 1, (BATCH,), generator=generator)
    x = torch.stack([ids[s: s + BLOCK] for s in starts])
    y = torch.stack([ids[s + 1: s + BLOCK + 1] for s in starts])
    return x, y


@torch.no_grad()
def generate(model, stoi, itos, prompt: str, length: int = 60,
             temperature: float = 0.7) -> str:
    model.eval()
    ids = torch.tensor([[stoi.get(ch, 0) for ch in prompt]], dtype=torch.long)
    for _ in range(length):
        logits = model(ids[:, -BLOCK:])                 # 최근 BLOCK 글자만 봄
        probs = F.softmax(logits[0, -1] / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=None)
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    model.train()
    return "".join(itos[int(i)] for i in ids[0])


if __name__ == "__main__":
    t0 = time.time()
    torch.manual_seed(SEED)
    print("=" * 70)
    print("미니 트랜스포머(GPT 구조) — 조립, 시동, 도로 주행")
    print("=" * 70 + "\n")

    # [1] 데이터 -----------------------------------------------------------
    text = hjh_data.tiny_corpus()[:30000]
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    ids = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    print(f"[1] 데이터: {len(text):,}자, 글자 어휘 {len(chars)}종")
    print(f"    학습 문제: 길이 {BLOCK} 시퀀스에서 위치마다 '다음 글자'를 동시에 예측")
    print(f"    (한 시퀀스가 {BLOCK}개의 문제 — 트랜스포머 병렬 학습의 힘)\n")

    # [2] 모델 -------------------------------------------------------------
    model = MiniGPT(len(chars))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[2] 모델: {D_MODEL}차원, {N_LAYERS}층, {N_HEADS}헤드 디코더")
    print(f"    파라미터 {n_params:,}개 — 구조는 GPT, 크기는 장난감\n")

    # [3] 학습 -------------------------------------------------------------
    print(f"[3] 학습: {STEPS} 스텝 (loss 와 생성문의 변화를 관찰)")
    prompt = "어제 "
    print(f"    step   0 | loss  -   | 생성: {generate(model, stoi, itos, prompt, 40)!r}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(SEED)
    losses = []
    for step in range(1, STEPS + 1):
        x, y = get_batch(ids, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(chars)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
        if step in (50, 150, STEPS):
            ppl = math.exp(losses[-1])
            print(f"    step {step:3d} | loss {losses[-1]:.3f} (퍼플렉시티 {ppl:5.1f}) "
                  f"| 생성: {generate(model, stoi, itos, prompt, 40)!r}")
    print(f"    -> 초기 loss ~ln({len(chars)})={math.log(len(chars)):.2f} (찍기 수준)에서")
    print(f"       내려올수록 모델이 다음 글자에 덜 '놀라게' 됩니다.\n")

    # loss 곡선 저장
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, linewidth=1, color="#3b6db4")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Mini transformer training loss")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "loss_curve.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    loss 곡선 저장: {path}\n")

    # [4] 생성 시험 --------------------------------------------------------
    print("[4] 생성 시험 — 프롬프트를 주고 이어쓰기")
    for p in ["어제 학생이 ", "주말에 요리사가 ", "아침에 "]:
        print(f"    {p!r} -> {generate(model, stoi, itos, p, 50)!r}")
    print()

    # [5] 어텐션 들여다보기 -------------------------------------------------
    print("[5] 학습된 어텐션의 인과 구조 확인 (1층 헤드0, 앞 5x5)")
    _ = model(ids[:BLOCK].unsqueeze(0))
    att = model.blocks[0].attn.last_weights[0, 0, :5, :5]
    for row in att:
        print("    " + " ".join(f"{v:.2f}" for v in row))
    print("    -> 오른쪽 위(미래)가 전부 0 — 인과 마스크가 지켜지고 있습니다.")
    print(f"\n총 실행 시간: {time.time() - t0:.1f}초 (90초 제한 안)")
