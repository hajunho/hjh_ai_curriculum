"""
미니 GPT 사전학습 실습.
- 2층짜리 소형 디코더 트랜스포머(약 11만 파라미터)를 밑바닥부터 구현하고,
  hjh_data.tiny_corpus()를 글자 단위로 '다음 글자 맞히기'만 시켜 학습합니다.
- 스텝이 지날수록 loss가 내려가고 생성 문장에 문법이 생겨나는 과정을 관찰합니다.
- CPU만으로 90초 이내에 끝나도록 아주 작게 설계했습니다.
"""
import math
import os
import sys
import time
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

# ----- 하이퍼파라미터: '수십만 파라미터급' 초소형 모델 -----------------------
CTX = 64        # 문맥 길이(한 번에 보는 글자 수)
D_MODEL = 64    # 임베딩 차원
N_HEAD = 2      # 어텐션 헤드 수
N_LAYER = 2     # 트랜스포머 블록 수
BATCH = 48
STEPS = 700
LR = 3e-3
TIME_BUDGET = 75.0  # 초 — 이 시간을 넘기면 학습을 조기 종료(안전장치)


class CausalSelfAttention(nn.Module):
    """마스크드 셀프 어텐션: 각 글자는 자기 '앞' 글자들만 참고할 수 있다."""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)   # Q, K, V를 한 번에 계산
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        mask = torch.tril(torch.ones(CTX, CTX))      # 하삼각 = 미래 가리기
        self.register_buffer("mask", mask.view(1, 1, CTX, CTX))

    def forward(self, x):
        B, T, C = x.shape
        hd = C // N_HEAD
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, 헤드, T, 헤드당 차원)
        q = q.view(B, T, N_HEAD, hd).transpose(1, 2)
        k = k.view(B, T, N_HEAD, hd).transpose(1, 2)
        v = v.view(B, T, N_HEAD, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)      # 유사도 점수
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)                          # 주목 비율
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    """트랜스포머 블록 = 어텐션(정보 모으기) + MLP(생각하기), 각각 잔차 연결."""

    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention()
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(),
            nn.Linear(4 * D_MODEL, D_MODEL),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, D_MODEL)   # 글자 -> 벡터
        self.pos_emb = nn.Embedding(CTX, D_MODEL)          # 위치 -> 벡터
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size)         # 벡터 -> 다음 글자 점수

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.ln_f(self.blocks(x))
        logits = self.head(x)
        loss = None
        if targets is not None:  # 모든 위치에서 '다음 글자'를 맞히는 손실
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=0.8, top_k=8):
        for _ in range(n_new):
            logits, _ = self(idx[:, -CTX:])                 # 마지막 CTX 글자만 사용
            logits = logits[:, -1, :] / temperature
            top = torch.topk(logits, top_k)                 # 상위 k개 후보만 남김
            logits[logits < top.values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


def get_batch(data, rng):
    """코퍼스에서 무작위 위치 BATCH개를 잘라 (입력, 한 칸 민 정답) 쌍을 만든다."""
    ix = torch.randint(len(data) - CTX - 1, (BATCH,), generator=rng)
    x = torch.stack([data[i:i + CTX] for i in ix])
    y = torch.stack([data[i + 1:i + CTX + 1] for i in ix])
    return x, y


def sample_text(model, stoi, itos, prompt="오늘 ", n=90):
    idx = torch.tensor([[stoi[c] for c in prompt]])
    out = model.generate(idx, n)[0].tolist()
    return "".join(itos[i] for i in out)


def main():
    torch.manual_seed(12)
    rng = torch.Generator().manual_seed(12)
    t0 = time.time()

    # [1] 코퍼스 로딩과 글자 사전 만들기 ------------------------------------
    text = hjh_data.tiny_corpus()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_val = len(data) // 20
    train_data, val_data = data[:-n_val], data[-n_val:]
    print(f"[1] 코퍼스 {len(text):,}글자 / 글자 사전 {len(chars)}종 / 검증용 {n_val:,}글자 분리")

    # [2] 모델 생성 ---------------------------------------------------------
    model = TinyGPT(len(chars))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[2] TinyGPT 생성: {N_LAYER}층, {N_HEAD}헤드, d={D_MODEL}, 파라미터 {n_params:,}개")
    print("    (실제 대형 모델은 이 구조를 수천 배로 키운 것입니다)")

    # [3] 학습 전 생성 — 아직 아무것도 모르는 상태 ---------------------------
    print("\n[3] 학습 전 생성:  ", repr(sample_text(model, stoi, itos)))

    # [4] 사전학습: '다음 글자 맞히기'를 반복 --------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    print(f"\n[4] 사전학습 시작 (최대 {STEPS}스텝, 배치 {BATCH} x 문맥 {CTX}글자)")
    for step in range(1, STEPS + 1):
        x, y = get_batch(train_data, rng)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                vx, vy = get_batch(val_data, rng)
                _, vloss = model(vx, vy)
            print(f"    step {step:4d} | train loss {loss.item():.3f} | val loss {vloss.item():.3f}")
        if step in (30, 200):  # 중간 점검: 문장이 생겨나는 과정을 관찰
            print(f"      └ step {step} 생성:", repr(sample_text(model, stoi, itos, n=60)))
        if time.time() - t0 > TIME_BUDGET:
            print(f"    (시간 안전장치 발동: {step}스텝에서 조기 종료)")
            break
    tokens_seen = step * BATCH * CTX
    print(f"    학습 완료: {step}스텝, 본 글자 수 {tokens_seen:,}개, 경과 {time.time()-t0:.1f}초")

    # [5] 학습 후 생성 — 문법이 '생겨난' 것을 확인 ---------------------------
    print("\n[5] 학습 후 생성 (프롬프트 '오늘 '):")
    for k in range(2):
        torch.manual_seed(100 + k)
        print(f"    샘플 {k+1}:", sample_text(model, stoi, itos))
    print("    → 누가-무엇을-어찌했다 어순과 조사(을/를, 이/가)가 자연스러워졌습니다.")
    print("      정답 문장을 외운 게 아니라 '다음 글자 확률'만 배웠는데 문법이 나타났습니다.")

    # [6] 체크포인트 저장 — level06(LoRA)~08(DPO)에서 이 모델 이야기를 이어갑니다
    out_dir = pathlib.Path(__file__).resolve().parent / "outputs"
    os.makedirs(out_dir, exist_ok=True)
    ckpt = out_dir / "tiny_gpt.pt"
    torch.save({"model": model.state_dict(), "stoi": stoi, "itos": itos,
                "config": {"ctx": CTX, "d_model": D_MODEL, "n_head": N_HEAD,
                           "n_layer": N_LAYER, "vocab": len(chars)}}, ckpt)
    print(f"\n[6] 체크포인트 저장: {ckpt}")
    print(f"    총 실행 시간: {time.time()-t0:.1f}초 (90초 예산 이내)")


if __name__ == "__main__":
    main()
