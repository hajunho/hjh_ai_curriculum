"""
DPO(Direct Preference Optimization) 선호도 정렬을 수식 그대로 구현합니다.
같은 질문에 대한 두 응답(chosen/rejected) 중 사람이 고른 쪽의 확률은 높이고
반대쪽은 낮추되, 참조(reference) 모델에서 너무 멀어지지 않게 묶어 둡니다.
보상 모델도 강화학습도 없이 분류 손실 하나로 정렬이 일어나는 과정을
chosen/rejected 로그확률과 margin 의 변화 표로 확인합니다.
"""
import copy
import pathlib
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
CTX, D, H, LAYERS = 96, 64, 2, 2
U, A, E = "<|user|>", "<|assistant|>", "<|end|>"
BETA = 0.1

# 고객 응대 선호 데이터: (질문, chosen=공손한 응답, rejected=책임 회피 응답)
PAIRS = [
    ("배송이 왜 이렇게 늦어요", "불편을 드려 죄송합니다. 배송 상태를 바로 확인해 드리겠습니다.",
     "그건 택배 회사에 물어보세요."),
    ("환불 받고 싶어요", "네, 바로 도와드리겠습니다. 주문 번호를 알려 주시겠어요?",
     "환불 규정을 먼저 읽어 보세요."),
    ("제품이 오자마자 고장났어요", "정말 죄송합니다. 새 제품으로 즉시 교환해 드리겠습니다.",
     "사용법을 잘못 보신 것 같은데요."),
    ("문의 답이 없어요", "늦어져서 죄송합니다. 지금 바로 답변을 드리겠습니다.",
     "저희도 바쁘니 기다려 보세요."),
    ("포인트가 사라졌어요", "확인 후 바로 복구해 드리겠습니다. 죄송합니다.",
     "포인트는 원래 없어지기도 합니다."),
    ("주소를 잘못 입력했어요", "걱정 마세요. 배송 전이니 주소를 바로 바꿔 드리겠습니다.",
     "그건 고객님 실수라서 어렵습니다."),
]


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        s = lambda t: t.view(B, T, self.h, C // self.h).transpose(1, 2)
        y = F.scaled_dot_product_attention(s(q), s(k), s(v), is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, D), nn.Embedding(CTX, D)
        self.blocks = nn.ModuleList(Block(D, H) for _ in range(LAYERS))
        self.lnf, self.head = nn.LayerNorm(D), nn.Linear(D, vocab, bias=False)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


def build_tok():
    text = "".join(q + w + l for q, w, l in PAIRS)
    itos = [U, A, E] + sorted(set(text))
    return itos, {s: i for i, s in enumerate(itos)}


def prompt_ids(stoi, q):
    return [stoi[U]] + [stoi[c] for c in q] + [stoi[A]]


def resp_ids(stoi, r):
    return [stoi[c] for c in r] + [stoi[E]]


def resp_logp(model, p_ids, r_ids):
    """프롬프트가 주어졌을 때 응답 전체의 로그확률 log π(y|x) (토큰 logp 의 합)."""
    idx = torch.tensor(p_ids + r_ids)[None]
    logps = F.log_softmax(model(idx[:, :-1]), dim=-1)[0]
    tgt = idx[0, 1:]
    start = len(p_ids) - 1  # 이 위치부터의 예측이 응답 토큰을 맞히는 예측
    pos = torch.arange(start, len(tgt))
    return logps[pos, tgt[start:]].sum()


@torch.no_grad()
def greedy_answer(model, stoi, itos, q, max_new=40):
    idx = torch.tensor(prompt_ids(stoi, q))[None]
    out = []
    for _ in range(max_new):
        nxt = int(model(idx[:, -CTX:])[:, -1].argmax())
        if nxt == stoi[E]:
            break
        out.append(nxt)
        idx = torch.cat([idx, torch.tensor([[nxt]])], 1)
    return "".join(itos[i] for i in out)


@torch.no_grad()
def pref_table(policy, ref, enc, title):
    print(title)
    print("    질문                logπ(chosen)  logπ(rejected)  margin(vs ref)")
    n_correct = 0
    for q, p, w, l in enc:
        pw, pl = resp_logp(policy, p, w).item(), resp_logp(policy, p, l).item()
        rw, rl = resp_logp(ref, p, w).item(), resp_logp(ref, p, l).item()
        margin = (pw - rw) - (pl - rl)
        n_correct += pw > pl
        print(f"    {q:<12s}{pw:12.1f}{pl:14.1f}{margin:14.2f}")
    print(f"    -> chosen 이 더 그럴듯한 쌍: {n_correct}/{len(enc)}")


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    print("=" * 66)
    print("Level 08 — 선호도 정렬 DPO (loss 수식을 그대로 구현)")
    print("=" * 66)

    # [1] 토크나이저와 데이터 인코딩
    itos, stoi = build_tok()
    enc = [(q, prompt_ids(stoi, q), resp_ids(stoi, w), resp_ids(stoi, l))
           for q, w, l in PAIRS]
    print(f"[1] 선호 쌍 {len(enc)}개 / vocab {len(itos)}")

    # [2] 준비 학습(SFT): chosen 과 rejected 를 '똑같이' 가르쳐 둘 다 그럴듯하게 만든다
    #     (현실에서도 DPO 는 SFT 를 마친 모델에서 출발합니다)
    policy = TinyGPT(len(itos))
    opt = torch.optim.AdamW(policy.parameters(), lr=2e-3)
    print("[2] 준비 SFT: 두 스타일을 같은 비중으로 학습 (아직 선호 없음)")
    for epoch in range(1, 81):
        total = torch.tensor(0.0)
        for _, p, w, l in enc:
            for r in (w, l):
                idx = torch.tensor(p + r)[None]
                logits = policy(idx[:, :-1])
                y = idx[0, 1:].clone()
                y[:len(p) - 1] = -100  # 응답 부분만 loss (level07 과 동일)
                total = total + F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y, ignore_index=-100)
        total = total / (2 * len(enc))
        opt.zero_grad(); total.backward(); opt.step()
        if epoch in (1, 40, 80):
            print(f"    epoch {epoch:2d}  loss {total.item():.3f}")

    # [3] 참조 모델 고정: 정렬 전의 자신을 복사해 얼려 둔다
    ref = copy.deepcopy(policy)
    for prm in ref.parameters():
        prm.requires_grad_(False)
    print("[3] reference = policy 복사본(동결). 시작 시점 margin 은 정확히 0 입니다.")
    pref_table(policy, ref, enc, "    --- DPO 학습 전 ---")
    q0 = PAIRS[0][0]
    print(f"    greedy 응답: {greedy_answer(policy, stoi, itos, q0)!r}")

    # [4] DPO 학습: loss = -log σ( β·[(logπ(yw)-logπref(yw)) - (logπ(yl)-logπref(yl))] )
    with torch.no_grad():  # 참조 모델의 로그확률은 변하지 않으므로 한 번만 계산
        ref_lp = [(resp_logp(ref, p, w), resp_logp(ref, p, l)) for _, p, w, l in enc]
    opt = torch.optim.AdamW(policy.parameters(), lr=2e-4)
    print(f"[4] DPO 학습 (beta={BETA})")
    for epoch in range(1, 31):
        loss = torch.tensor(0.0)
        for (_, p, w, l), (rw, rl) in zip(enc, ref_lp):
            pw, pl = resp_logp(policy, p, w), resp_logp(policy, p, l)
            logits = BETA * ((pw - rw) - (pl - rl))  # DPO 수식의 괄호 안
            loss = loss - F.logsigmoid(logits)
        loss = loss / len(enc)
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch in (1, 10, 20, 30):
            print(f"    epoch {epoch:2d}  dpo_loss {loss.item():.4f}")

    # [5] 정렬 결과: chosen ↑ / rejected ↓, margin 이 벌어진다
    pref_table(policy, ref, enc, "[5] --- DPO 학습 후 ---")
    print(f"    greedy 응답: {greedy_answer(policy, stoi, itos, q0)!r}")
    print("[6] RLHF 는 보상 모델 + PPO 4개 모델이 필요하지만, DPO 는 위 손실 한 줄로")
    print("    같은 목표를 달성합니다. 이것이 DPO 가 사랑받는 이유입니다.")


if __name__ == "__main__":
    main()
