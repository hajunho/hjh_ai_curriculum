"""
LoRA(Low-Rank Adaptation) 파인튜닝을 밑바닥부터 구현하는 실습입니다.
미니 GPT 를 한국어 코퍼스로 짧게 사전학습한 뒤, 새 말투(하오체) 데이터에
적응시킬 때 전체 가중치를 다시 학습하는 대신 attention 선형층에
저랭크 어댑터(W + scale·B@A)를 붙여 그 부분만 학습합니다.
전체 파인튜닝 대비 학습 파라미터 수가 얼마나 줄어드는지 비교합니다.
"""
import pathlib
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

SEED = 42
CTX, D, H, LAYERS = 64, 64, 2, 2


class CausalSelfAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3 * d)   # LoRA 를 붙일 자리 1
        self.proj = nn.Linear(d, d)      # LoRA 를 붙일 자리 2

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        s = lambda t: t.view(B, T, self.h, C // self.h).transpose(1, 2)
        y = F.scaled_dot_product_attention(s(q), s(k), s(v), is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, h)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, D)
        self.pos = nn.Embedding(CTX, D)
        self.blocks = nn.ModuleList(Block(D, H) for _ in range(LAYERS))
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab, bias=False)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T))
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


class LoRALinear(nn.Module):
    """기존 선형층 W 는 얼려 두고, 저랭크 보정항 scale·B@A 만 학습합니다."""

    def __init__(self, base, r=4, alpha=8):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        # A 는 작은 무작위값, B 는 0 으로 시작 → 부착 직후엔 원본과 완전히 같은 출력
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.02)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)


def make_style_corpus(n=900, seed=7):
    # 사전학습 코퍼스와 같은 문형이지만 어미만 '~했소'(하오체)인 새 도메인 데이터
    rng = random.Random(seed)
    times = ["어제", "오늘", "아침에", "저녁에", "주말에"]
    subs = ["학생이", "회사원이", "요리사가", "개발자가", "선생님이"]
    objs = ["보고서를", "김치찌개를", "프로그램을", "편지를", "계획서를"]
    verbs = ["만들었소", "고쳤소", "확인했소", "정리했소", "준비했소"]
    return " ".join(f"{rng.choice(times)} {rng.choice(subs)} "
                    f"{rng.choice(objs)} {rng.choice(verbs)}." for _ in range(n))


def encode(text, stoi):
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def get_batch(data, bs, gen):
    ix = torch.randint(len(data) - CTX - 1, (bs,), generator=gen)
    x = torch.stack([data[i:i + CTX] for i in ix])
    y = torch.stack([data[i + 1:i + CTX + 1] for i in ix])
    return x, y


def train(model, data, steps, lr, gen, tag):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    for step in range(1, steps + 1):
        x, y = get_batch(data, 24, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % (steps // 4) == 0:
            print(f"    {tag} step {step:4d}/{steps}  loss {loss.item():.3f}")


@torch.no_grad()
def eval_loss(model, data, gen, iters=20):
    total = 0.0
    for _ in range(iters):
        x, y = get_batch(data, 24, gen)
        logits = model(x)
        total += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    return total / iters


@torch.no_grad()
def sample(model, stoi, itos, prompt, n=48, gen=None):
    idx = encode(prompt, stoi)[None]
    for _ in range(n):
        logits = model(idx[:, -CTX:])[:, -1] / 0.7
        nxt = torch.multinomial(logits.softmax(-1), 1, generator=gen)
        idx = torch.cat([idx, nxt], 1)
    return "".join(itos[i] for i in idx[0].tolist())


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    print("=" * 62)
    print("Level 06 — 파인튜닝과 LoRA (어댑터 직접 구현)")
    print("=" * 62)

    # [1] 데이터: 사전학습 코퍼스(평서체) + 새 말투 코퍼스(하오체)
    base_text = hjh_data.tiny_corpus()
    style_text = make_style_corpus()
    chars = sorted(set(base_text + style_text))
    stoi = {c: i for i, c in enumerate(chars)}
    base_data, style_data = encode(base_text, stoi), encode(style_text, stoi)
    print(f"[1] 코퍼스: 사전학습 {len(base_data):,}자 / 새 말투 {len(style_data):,}자 / vocab {len(chars)}")

    # [2] 베이스 모델 사전학습 — '~했다'체만 배운 상태를 만든다
    model = TinyGPT(len(chars))
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[2] 베이스 모델 사전학습 (파라미터 {n_total:,}개)")
    train(model, base_data, steps=400, lr=3e-3, gen=gen, tag="사전학습")
    print(f"    생성 샘플: {sample(model, stoi, chars, '저녁에 요리사가 ', gen=gen)!r}")

    # [3] 파인튜닝 전, 새 말투 데이터에 대한 성능
    before = eval_loss(model, style_data, gen)
    print(f"[3] 파인튜닝 전 새 말투 loss = {before:.3f} (하오체를 모른다)")

    # [4] 베이스 전체를 동결하고 attention 층에만 LoRA 부착
    for p in model.parameters():
        p.requires_grad_(False)
    for blk in model.blocks:
        blk.attn.qkv = LoRALinear(blk.attn.qkv, r=4, alpha=8)
        blk.attn.proj = LoRALinear(blk.attn.proj, r=4, alpha=8)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[4] 전체 파인튜닝이라면 {n_total:,}개를 전부 다시 학습해야 하지만,")
    print(f"    LoRA(r=4)는 {n_train:,}개만 학습합니다 → 전체의 {100 * n_train / n_total:.2f}%")

    # [5] LoRA 파인튜닝 — 어댑터(A, B)만 갱신된다
    print("[5] LoRA 파인튜닝 (베이스 가중치는 1비트도 바뀌지 않음)")
    train(model, style_data, steps=300, lr=3e-3, gen=gen, tag="LoRA")
    after = eval_loss(model, style_data, gen)
    print(f"    새 말투 loss {before:.3f} -> {after:.3f}")
    print(f"    생성 샘플: {sample(model, stoi, chars, '저녁에 요리사가 ', gen=gen)!r}")

    # [6] 병합(merge): W' = W + scale·B@A 로 합치면 추론 시 추가 비용이 0
    lora = model.blocks[0].attn.qkv
    merged = nn.Linear(lora.base.in_features, lora.base.out_features)
    with torch.no_grad():
        merged.weight.copy_(lora.base.weight + lora.scale * (lora.B @ lora.A))
        merged.bias.copy_(lora.base.bias)
        x = torch.randn(2, 5, D, generator=gen)
        same = torch.allclose(lora(x), merged(x), atol=1e-5)
    print(f"[6] 병합 검증: 어댑터 출력 == 병합된 가중치 출력 ? {same}")
    print("    배포 시엔 병합해 파일 하나로, 실험 시엔 어댑터만 갈아끼우면 됩니다.")


if __name__ == "__main__":
    main()
