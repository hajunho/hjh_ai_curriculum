"""
지시학습 SFT(Supervised Fine-Tuning)를 미니어처로 구현하는 실습입니다.
사전학습만 마친 베이스 모델은 문장을 '이어쓰기'만 할 뿐 질문에 답하지 않습니다.
챗 템플릿(<|user|>/<|assistant|>/<|end|>)으로 지시-응답 데이터를 만들고,
응답 부분에만 loss 를 계산하는 마스킹(loss masking)으로
'지시를 따르는 형식'을 가르치는 과정을 학습 전/후 생성으로 비교합니다.
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
PAD, U, A, E = "<pad>", "<|user|>", "<|assistant|>", "<|end|>"


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


class CharTok:
    """특수 토큰(템플릿 구분자)은 통째로 1개 토큰, 나머지는 글자 단위."""

    def __init__(self, text):
        self.itos = [PAD, U, A, E] + sorted(set(text))
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


NAMES = ["민수", "지현", "철수", "영희", "수진", "동현", "하늘", "보라"]
ITEMS = [("연필", 12), ("공책", 7), ("지우개", 30), ("가위", 5), ("풀", 21), ("자", 9)]


def build_sft_data():
    data = [(f"{n}에게 인사해", f"{n}님, 안녕하세요! 좋은 하루 보내세요.") for n in NAMES]
    for a in range(1, 6):
        for b in range(1, 6):
            if a != b:  # a == b 조합은 '본 적 없는 지시' 시험용으로 남겨 둔다
                data.append((f"{a} 더하기 {b}는?", f"{a} 더하기 {b}는 {a + b}입니다."))
    data += [(f"{it} 재고 알려줘", f"{it} 재고는 {cnt}개입니다.") for it, cnt in ITEMS]
    return data


def render(tok, instr, resp=None):
    # 챗 템플릿: <|user|>지시<|assistant|>응답<|end|>
    ids = [tok.stoi[U]] + tok.encode(instr) + [tok.stoi[A]]
    if resp is not None:
        ids += tok.encode(resp) + [tok.stoi[E]]
    return ids


def make_sft_tensors(tok, data):
    # x=ids[:-1], y=ids[1:] 로 다음 토큰 예측을 만들되,
    # 프롬프트(지시) 구간의 정답은 -100 으로 가려 loss 에서 제외한다.
    rows = [(render(tok, i, r), len(render(tok, i))) for i, r in data]
    L = max(len(ids) for ids, _ in rows) - 1
    X = torch.full((len(rows), L), tok.stoi[PAD], dtype=torch.long)
    Y = torch.full((len(rows), L), -100, dtype=torch.long)
    for k, (ids, plen) in enumerate(rows):
        x, y = ids[:-1], ids[1:]
        X[k, :len(x)] = torch.tensor(x)
        for i in range(plen - 1, len(y)):  # 응답 토큰(과 <|end|>)만 정답으로 남긴다
            Y[k, i] = y[i]
    return X, Y


def get_batch(data, bs, gen):
    ix = torch.randint(len(data) - CTX - 1, (bs,), generator=gen)
    return (torch.stack([data[i:i + CTX] for i in ix]),
            torch.stack([data[i + 1:i + CTX + 1] for i in ix]))


@torch.no_grad()
def continue_text(model, tok, prompt, n=40, gen=None):
    idx = torch.tensor(tok.encode(prompt))[None]
    for _ in range(n):
        logits = model(idx[:, -CTX:])[:, -1] / 0.7
        nxt = torch.multinomial(logits.softmax(-1), 1, generator=gen)
        idx = torch.cat([idx, nxt], 1)
    return tok.decode(idx[0].tolist())


@torch.no_grad()
def answer(model, tok, instr, max_new=44):
    # 템플릿을 채워 넣고 <|end|> 가 나올 때까지 탐욕적(greedy) 생성
    idx = torch.tensor(render(tok, instr))[None]
    out = []
    for _ in range(max_new):
        nxt = int(model(idx[:, -CTX:])[:, -1].argmax())
        if nxt == tok.stoi[E]:
            break
        out.append(nxt)
        idx = torch.cat([idx, torch.tensor([[nxt]])], 1)
    return tok.decode(out)


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    print("=" * 62)
    print("Level 07 — 지시학습(SFT): 이어쓰기 기계를 비서로 만들기")
    print("=" * 62)

    # [1] 토크나이저: 코퍼스 + 지시 데이터 전체 글자 + 특수 토큰 4개
    sft_data = build_sft_data()
    corpus = hjh_data.tiny_corpus()
    all_text = corpus + "".join(i + r for i, r in sft_data) + "0123456789 더하기는?"
    tok = CharTok(all_text)
    print(f"[1] 지시-응답 데이터 {len(sft_data)}쌍 / vocab {len(tok.itos)} (특수 토큰 4개 포함)")

    # [2] 베이스 모델 사전학습 — 오직 '다음 글자 이어쓰기'만 배운다
    model = TinyGPT(len(tok.itos))
    corpus_ids = torch.tensor(tok.encode(corpus))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    print("[2] 베이스 모델 사전학습 (300 스텝)")
    for step in range(1, 301):
        x, y = get_batch(corpus_ids, 24, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step in (1, 150, 300):
            print(f"    step {step:3d}  loss {loss.item():.3f}")

    # [3] 베이스 모델의 한계: 이어쓰기는 잘하지만 지시는 못 따른다
    print("[3] 베이스 모델은 '이어쓰기'만 합니다")
    print(f"    이어쓰기  : {continue_text(model, tok, '오늘 회사원이 ', gen=gen)!r}")
    print(f"    지시를 주면: {answer(model, tok, '3 더하기 5는?')!r}  <- 답이 아니라 아무 글자")

    # [4] 챗 템플릿과 loss 마스킹 시각화
    instr, resp = sft_data[10]
    ids, plen = render(tok, instr, resp), len(render(tok, instr))
    marks = "".join("-" if i < plen else "#" for i in range(len(ids)))
    print("[4] 챗 템플릿과 loss 마스킹 (-: loss 제외 / #: loss 계산)")
    print(f"    토큰: {U}{instr}{A}{resp}{E}")
    print(f"    마스크: {marks}  (프롬프트 {plen}토큰 제외, 응답 {len(ids) - plen}토큰만 학습)")

    # [5] SFT: 응답 토큰에만 cross entropy(ignore_index=-100)
    X, Y = make_sft_tensors(tok, sft_data)
    print(f"[5] SFT 학습 (배치 {X.shape}, 700 스텝)")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(1, 701):
        ix = torch.randint(len(X), (16,), generator=gen)
        logits = model(X[ix])
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y[ix].view(-1),
                               ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
        if step in (1, 350, 700):
            print(f"    step {step:3d}  loss {loss.item():.3f}")

    # [6] SFT 후: 같은 지시에 대한 응답 비교
    print("[6] SFT 후 응답 (greedy)")
    for q, note in [("3 더하기 5는?", "학습에 있던 지시"),
                    ("공책 재고 알려줘", "학습에 있던 지시"),
                    ("4 더하기 4는?", "본 적 없는 조합(a=b) — 형식은 지키는지 보세요")]:
        print(f"    Q: {q:<14s} A: {answer(model, tok, q)!r}  ({note})")
    print("    형식(템플릿·존댓말·<|end|>)은 확실히 배우지만, 내용의 일반화는")
    print("    모델·데이터가 커져야 나옵니다. 이것이 스케일이 필요한 이유입니다.")


if __name__ == "__main__":
    main()
