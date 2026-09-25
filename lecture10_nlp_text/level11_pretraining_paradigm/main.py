"""
level11 — 사전학습 패러다임: BERT 와 GPT

같은 미니 트랜스포머로 두 가지 사전학습 과제를 대비합니다.
  A. 빈칸 채우기(MLM, BERT 방식): 양방향 어텐션 + 가린 단어 복원
  B. 이어쓰기(CLM, GPT 방식): 인과 마스크 + 다음 단어 예측
마지막으로 '사전학습 -> 미세조정'의 가치를 리뷰 감성 분류로 증명합니다.
전체 학습 합계가 CPU 30초 이내로 끝나도록 아주 작게 설계했습니다.
"""

import sys
import time
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

SEED = 42
D_MODEL, N_HEADS, N_LAYERS = 32, 2, 2


class MiniTransformer(nn.Module):
    """causal=True 면 GPT 식 디코더, False 면 BERT 식 양방향 인코더."""

    def __init__(self, vocab_size: int, block: int, causal: bool):
        super().__init__()
        self.block, self.causal = block, causal
        self.tok = nn.Embedding(vocab_size, D_MODEL)
        self.pos = nn.Embedding(block, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            D_MODEL, N_HEADS, dim_feedforward=4 * D_MODEL,
            batch_first=True, dropout=0.0, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, N_LAYERS)
        self.ln = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size)

    def hidden(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T))
        mask = None
        if self.causal:                       # 미래를 못 보게 하는 인과 마스크
            mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        return self.ln(self.blocks(x, mask=mask))

    def forward(self, idx):
        return self.head(self.hidden(idx))


def train_model(model, make_batch, steps: int, label: str) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(1, steps + 1):
        x, y = make_batch()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               y.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
        if step in (1, steps):
            print(f"    {label} step {step:3d}: loss {float(loss):.3f}")


if __name__ == "__main__":
    t0 = time.time()
    torch.manual_seed(SEED)
    print("=" * 70)
    print("사전학습 패러다임 — 빈칸 채우기(BERT) vs 이어쓰기(GPT)")
    print("=" * 70 + "\n")

    # ---------------- A. tiny_corpus 를 단어 단위로 준비 -------------------
    words = hjh_data.tiny_corpus().replace(".", " .").split()[:12000]
    vocab = sorted(set(words)) + ["[MASK]"]
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}
    MASK = stoi["[MASK]"]
    ids = torch.tensor([stoi[w] for w in words])
    BLOCK = 10

    print("[1] 문제 출제기 — 라벨 없이 텍스트 자체가 문제집이 된다")
    sample = words[:5]
    print(f"    원문          : {' '.join(sample)}")
    print(f"    MLM 문제(BERT): {' '.join(['[MASK]' if i == 1 else w for i, w in enumerate(sample)])}"
          f"  -> 정답 '{sample[1]}'")
    print(f"    CLM 문제(GPT) : '{' '.join(sample[:3])}' 다음 단어는? -> 정답 '{sample[3]}'")
    print(f"    (단어 어휘 {len(vocab)}종, 학습 토큰 {len(ids):,}개)\n")

    gen = torch.Generator().manual_seed(SEED)

    def batch_starts(n=32):
        return torch.randint(0, len(ids) - BLOCK - 1, (n,), generator=gen)

    def mlm_batch():
        x = torch.stack([ids[s: s + BLOCK] for s in batch_starts()]).clone()
        y = torch.full_like(x, -100)                     # 가린 곳만 채점
        mask_pos = torch.rand(x.shape, generator=gen) < 0.15
        y[mask_pos] = x[mask_pos]
        x[mask_pos] = MASK
        return x, y

    def clm_batch():
        starts = batch_starts()
        x = torch.stack([ids[s: s + BLOCK] for s in starts])
        y = torch.stack([ids[s + 1: s + BLOCK + 1] for s in starts])
        return x, y

    print("[2] 같은 구조, 다른 과제로 두 모델 학습 (각 300스텝)")
    bert = MiniTransformer(len(vocab), BLOCK, causal=False)   # 양방향
    gpt = MiniTransformer(len(vocab), BLOCK, causal=True)     # 단방향
    train_model(bert, mlm_batch, 300, "BERT식(빈칸)")
    train_model(gpt, clm_batch, 300, "GPT식(이어쓰기)")
    print()

    # ---------------- [3] 빈칸 채우기 시험 --------------------------------
    print("[3] 빈칸 채우기 시험 — 뒤 문맥을 볼 수 있는가가 갈림길")
    quiz = ["[MASK]", "학생이", "김치찌개를", "만들었다", "."]
    x = torch.tensor([[stoi[w] for w in quiz]])
    with torch.no_grad():
        bert_top = torch.topk(F.softmax(bert(x)[0, 0], -1), 3)
        gpt_top = torch.topk(F.softmax(gpt(x)[0, 0], -1), 3)   # 왼쪽 문맥이 없음!
    fmt = lambda t: ", ".join(f"'{itos[int(i)]}'({p:.0%})"
                              for p, i in zip(t.values, t.indices))
    print(f"    문제: {' '.join(quiz)}  (문장 첫 단어를 가림)")
    print(f"    BERT식 답: {fmt(bert_top)}")
    print(f"    GPT식 답 : {fmt(gpt_top)}")
    print("    -> BERT식은 뒤 문맥('학생이...')을 보고 시간 표현 자리임을 알지만,")
    print("       GPT식은 빈칸 왼쪽에 아무것도 없어 사실상 찍을 수밖에 없습니다.\n")

    # ---------------- [4] 이어쓰기 시험 -----------------------------------
    print("[4] 이어쓰기 시험 — 생성은 이어쓰기를 배운 쪽의 본업")
    def continue_with(model, prompt_words, n=8):
        seq = [stoi[w] for w in prompt_words]
        for _ in range(n):
            x = torch.tensor([seq[-BLOCK:]])
            with torch.no_grad():
                if model.causal:
                    logits = model(x)[0, -1]              # 다음 단어 예측
                else:                                     # BERT식: 끝에 [MASK] 붙여 억지 생성
                    x = torch.cat([x, torch.tensor([[MASK]])], dim=1)[:, -BLOCK:]
                    logits = model(x)[0, -1]
            probs = F.softmax(logits / 0.7, -1)
            seq.append(int(torch.multinomial(probs, 1, generator=gen)))
        return " ".join(itos[i] for i in seq)
    prompt = ["주말에", "요리사가"]
    print(f"    GPT식 : {continue_with(gpt, prompt)}")
    print(f"    BERT식: {continue_with(bert, prompt)}")
    print("    -> GPT식은 문형('목적어 동사 . 시간 ...')을 자연스럽게 잇지만,")
    print("       BERT식은 '다음에 올 말'의 확률을 배운 적이 없어 흐름이 어색합니다.\n")

    # ---------------- [5] 사전학습 -> 미세조정 맛보기 ----------------------
    print("[5] 미세조정 맛보기 — 사전학습이 '밑천'이 되는가 (리뷰 감성 분류)")
    rows = hjh_data.review_corpus(400, seed=3)
    chars = sorted({ch for r in rows for ch in r["text"]})
    cstoi = {ch: i for i, ch in enumerate(chars)}
    CBLOCK = 32
    def encode(text):
        s = [cstoi[ch] for ch in text[:CBLOCK]]
        return s + [0] * (CBLOCK - len(s))
    corpus_ids = torch.tensor([i for r in rows for i in encode(r["text"])])

    # 5-1. 라벨 없이 리뷰 텍스트로 글자 단위 CLM 사전학습
    lm = MiniTransformer(len(chars), CBLOCK, causal=True)
    def char_clm_batch():
        starts = torch.randint(0, len(corpus_ids) - CBLOCK - 1, (32,), generator=gen)
        x = torch.stack([corpus_ids[s: s + CBLOCK] for s in starts])
        y = torch.stack([corpus_ids[s + 1: s + CBLOCK + 1] for s in starts])
        return x, y
    train_model(lm, char_clm_batch, 800, "사전학습(리뷰 이어쓰기)")

    # 5-2. 소량(60건) 라벨로 미세조정: 모델의 문장 표현 + 로지스틱 헤드
    X_ids = torch.tensor([encode(r["text"]) for r in rows])
    y_all = [r["label"] for r in rows]
    def features(model):
        with torch.no_grad():
            return model.hidden(X_ids).mean(dim=1).numpy()   # 문장 표현(평균 풀링)
    scratch = MiniTransformer(len(chars), CBLOCK, causal=True)  # 사전학습 없는 대조군
    results = {}
    for name, model in (("사전학습 O", lm), ("사전학습 X(무작위 가중치)", scratch)):
        feats = features(model)
        clf = LogisticRegression(max_iter=1000, random_state=SEED)
        clf.fit(feats[:60], y_all[:60])                      # 라벨 단 60건으로 학습
        results[name] = clf.score(feats[200:], y_all[200:])  # 200건으로 평가
    for name, acc in results.items():
        print(f"    {name:22s}: 라벨 60건 학습 -> 평가 정확도 {acc:.1%}")
    print("    -> 같은 구조라도 '언어를 먼저 배운' 쪽이 소량 라벨로 훨씬 잘 배웁니다.")
    print("       이것이 사전학습 -> 미세조정 패러다임이 세상을 바꾼 이유입니다.")
    print(f"\n총 실행 시간: {time.time() - t0:.1f}초 (90초 제한 안)")
