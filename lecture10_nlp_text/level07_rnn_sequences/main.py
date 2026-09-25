"""
level07 — RNN 과 시퀀스 모델

tiny_corpus 일부로 '글자 단위 다음 글자 예측' 미니 언어모델을
torch RNN 으로 학습합니다. 학습 전/중/후의 생성문 변화, 다음 글자
확률, 온도(temperature)에 따른 생성 차이까지 시연합니다.
CPU 에서 몇 초 만에 끝나도록 아주 작게 설계했습니다.
"""

import sys
import time
import pathlib

import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

SEED = 42
SEQ_LEN = 40          # 한 번에 읽는 글자 수
HIDDEN = 64           # 릴레이 메모(은닉 상태)의 크기
EMBED = 32
EPOCHS = 12
BATCH = 64


class CharRNN(nn.Module):
    """임베딩 -> RNN(릴레이 메모) -> 다음 글자 점수."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, EMBED)
        self.rnn = nn.RNN(EMBED, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, vocab_size)

    def forward(self, x, h=None):
        emb = self.embed(x)                 # (B, T, EMBED)
        out, h = self.rnn(emb, h)           # out: 모든 시점의 메모 (B, T, HIDDEN)
        return self.head(out), h            # 시점마다 다음 글자 점수


def make_dataset(text: str, stoi: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """입력 = 글자 시퀀스, 정답 = 그것을 한 글자 민 시퀀스."""
    ids = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    n_chunks = (len(ids) - 1) // SEQ_LEN
    x = ids[: n_chunks * SEQ_LEN].view(n_chunks, SEQ_LEN)
    y = ids[1: n_chunks * SEQ_LEN + 1].view(n_chunks, SEQ_LEN)
    return x, y


@torch.no_grad()
def generate(model, stoi, itos, prompt: str, length: int = 60,
             temperature: float = 0.8) -> str:
    """프롬프트 뒤에 length 글자를 이어 씁니다."""
    model.eval()
    ids = [stoi.get(ch, 0) for ch in prompt]
    x = torch.tensor([ids], dtype=torch.long)
    logits, h = model(x)                              # 프롬프트 전체를 읽어 메모 구성
    out = list(prompt)
    last = logits[0, -1]
    for _ in range(length):
        probs = torch.softmax(last / temperature, dim=-1)
        idx = int(torch.multinomial(probs, 1))
        out.append(itos[idx])
        logits, h = model(torch.tensor([[idx]]), h)   # 메모를 이어받아 한 글자씩
        last = logits[0, -1]
    model.train()
    return "".join(out)


@torch.no_grad()
def next_char_topk(model, stoi, itos, prompt: str, k: int = 3):
    model.eval()
    x = torch.tensor([[stoi.get(ch, 0) for ch in prompt]], dtype=torch.long)
    logits, _ = model(x)
    probs = torch.softmax(logits[0, -1], dim=-1)
    top = torch.topk(probs, k)
    model.train()
    return [(itos[int(i)], float(p)) for p, i in zip(top.values, top.indices)]


if __name__ == "__main__":
    t0 = time.time()
    torch.manual_seed(SEED)
    print("=" * 70)
    print("RNN 언어모델 — 릴레이 메모로 다음 글자 맞히기")
    print("=" * 70 + "\n")

    # [1] 데이터 준비 ------------------------------------------------------
    text = hjh_data.tiny_corpus()[:20000]
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    x, y = make_dataset(text, stoi)
    print(f"[1] 데이터: {len(text):,}자, 글자 어휘 {len(chars)}종, "
          f"학습 시퀀스 {x.shape[0]}개 (길이 {SEQ_LEN})")
    sample = text[:20]
    print(f"    입력 예: {sample!r}")
    print(f"    정답 예: {text[1:21]!r}  <- 입력을 한 글자 민 것\n")

    # [2] 모델 조립 --------------------------------------------------------
    model = CharRNN(len(chars))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[2] 모델: 임베딩({EMBED}) -> RNN(메모 {HIDDEN}칸) -> 출력층")
    print(f"    파라미터 {n_params:,}개 (요즘 LLM 의 수십억 분의 일)\n")

    # [3] 학습 — 생성문이 말이 되어 가는 과정 -------------------------------
    print("[3] 학습: loss 하강과 생성문의 변화")
    prompt = "어제 학생이 "
    print(f"    학습 전 생성: {generate(model, stoi, itos, prompt)!r}")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(x.shape[0])
        total = 0.0
        for i in range(0, x.shape[0], BATCH):
            idx = perm[i:i + BATCH]
            logits, _ = model(x[idx])                 # 문장마다 메모는 0에서 시작
            loss = loss_fn(logits.reshape(-1, len(chars)), y[idx].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(idx)
        avg = total / x.shape[0]
        if epoch in (1, EPOCHS // 2, EPOCHS):
            print(f"    epoch {epoch}: loss {avg:.3f} | 생성: "
                  f"{generate(model, stoi, itos, prompt, 40)!r}")
    print("    -> loss 가 내려갈수록 '주어+목적어+동사.' 골격이 잡혀 갑니다.\n")

    # [4] 다음 글자 예측 ---------------------------------------------------
    print("[4] 다음 글자 예측 확률 top3")
    for p in ["학생이 보고서를 만들", "오늘 회사원이 ", "김치찌"]:
        top = ", ".join(f"'{ch}'({prob:.0%})" for ch, prob in
                        next_char_topk(model, stoi, itos, p))
        print(f"    {p!r} 다음: {top}")
    print()

    # [5] 온도 실험 --------------------------------------------------------
    print("[5] 온도(temperature) 다이얼 — LLM API 의 그 파라미터")
    for temp in (0.3, 1.5):
        text_out = generate(model, stoi, itos, "주말에 ", 50, temperature=temp)
        print(f"    온도 {temp}: {text_out!r}")
    print("    -> 낮으면 안전하고 단조롭게, 높으면 다양하지만 엉뚱하게.")
    print("\n    한계 메모: RNN 은 릴레이 메모 하나에 과거를 요약하므로 문장이")
    print("    길어지면 앞 내용이 흐려집니다(장기 의존성). 해결책은 다음 레벨의 어텐션.")
    print(f"\n총 실행 시간: {time.time() - t0:.1f}초")
