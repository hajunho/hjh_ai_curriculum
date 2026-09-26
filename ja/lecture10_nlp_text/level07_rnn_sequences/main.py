"""
level07 — RNN とシーケンスモデル

tiny_corpus の一部で「文字単位の次の文字予測」ミニ言語モデルを
torch の RNN で学習します。学習前/途中/後の生成文の変化、次の文字の
確率、温度 (temperature) による生成の違いまで実演します。
CPU で数秒で終わるよう、ごく小さく設計してあります。
"""

import sys
import time
import pathlib

import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

SEED = 42
SEQ_LEN = 40          # 一度に読む文字数
HIDDEN = 64           # リレーメモ (隠れ状態) の大きさ
EMBED = 32
EPOCHS = 12
BATCH = 64


class CharRNN(nn.Module):
    """埋め込み -> RNN (リレーメモ) -> 次の文字のスコア。"""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, EMBED)
        self.rnn = nn.RNN(EMBED, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, vocab_size)

    def forward(self, x, h=None):
        emb = self.embed(x)                 # (B, T, EMBED)
        out, h = self.rnn(emb, h)           # out: 全時点のメモ (B, T, HIDDEN)
        return self.head(out), h            # 時点ごとに次の文字のスコア


def make_dataset(text: str, stoi: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """入力 = 文字シーケンス、正解 = それを 1 文字ずらしたシーケンス。"""
    ids = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    n_chunks = (len(ids) - 1) // SEQ_LEN
    x = ids[: n_chunks * SEQ_LEN].view(n_chunks, SEQ_LEN)
    y = ids[1: n_chunks * SEQ_LEN + 1].view(n_chunks, SEQ_LEN)
    return x, y


@torch.no_grad()
def generate(model, stoi, itos, prompt: str, length: int = 60,
             temperature: float = 0.8) -> str:
    """プロンプトの後ろに length 文字を書き継ぎます。"""
    model.eval()
    ids = [stoi.get(ch, 0) for ch in prompt]
    x = torch.tensor([ids], dtype=torch.long)
    logits, h = model(x)                              # プロンプト全体を読んでメモを構成
    out = list(prompt)
    last = logits[0, -1]
    for _ in range(length):
        probs = torch.softmax(last / temperature, dim=-1)
        idx = int(torch.multinomial(probs, 1))
        out.append(itos[idx])
        logits, h = model(torch.tensor([[idx]]), h)   # メモを受け継いで 1 文字ずつ
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
    print("RNN 言語モデル — リレーメモで次の文字を当てる")
    print("=" * 70 + "\n")

    # [1] データ準備 -------------------------------------------------------
    text = hjh_data.tiny_corpus()[:20000]
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    x, y = make_dataset(text, stoi)
    print(f"[1] データ: {len(text):,}文字、文字語彙 {len(chars)}種、"
          f"学習シーケンス {x.shape[0]}個 (長さ {SEQ_LEN})")
    sample = text[:20]
    print(f"    入力の例: {sample!r}")
    print(f"    正解の例: {text[1:21]!r}  <- 入力を 1 文字ずらしたもの\n")

    # [2] モデルの組み立て ---------------------------------------------------
    model = CharRNN(len(chars))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[2] モデル: 埋め込み({EMBED}) -> RNN (メモ {HIDDEN}マス) -> 出力層")
    print(f"    パラメータ {n_params:,}個 (最近の LLM の数十億分の一)\n")

    # [3] 学習 — 生成文が言葉になっていく過程 ---------------------------------
    print("[3] 学習: loss の下降と生成文の変化")
    prompt = "昨日 学生が "
    print(f"    学習前の生成: {generate(model, stoi, itos, prompt)!r}")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(x.shape[0])
        total = 0.0
        for i in range(0, x.shape[0], BATCH):
            idx = perm[i:i + BATCH]
            logits, _ = model(x[idx])                 # 文ごとにメモは 0 から開始
            loss = loss_fn(logits.reshape(-1, len(chars)), y[idx].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(idx)
        avg = total / x.shape[0]
        if epoch in (1, EPOCHS // 2, EPOCHS):
            print(f"    epoch {epoch}: loss {avg:.3f} | 生成: "
                  f"{generate(model, stoi, itos, prompt, 40)!r}")
    print("    -> loss が下がるほど「主語+目的語+動詞。」の骨格が整っていきます。\n")

    # [4] 次の文字の予測 ----------------------------------------------------
    print("[4] 次の文字の予測確率 top3")
    for p in ["学生が 報告書を 作", "今日 会社員が ", "カレーライ"]:
        top = ", ".join(f"'{ch}'({prob:.0%})" for ch, prob in
                        next_char_topk(model, stoi, itos, p))
        print(f"    {p!r} の次: {top}")
    print()

    # [5] 温度の実験 --------------------------------------------------------
    print("[5] 温度 (temperature) のダイヤル — LLM API のあのパラメータ")
    for temp in (0.3, 1.5):
        text_out = generate(model, stoi, itos, "週末に ", 50, temperature=temp)
        print(f"    温度 {temp}: {text_out!r}")
    print("    -> 低いと安全で単調に、高いと多彩だが突拍子もなく。")
    print("\n    限界メモ: RNN はリレーメモ 1 枚に過去を要約するので、文が")
    print("    長くなると前の内容がぼやけます (長期依存性)。解決策は次のレベルのアテンション。")
    print(f"\n総実行時間: {time.time() - t0:.1f}秒")
