"""
level10 — ミニトランスフォーマーを自作する

torch で小型 GPT (デコーダ専用トランスフォーマー) を作り、
tiny_corpus 3 万文字で「次の文字予測」を学習させます。
loss の下降と生成文の変化、パープレキシティ、アテンションの因果構造、
loss 曲線の PNG 保存まで実演します。CPU で 60 秒以内に終わります。
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
BLOCK = 32          # コンテキスト長 (モデルが一度に見る文字数)
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
BATCH = 48
STEPS = 400
LR = 3e-3


class CausalSelfAttention(nn.Module):
    """因果マスク付きマルチヘッドセルフアテンション (level09 の部品の torch 版)。"""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        mask = torch.triu(torch.ones(BLOCK, BLOCK), diagonal=1).bool()
        self.register_buffer("mask", mask)
        self.last_weights = None                     # 実演用: アテンションを保存

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(D_MODEL, dim=2)
        def heads(t):                                # (B, T, C) -> (B, H, T, C/H)
            return t.view(B, T, N_HEADS, C // N_HEADS).transpose(1, 2)
        q, k, v = heads(q), heads(k), heads(v)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // N_HEADS)
        att = att.masked_fill(self.mask[:T, :T], float("-inf"))   # 未来のカンニング防止
        att = F.softmax(att, dim=-1)
        self.last_weights = att.detach()
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    """アテンション -> 残差+ノルム -> FF -> 残差+ノルム (Pre-LN 方式)。"""

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
        x = self.tok_embed(idx) + self.pos_embed(pos)   # トークン + 座席番号札
        x = self.blocks(x)
        return self.head(self.ln_final(x))              # (B, T, vocab)


def get_batch(ids: torch.Tensor, generator: torch.Generator):
    """コーパスの任意の位置 BATCH 個を切り出して (入力, 1 文字ずらした正解) のペアに。"""
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
        logits = model(ids[:, -BLOCK:])                 # 直近 BLOCK 文字だけを見る
        probs = F.softmax(logits[0, -1] / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=None)
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    model.train()
    return "".join(itos[int(i)] for i in ids[0])


if __name__ == "__main__":
    t0 = time.time()
    torch.manual_seed(SEED)
    print("=" * 70)
    print("ミニトランスフォーマー (GPT 構造) — 組み立て、始動、路上走行")
    print("=" * 70 + "\n")

    # [1] データ ------------------------------------------------------------
    text = hjh_data.tiny_corpus()[:30000]
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    ids = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    print(f"[1] データ: {len(text):,}文字、文字語彙 {len(chars)}種")
    print(f"    学習問題: 長さ {BLOCK} のシーケンスで、位置ごとに「次の文字」を同時に予測")
    print(f"    (1 シーケンスが {BLOCK}個の問題 — トランスフォーマー並列学習の力)\n")

    # [2] モデル ------------------------------------------------------------
    model = MiniGPT(len(chars))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[2] モデル: {D_MODEL}次元、{N_LAYERS}層、{N_HEADS}ヘッドのデコーダ")
    print(f"    パラメータ {n_params:,}個 — 構造は GPT、大きさはおもちゃ\n")

    # [3] 学習 --------------------------------------------------------------
    print(f"[3] 学習: {STEPS} ステップ (loss と生成文の変化を観察)")
    prompt = "昨日 "
    print(f"    step   0 | loss  -   | 生成: {generate(model, stoi, itos, prompt, 40)!r}")
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
            print(f"    step {step:3d} | loss {losses[-1]:.3f} (パープレキシティ {ppl:5.1f}) "
                  f"| 生成: {generate(model, stoi, itos, prompt, 40)!r}")
    print(f"    -> 初期 loss ~ln({len(chars)})={math.log(len(chars)):.2f} (当てずっぽうの水準) から")
    print(f"       下がるほど、モデルが次の文字に「驚かなく」なっていきます。\n")

    # loss 曲線の保存
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
    print(f"    loss 曲線を保存: {path}\n")

    # [4] 生成テスト ---------------------------------------------------------
    print("[4] 生成テスト — プロンプトを与えて書き継ぐ")
    for p in ["昨日 学生が ", "週末に 料理人が ", "朝に "]:
        print(f"    {p!r} -> {generate(model, stoi, itos, p, 50)!r}")
    print()

    # [5] アテンションをのぞく -------------------------------------------------
    print("[5] 学習済みアテンションの因果構造の確認 (1 層目ヘッド0、先頭 5x5)")
    _ = model(ids[:BLOCK].unsqueeze(0))
    att = model.blocks[0].attn.last_weights[0, 0, :5, :5]
    for row in att:
        print("    " + " ".join(f"{v:.2f}" for v in row))
    print("    -> 右上 (未来) が全部 0 — 因果マスクが守られています。")
    print(f"\n総実行時間: {time.time() - t0:.1f}秒 (90 秒制限内)")
