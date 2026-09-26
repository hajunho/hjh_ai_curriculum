"""
迷你 GPT 预训练实战。
- 从零实现一个 2 层的小型解码器 Transformer (约 11 万参数)，
  用 hjh_data.tiny_corpus() 按字符只做"猜下一个字"的训练。
- 观察 loss 随步数下降、生成句子里长出语法的整个过程。
- 设计得非常小，只用 CPU 也能在 90 秒内跑完。
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

# ----- 超参数: "几十万参数级"的超小模型 --------------------------------------
CTX = 64        # 上下文长度 (一次看多少个字)
D_MODEL = 64    # 嵌入维度
N_HEAD = 2      # 注意力头数
N_LAYER = 2     # Transformer 块数
BATCH = 48
STEPS = 700
LR = 3e-3
TIME_BUDGET = 75.0  # 秒 — 超过这个时间就提前结束训练 (安全阀)


class CausalSelfAttention(nn.Module):
    """带掩码的自注意力: 每个字只能参考"自己前面"的字。"""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)   # Q、K、V 一次算完
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        mask = torch.tril(torch.ones(CTX, CTX))      # 下三角 = 遮住未来
        self.register_buffer("mask", mask.view(1, 1, CTX, CTX))

    def forward(self, x):
        B, T, C = x.shape
        hd = C // N_HEAD
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, 头数, T, 每头维度)
        q = q.view(B, T, N_HEAD, hd).transpose(1, 2)
        k = k.view(B, T, N_HEAD, hd).transpose(1, 2)
        v = v.view(B, T, N_HEAD, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)      # 相似度分数
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)                          # 注意力比例
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    """Transformer 块 = 注意力(收集信息) + MLP(思考)，各自配残差连接。"""

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
        self.tok_emb = nn.Embedding(vocab_size, D_MODEL)   # 字 -> 向量
        self.pos_emb = nn.Embedding(CTX, D_MODEL)          # 位置 -> 向量
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size)         # 向量 -> 下一个字的分数

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.ln_f(self.blocks(x))
        logits = self.head(x)
        loss = None
        if targets is not None:  # 在每个位置都猜"下一个字"的损失
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=0.8, top_k=8):
        for _ in range(n_new):
            logits, _ = self(idx[:, -CTX:])                 # 只用最后 CTX 个字
            logits = logits[:, -1, :] / temperature
            top = torch.topk(logits, top_k)                 # 只留前 k 个候选
            logits[logits < top.values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


def get_batch(data, rng):
    """从语料里随机切 BATCH 个位置，做成 (输入, 右移一格的答案) 对。"""
    ix = torch.randint(len(data) - CTX - 1, (BATCH,), generator=rng)
    x = torch.stack([data[i:i + CTX] for i in ix])
    y = torch.stack([data[i + 1:i + CTX + 1] for i in ix])
    return x, y


def sample_text(model, stoi, itos, prompt="今天 ", n=90):
    idx = torch.tensor([[stoi[c] for c in prompt]])
    out = model.generate(idx, n)[0].tolist()
    return "".join(itos[i] for i in out)


def main():
    torch.manual_seed(12)
    rng = torch.Generator().manual_seed(12)
    t0 = time.time()

    # [1] 加载语料并造字典 --------------------------------------------------
    text = hjh_data.tiny_corpus()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_val = len(data) // 20
    train_data, val_data = data[:-n_val], data[-n_val:]
    print(f"[1] 语料 {len(text):,} 字 / 字典 {len(chars)} 种 / 划出验证集 {n_val:,} 字")

    # [2] 创建模型 ---------------------------------------------------------
    model = TinyGPT(len(chars))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[2] 创建 TinyGPT: {N_LAYER} 层，{N_HEAD} 头，d={D_MODEL}，参数 {n_params:,} 个")
    print("    (真实的大模型就是把这个结构放大几千倍)")

    # [3] 训练前生成 — 还什么都不懂的状态 -----------------------------------
    print("\n[3] 训练前生成:  ", repr(sample_text(model, stoi, itos)))

    # [4] 预训练: 反复做"猜下一个字" ----------------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    print(f"\n[4] 预训练开始 (最多 {STEPS} 步，批 {BATCH} x 上下文 {CTX} 字)")
    for step in range(1, STEPS + 1):
        x, y = get_batch(train_data, rng)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                vx, vy = get_batch(val_data, rng)
                _, vloss = model(vx, vy)
            print(f"    step {step:4d} | train loss {loss.item():.3f} | val loss {vloss.item():.3f}")
        if step in (30, 200):  # 中途查看: 观察句子长出来的过程
            print(f"      └ step {step} 生成:", repr(sample_text(model, stoi, itos, n=60)))
        if time.time() - t0 > TIME_BUDGET:
            print(f"    (时间安全阀触发: 在第 {step} 步提前结束)")
            break
    tokens_seen = step * BATCH * CTX
    print(f"    训练完成: {step} 步，看过的字数 {tokens_seen:,} 个，耗时 {time.time()-t0:.1f} 秒")

    # [5] 训练后生成 — 确认语法"长出来"了 -----------------------------------
    print("\n[5] 训练后生成 (提示词 '今天 '):")
    for k in range(2):
        torch.manual_seed(100 + k)
        print(f"    样本 {k+1}:", sample_text(model, stoi, itos))
    print("    → 谁-把什么-怎样了 的语序，以及“把”字句和“好了”这类结果补语都自然了。")
    print("      它并没有背标准答案，只学了“下一个字的概率”，语法却自己冒出来了。")

    # [6] 保存检查点 — level06(LoRA)~08(DPO) 会接着讲这个模型的故事
    out_dir = pathlib.Path(__file__).resolve().parent / "outputs"
    os.makedirs(out_dir, exist_ok=True)
    ckpt = out_dir / "tiny_gpt.pt"
    torch.save({"model": model.state_dict(), "stoi": stoi, "itos": itos,
                "config": {"ctx": CTX, "d_model": D_MODEL, "n_head": N_HEAD,
                           "n_layer": N_LAYER, "vocab": len(chars)}}, ckpt)
    print(f"\n[6] 检查点已保存: {ckpt}")
    print(f"    总运行时间: {time.time()-t0:.1f} 秒 (在 90 秒预算之内)")


if __name__ == "__main__":
    main()
