"""
从零实现 BPE (Byte Pair Encoding) 分词器。
1) 在 tiny_corpus 上反复合并"最常相邻出现的字符对"来构建 vocab；
2) 比较不同合并次数 (vocab 大小) 下同一句话会被切成多少个 token；
3) 观察有无数字分离 (digit split) 时数字的切分方式有何不同，
   以及这如何影响模型的数学能力。
"""

import sys
import pathlib
import random
from collections import Counter

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

END = "</w>"  # 词尾标记 — 用来区分"报告"出现在词中还是词尾。


def word_freqs(text: str) -> Counter:
    """按空格统计词频，得到 {字符元组: 频次} 字典。"""
    freqs = Counter()
    for word in text.split():
        freqs[tuple(list(word) + [END])] += 1
    return freqs


def count_pairs(freqs: Counter, digit_split: bool) -> Counter:
    """统计相邻符号对的出现频次。digit_split=True 时，含数字的符号
    不参与合并，从而让数字始终保持一位一位的状态。"""
    pairs = Counter()
    for symbols, f in freqs.items():
        for a, b in zip(symbols, symbols[1:]):
            if digit_split and (any(c.isdigit() for c in a) or
                                any(c.isdigit() for c in b)):
                continue
            pairs[(a, b)] += f
    return pairs


def merge_pair(freqs: Counter, pair) -> Counter:
    """在所有词里把该字符对粘成一个符号。"""
    a, b = pair
    new_freqs = Counter()
    for symbols, f in freqs.items():
        merged, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        new_freqs[tuple(merged)] += f
    return new_freqs


def train_bpe(text: str, num_merges: int, digit_split: bool = False,
              verbose: bool = False):
    """BPE 训练: 把最高频的字符对合并 num_merges 次，返回合并规则列表。"""
    freqs = word_freqs(text)
    merges = []
    for step in range(num_merges):
        pairs = count_pairs(freqs, digit_split)
        if not pairs:            # 没有可合并的对就提前结束
            break
        best, best_n = pairs.most_common(1)[0]
        freqs = merge_pair(freqs, best)
        merges.append(best)
        if verbose and (step < 8 or (step + 1) % 20 == 0):
            print(f"    合并 {step + 1:3d}: '{best[0]}' + '{best[1]}'"
                  f" -> '{best[0] + best[1]}'  (出现 {best_n} 次)")
    return merges


def encode(text: str, merges) -> list:
    """按训练时学到的顺序应用合并规则，对新句子做分词。"""
    tokens = []
    for word in text.split():
        symbols = list(word) + [END]
        for pair in merges:  # 按学到的顺序应用 (即优先级)
            symbols = list(next(iter(
                merge_pair(Counter({tuple(symbols): 1}), pair))))
        tokens.extend(symbols)
    return [t.replace(END, "") for t in tokens if t != END]


def make_price_corpus(n: int = 300, seed: int = 42) -> str:
    """数字分离实验用的销售汇报语料 (金额多以 0 结尾)。"""
    rng = random.Random(seed)
    lines = [f"销售额 {rng.randrange(1, 999) * 100}韩元 已记录"
             for _ in range(n)]
    return " ".join(lines)


if __name__ == "__main__":
    corpus = hjh_data.tiny_corpus()
    print(f"[1] 加载语料: {len(corpus):,} 字，不同字符 "
          f"{len(set(corpus)):,} 种 (从单字出发)")

    # ---- 观察 BPE 合并过程 ----
    print("\n[2] BPE 训练 — 从最常相邻出现的字符对开始依次合并。")
    merges = train_bpe(corpus, num_merges=60, verbose=True)
    print(f"    共完成 {len(merges)} 次合并 (请求 60 次，字符对用尽则提前结束)")

    # ---- vocab 大小的取舍 ----
    sample = "昨天 学生 把报告 做好了。"
    print(f"\n[3] vocab 大小的取舍 — 同一句话会变成几个 token")
    print(f"    句子: \"{sample}\"")
    base_vocab = len(set(corpus)) + 1  # 字符种数 + </w>
    for n in [0, 10, 30, 60]:
        m = train_bpe(corpus, num_merges=n)
        toks = encode(sample, m)
        print(f"    合并 {n:3d}次 (vocab~{base_vocab + len(m):3d}):"
              f" {len(toks):2d}个token -> {toks}")
    print("    -> vocab 越大句子越短(推理成本下降)，")
    print("       但嵌入表随之膨胀，稀有 token 得到的训练机会也变少。")

    # ---- 数字分离 (digit split) 实验 ----
    print("\n[4] 数字分离实验 — 用金额密集的语料把 BPE 训练两遍")
    price_corpus = make_price_corpus()
    m_free = train_bpe(price_corpus, num_merges=40, digit_split=False)
    m_split = train_bpe(price_corpus, num_merges=40, digit_split=True)
    digit_merges = [a + b for a, b in m_free
                    if any(c.isdigit() for c in a + b)]
    print(f"    不分离: 产生了 {len(digit_merges)} 个含数字的合并"
          f" 例: {digit_merges[:6]}")

    for test in ["98700韩元", "12500韩元"]:
        t_free = encode(test, m_free)
        t_split = encode(test, m_split)
        print(f"    \"{test}\"  不分离: {t_free}")
        print(f"    {'':>10}  分离后: {t_split}")

    print("\n[5] 这为什么对数学很重要")
    print("    不分离: 会出现 '00'、'500' 这样的整块 token，987+13 这类")
    print("      进位计算就无法按'数位'来看待。")
    print("    分离后: 所有数字都一致地切成一位一位，模型才有机会")
    print("      学到数位规则(进位等)。")
    print("    这正是当今主流 LLM 强制把数字按 1~3 位切分的原因。")
