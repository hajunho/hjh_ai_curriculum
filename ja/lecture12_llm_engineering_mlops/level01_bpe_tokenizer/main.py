"""
BPE(Byte Pair Encoding) トークナイザーをゼロから実装します。
1) tiny_corpus から「最も頻繁にくっついて現れる文字ペア」を繰り返し統合して vocab を作り
2) 統合回数(vocab サイズ)によって文が何トークンになるかを比べ
3) 数字分割(digit split)がある場合とない場合で数字のトークン化がどう変わり
   数学の能力に影響するのかを確認します。
"""

import sys
import pathlib
import random
from collections import Counter

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

END = "</w>"  # 単語末の印 — 「学生」と「学生が」を区別できるようにします。


def word_freqs(text: str) -> Counter:
    """空白を区切りに単語を数えて {文字タプル: 頻度} の辞書を作ります。"""
    freqs = Counter()
    for word in text.split():
        freqs[tuple(list(word) + [END])] += 1
    return freqs


def count_pairs(freqs: Counter, digit_split: bool) -> Counter:
    """隣り合うシンボルペアの出現頻度を数えます。digit_split=True なら数字を
    含むシンボルを統合候補から外し、数字を常に1桁ずつ保ちます。"""
    pairs = Counter()
    for symbols, f in freqs.items():
        for a, b in zip(symbols, symbols[1:]):
            if digit_split and (any(c.isdigit() for c in a) or
                                any(c.isdigit() for c in b)):
                continue
            pairs[(a, b)] += f
    return pairs


def merge_pair(freqs: Counter, pair) -> Counter:
    """すべての単語の中で、その文字ペアを1つのシンボルにくっつけます。"""
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
    """BPE 学習: 最頻のペアを num_merges 回統合し、統合ルールを返します。"""
    freqs = word_freqs(text)
    merges = []
    for step in range(num_merges):
        pairs = count_pairs(freqs, digit_split)
        if not pairs:            # 統合できるペアがなくなれば早期終了
            break
        best, best_n = pairs.most_common(1)[0]
        freqs = merge_pair(freqs, best)
        merges.append(best)
        if verbose and (step < 8 or (step + 1) % 20 == 0):
            print(f"    統合 {step + 1:3d}: '{best[0]}' + '{best[1]}'"
                  f" -> '{best[0] + best[1]}'  ({best_n}回出現)")
    return merges


def encode(text: str, merges) -> list:
    """学習した統合ルールを順番に適用して、新しい文をトークン化します。"""
    tokens = []
    for word in text.split():
        symbols = list(word) + [END]
        for pair in merges:  # 学習したときの順番で適用 (優先順位)
            symbols = list(next(iter(
                merge_pair(Counter({tuple(symbols): 1}), pair))))
        tokens.extend(symbols)
    return [t.replace(END, "") for t in tokens if t != END]


def make_price_corpus(n: int = 300, seed: int = 42) -> str:
    """数字分割の実験用の売上報告文コーパス (末尾に0が多い金額)。"""
    rng = random.Random(seed)
    lines = [f"売上 {rng.randrange(1, 999) * 100}ウォン 記録"
             for _ in range(n)]
    return " ".join(lines)


if __name__ == "__main__":
    corpus = hjh_data.tiny_corpus()
    print(f"[1] コーパス読み込み: {len(corpus):,}文字、固有の文字 "
          f"{len(set(corpus)):,}種 (文字単位から出発)")

    # ---- BPE の統合過程を観察 ----
    print("\n[2] BPE 学習 — 最も頻繁にくっつく文字ペアから順に統合します。")
    merges = train_bpe(corpus, num_merges=60, verbose=True)
    print(f"    合計 {len(merges)}回の統合が完了 (要求60回、ペアが尽きれば早期終了)")

    # ---- vocab サイズのトレードオフ ----
    sample = "昨日 学生が 報告書を 作った。"
    print(f"\n[3] vocab サイズのトレードオフ — 同じ文が何トークンになるか")
    print(f"    文: \"{sample}\"")
    base_vocab = len(set(corpus)) + 1  # 文字の種類数 + </w>
    for n in [0, 10, 30, 60]:
        m = train_bpe(corpus, num_merges=n)
        toks = encode(sample, m)
        print(f"    統合 {n:3d}回 (vocab~{base_vocab + len(m):3d}):"
              f" {len(toks):2d}トークン -> {toks}")
    print("    -> vocab が大きいほど文は短くなりますが(推論コストの削減)、")
    print("       埋め込み表が大きくなり、希少なトークンは学習機会が減ります。")

    # ---- 数字分割(digit split)の実験 ----
    print("\n[4] 数字分割の実験 — 金額が多いコーパスで BPE を2回学習")
    price_corpus = make_price_corpus()
    m_free = train_bpe(price_corpus, num_merges=40, digit_split=False)
    m_split = train_bpe(price_corpus, num_merges=40, digit_split=True)
    digit_merges = [a + b for a, b in m_free
                    if any(c.isdigit() for c in a + b)]
    print(f"    分割なし: 数字が混ざった統合が {len(digit_merges)}個できました"
          f" 例: {digit_merges[:6]}")

    for test in ["98700ウォン", "12500ウォン"]:
        t_free = encode(test, m_free)
        t_split = encode(test, m_split)
        print(f"    \"{test}\"  分割なし: {t_free}")
        print(f"    {'':>13}  分割あり: {t_split}")

    print("\n[5] なぜ数学に重要なのか")
    print("    分割なし: '00'、'500' のような塊トークンができて、987+13 のような")
    print("      繰り上がりの計算を「桁」単位で見ることができなくなります。")
    print("    分割あり: すべての数が1桁ずつ一貫して切られるので、モデルが")
    print("      桁のルール(繰り上がりなど)を学ぶ機会が生まれます。")
    print("    実際の最新 LLM が数字を1〜3桁単位で強制分割する理由です。")
