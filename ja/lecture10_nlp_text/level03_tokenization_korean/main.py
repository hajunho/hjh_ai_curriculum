"""
level03 — トークン化と日本語の特殊性

同じ文を 空白 / 文字 / n-gram / ミニ辞書ベース形態素解析 の
4 つの戦略でトークン化して比較し、語彙サイズの実験と
ミニ BPE (Byte Pair Encoding) の学習まで自分の手で実装します。
NLP 専用パッケージなし、標準ライブラリだけで動作します。
"""

import re
import sys
import pathlib
from collections import Counter

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

# ---------------------------------------------------------------------------
# トークン化戦略 4 種
# ---------------------------------------------------------------------------

def tokenize_space(text: str) -> list[str]:
    """戦略1: 空白分割 — 英語の基本。分かち書きのない日本語では文がほぼ丸ごと残ります。"""
    return text.split()


def tokenize_syllable(text: str) -> list[str]:
    """戦略2: 文字分割 — 未登録語に強いものの、1 文字では意味が薄まります。"""
    return [ch for ch in text if not ch.isspace()]


def tokenize_ngram(text: str, n: int = 2) -> list[str]:
    """戦略3: 文字 n-gram — 連続する n 文字を重ねながら束ねます (日本語検索の定番技法)。"""
    grams = []
    for word in text.split():
        if len(word) < n:
            grams.append(word)
        else:
            grams.extend(word[i:i + n] for i in range(len(word) - n + 1))
    return grams


# --- 戦略4: ミニ辞書ベース形態素解析器 ---------------------------------------
# 実際の形態素解析器 (MeCab など) の骨格のうち「辞書 + 最長一致」を縮小実装した教育用です。
# 本物はコスト最小化 (ラティス探索) を使いますが、原理の入口はこれで十分つかめます。

NOUNS = ["リピート購入", "問い合わせ", "期待以上", "説明書", "一週間", "雰囲気",
         "配送", "梱包", "値段", "品質", "店員", "店内", "写真", "性能",
         "設置", "説明", "状態", "返事", "二度", "色", "気", "星", "数", "点"]
STEMS = ["素晴らしい", "がっかりし", "分かりにくく", "うるさかった", "壊れ", "破れ",
         "速く", "弱く", "狭く", "違っ", "違い", "良い", "悪", "丈夫", "丁寧",
         "親切", "快適", "簡単", "楽", "満足し", "助かり", "驚き", "届き",
         "買わ", "思っ", "思い", "迷い", "かかり", "つけ", "利用し"]
JOSA = ["から", "まで", "より", "など", "のに", "が", "の", "に", "を",
        "と", "は", "も", "で", "や"]  # 助詞
EOMI = ["ませんでした", "ました", "ています", "ません", "でした", "ないと",
        "ます", "です", "ない", "たい", "て", "た"]  # 語尾・助動詞

# 最長一致用に (表層形, タグ) を長い順に並べた統合辞書
_DICT = sorted(
    [(w, "") for w in NOUNS]
    + [(w, "-") for w in STEMS]
    + [(w, "(助詞)") for w in JOSA]
    + [(w, "(語尾)") for w in EOMI],
    key=lambda e: len(e[0]), reverse=True,
)


def analyze_word(word: str) -> list[str]:
    """連続した文字列を最長一致法で [名詞 / 語幹- / 助詞 / 語尾] に解体します。"""
    tokens = []
    i = 0
    while i < len(word):
        for entry, tag in _DICT:
            if word.startswith(entry, i):
                tokens.append(entry + ("-" if tag == "-" else "") + (tag if tag not in ("", "-") else ""))
                i += len(entry)
                break
        else:
            tokens.append(word[i])          # 未登録の文字はそのまま 1 文字
            i += 1
    return tokens


def tokenize_morph(text: str) -> list[str]:
    """戦略4: ミニ形態素解析 — 空白で切れた塊ごとに analyze_word を適用。"""
    tokens = []
    for word in text.split():
        tokens.extend(analyze_word(word))
    return tokens


# ---------------------------------------------------------------------------
# ミニ BPE — よくくっつく文字ペアを繰り返し併合
# ---------------------------------------------------------------------------

def bpe_train(corpus_words: list[str], n_merges: int) -> list[tuple[str, str]]:
    """コーパスから併合ルールを学習します。戻り値: [(左の断片, 右の断片), ...]"""
    # 各単語を文字リストとして初期化
    words = [list(w) for w in corpus_words]
    merges = []
    for _ in range(n_merges):
        pair_count = Counter()
        for w in words:
            for a, b in zip(w, w[1:]):
                pair_count[(a, b)] += 1
        if not pair_count:
            break
        (a, b), freq = pair_count.most_common(1)[0]
        if freq < 2:                       # 2 回未満なら併合する価値なし
            break
        merges.append((a, b))
        merged = a + b
        for w in words:                    # すべての単語に併合を適用
            i = 0
            while i < len(w) - 1:
                if w[i] == a and w[i + 1] == b:
                    w[i:i + 2] = [merged]
                else:
                    i += 1
    return merges


def bpe_encode(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """学習した併合ルールを順番に適用して単語をトークン化します。"""
    pieces = list(word)
    for a, b in merges:
        i = 0
        while i < len(pieces) - 1:
            if pieces[i] == a and pieces[i + 1] == b:
                pieces[i:i + 2] = [a + b]
            else:
                i += 1
    return pieces


# ---------------------------------------------------------------------------
# 実演
# ---------------------------------------------------------------------------

def demo_compare(sentence: str) -> None:
    print("[1] 同じ文、4 つのトークン化戦略")
    print(f"    文: {sentence!r}\n")
    strategies = [
        ("空白分割", tokenize_space(sentence)),
        ("文字分割", tokenize_syllable(sentence)),
        ("2-gram", tokenize_ngram(sentence, 2)),
        ("ミニ形態素", tokenize_morph(sentence)),
    ]
    for name, tokens in strategies:
        print(f"    {name:6s} ({len(tokens):2d}個): {tokens}")
    print("    -> 空白分割は日本語では文がほぼ丸ごと 1 トークンになってしまいます。\n")


def demo_morph_inside() -> None:
    print("[2] ミニ形態素解析器の内部 — 連続した文字列を「最長一致」で解体")
    for word in ["配送が", "梱包は", "速くて", "助かりました", "満足しています", "新製品が"]:
        print(f"    {word:10s} -> {analyze_word(word)}")
    print("    -> 「新製品」のように辞書にない言葉は 1 文字ずつバラバラになります (未登録語問題)。\n")


def demo_vocab_size() -> None:
    print("[3] 語彙サイズの実験 — レビュー 60 件を戦略別にトークン化すると")
    reviews = [r["text"] for r in hjh_data.review_corpus(60, seed=3)]
    strategies = {
        "空白分割": tokenize_space,
        "文字分割": tokenize_syllable,
        "2-gram": tokenize_ngram,
        "ミニ形態素": tokenize_morph,
    }
    for name, fn in strategies.items():
        vocab = Counter()
        for r in reviews:
            vocab.update(fn(r))
        delivery = sorted(t for t in vocab if t.startswith("配送"))[:4]
        print(f"    {name:6s}: 語彙 {len(vocab):4d}種 | 「配送」系トークン: {delivery}")
    print("    -> 空白分割では「配送…ました」の文全体が 1 語彙になり、同じ概念がまとまりません。")
    print("       形態素方式は「配送」+助詞に分けて語彙を圧縮します。\n")


def demo_bpe() -> None:
    print("[4] ミニ BPE 学習 — よくくっつく文字ペアを併合してトークンを「発明」する")
    reviews = [r["text"] for r in hjh_data.review_corpus(200, seed=3)]
    corpus_words = [w for r in reviews for w in r.split()]
    merges = bpe_train(corpus_words, n_merges=60)
    print(f"    併合ルール {len(merges)}個を学習。最初の 10 個:")
    for i, (a, b) in enumerate(merges[:10], start=1):
        print(f"      {i:2d}. '{a}' + '{b}' -> '{a + b}'")
    print("\n    学習したルールで単語をトークン化:")
    for word in ["配送が", "品質が", "満足しています", "超高速配送"]:
        note = "  <- 初めて見る単語も断片で表現!" if word == "超高速配送" else ""
        print(f"      {word:10s} -> {bpe_encode(word, merges)}{note}")
    print("    -> よく出てきた「配送が」は少数のトークンに、初めて見る「超高速配送」は")
    print("       学習済みの断片 (「配送」を含む) の組み合わせで表現されます。")
    print("       GPT 系トークナイザー (BPE) の原理です。日本語の学習データが少ないと")
    print("       併合ルールが育たず、トークン数 (= API 料金) が増えてしまいます。")


if __name__ == "__main__":
    print("=" * 70)
    print("トークン化と日本語の特殊性 — 切る単位が品質を決める")
    print("=" * 70 + "\n")
    demo_compare("配送が速くて助かりました")
    demo_morph_inside()
    demo_vocab_size()
    demo_bpe()
