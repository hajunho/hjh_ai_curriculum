"""
level01 — テキスト前処理の基礎

散らかった日本語レビュー (顔文字、URL、繰り返し文字、余分な空白、全角/半角の混在) を
段階的なパイプラインでクリーニングする過程を実演します。
前処理の前後で単語頻度の集計を比較し、「なぜ前処理が必須なのか」を確認します。
"""

import random
import re
import sys
import pathlib
import unicodedata
from collections import Counter

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

# ---------------------------------------------------------------------------
# [1] 汚れたレビューを作る — 現実のテキストはきれいではありません
# ---------------------------------------------------------------------------

NOISE_PREFIX = ["★★★ ", "【写真レビュー】 ", "", "♡♡ "]
NOISE_SUFFIX = [" ｗｗｗｗｗ", " (泣)(泣)", "!!!!!", " よかったですーーーー",
                " 参考: http://blog.example.com/review123", ""]


def make_dirty_reviews(n: int = 8, seed: int = 42) -> list[str]:
    """hjh_data のレビューにわざとノイズを乗せて「現実的な」レビューを作ります。"""
    rng = random.Random(seed)
    base = hjh_data.review_corpus(n * 3, seed=3)[:n]
    dirty = []
    for row in base:
        text = (rng.choice(NOISE_PREFIX) + row["text"].replace(" ", "  ", 1)
                + rng.choice(NOISE_SUFFIX))
        if rng.random() < 0.5:
            text = "  " + text + "   "          # 前後の余分な空白
        if rng.random() < 0.4:
            text = text.replace("ました", "ました　ＴＨＡＮＫＳ")  # 全角英字・全角空白を混ぜる
        dirty.append(text)
    return dirty


# ---------------------------------------------------------------------------
# [2] パイプラインの各段階を定義 — 文字列 in、文字列 out の小さな関数たち
# ---------------------------------------------------------------------------

STOPWORDS = {"ただ", "また", "参考", "ww", "そして", "しかし"}
# ひらがなの助詞・語尾はキーワード切り出しの段階で自然に落ちます
# (完全ではありません — 本格的な形態素解析は level03)


def lowercase_and_strip_url(text: str) -> str:
    """NFKC 正規化 (全角英数字→半角、全角空白→半角) + 小文字化 + URL 除去。
    URL を先に消しておくと後の段階が楽になります。"""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"https?://\S+", " ", text)


def remove_special(text: str) -> str:
    """漢字/ひらがな/カタカナ/英数字/空白だけを残して除去 (★、♡、!、【】 など)。"""
    return re.sub(r"[^一-龯々ぁ-んァ-ヶーa-z0-9\s]", " ", text)


def collapse_repeats(text: str) -> str:
    """同じ文字の 3 回以上の繰り返しを 2 回に圧縮: ですーーーー -> ですーー、wwww -> ww"""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def normalize_space(text: str) -> str:
    """連続する空白を 1 つに、前後の空白を除去。"""
    return re.sub(r"\s+", " ", text).strip()


def drop_stopwords_and_josa(text: str) -> str:
    """ストップワード除去 + 漢字・カタカナ・英数字の連続を「キーワード」として切り出す
    簡易トークン化。日本語には空白がないので、ひらがなの助詞・活用語尾 (が、を、ました...)
    はここで自然にふるい落とされます (本格的な形態素解析は level03)。"""
    tokens = re.findall(r"[一-龯々]+|[ァ-ヶー]+|[a-z0-9]+", text)
    words = [w for w in tokens if w not in STOPWORDS]
    return " ".join(words)


PIPELINE = [
    ("NFKC正規化・URL除去", lowercase_and_strip_url),
    ("特殊文字の整理", remove_special),
    ("繰り返し文字の圧縮", collapse_repeats),
    ("空白の正規化", normalize_space),
    ("ストップワード除去・簡易キーワード切り出し", drop_stopwords_and_josa),
]


def clean(text: str) -> str:
    """パイプライン全体を順番に通します。"""
    for _, step in PIPELINE:
        text = step(text)
    return text


# ---------------------------------------------------------------------------
# 実演
# ---------------------------------------------------------------------------

def demo_step_by_step(sample: str) -> None:
    print("[2] 1 件を段階ごとに通してみる")
    print(f"    原文: {sample!r}")
    text = sample
    for i, (name, step) in enumerate(PIPELINE, start=1):
        text = step(text)
        print(f"    第{i}段階 {name:22s} -> {text!r}")
    print()


def word_freq(texts: list[str], top: int = 8) -> list[tuple[str, int]]:
    counter = Counter()
    for t in texts:
        counter.update(w for w in t.split() if len(w) >= 2)
    return counter.most_common(top)


if __name__ == "__main__":
    print("=" * 70)
    print("テキスト前処理 — 散らかったレビューをパイプラインでクリーニングする")
    print("=" * 70 + "\n")

    dirty = make_dirty_reviews(n=8, seed=42)
    print(f"[1] 汚れたレビュー {len(dirty)}件を生成 (記号・URL・繰り返し文字・全角/半角の混在を含む)")
    for d in dirty[:3]:
        print(f"    例: {d!r}")
    print()

    demo_step_by_step(dirty[0])

    print("[3] 全件を一括クリーニング (原文 -> 結果)")
    cleaned = [clean(d) for d in dirty]
    for before, after in zip(dirty, cleaned):
        print(f"    {before.strip()[:22]:24s} -> {after}")
    print()

    print("[4] 前処理の前後で単語頻度 top8 を比較")
    before_freq = word_freq(dirty)
    after_freq = word_freq(cleaned)
    print("    前処理前                              | 前処理後")
    print("    " + "-" * 60)
    for (bw, bc), (aw, ac) in zip(before_freq, after_freq):
        print(f"    {bw[:18]:20s} {bc}回      | {aw:10s} {ac}回")
    print()
    print("    -> 前処理前は文全体が 1 つの塊として数えられ、「ＴＨＡＮＫＳ。」のようなノイズも混ざりますが、")
    print("       前処理後は「品質」「梱包」のようなキーワード単位に揃い、集計がレポートに使える形になります。")
    print("\n結論: 前処理は「目的に合わせた下ごしらえ」です。次のレベルで正規表現を本格的に学びます。")
