"""
level01 — 文本预处理基础

把脏乱的中文评论 (表情符号、URL、重复字符、全角字符、繁体异形字、
杂乱空白、停用词) 用分步流水线洗干净。
再对比预处理前后的词频统计，亲眼确认"为什么预处理是必修课"。
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
# [1] 制造脏评论 — 现实中的文本从来不干净
# ---------------------------------------------------------------------------

NOISE_PREFIX = ["★★★ ", "【买家秀】", "", "♡♡ "]
NOISE_SUFFIX = [" 哈哈哈哈哈", " 呜呜呜呜", "！！！！！", " 好好好好好",
                " 参考：http://blog.example.com/review123", ""]
# 繁体异形字: 意思完全一样，但字符码点不同 -> 会被当成两个不同的词来统计
VARIANTS = {"满意": "滿意", "价格": "價格", "质量": "質量", "包装": "包裝",
            "颜色": "顏色", "环境": "環境"}


def make_dirty_reviews(n: int = 8, seed: int = 42) -> list[str]:
    """给 hjh_data 的评论故意掺进杂质，做出"很现实"的评论。"""
    rng = random.Random(seed)
    base = hjh_data.review_corpus(n * 3, seed=3)[:n]
    dirty = []
    for row in base:
        text = (rng.choice(NOISE_PREFIX) + row["text"].replace(" ", "　　", 1)
                + rng.choice(NOISE_SUFFIX))
        if rng.random() < 0.5:
            text = "  " + text + "   "          # 前后杂乱空白
        if rng.random() < 0.4:
            text = text + "ＧＯＯＤ！"           # 全角英文字母混入
        if rng.random() < 0.5:                  # 一半写成繁体异形字
            for simp, trad in VARIANTS.items():
                text = text.replace(simp, trad)
        dirty.append(text)
    return dirty


# ---------------------------------------------------------------------------
# [2] 定义流水线的每一步 — 字符串进、字符串出的小函数
# ---------------------------------------------------------------------------

# 停用词: 几乎每句都有、区分不出文档的字词
STOPWORDS = ["的", "了", "着", "很", "非常", "真", "太", "也", "就", "都",
             "这个", "这种", "不过", "参考", "买家秀"]
# 中文标点 — 它们是中文里唯一"免费"的词边界线索
PUNCT = "。，、！？；：…（）《》“”‘’—·"


def normalize_width_and_strip_url(text: str) -> str:
    """全角转半角 (NFKC) + 转小写 + 删 URL。URL 先删掉，后面几步才轻松。"""
    text = unicodedata.normalize("NFKC", text)   # ＧＯＯＤ -> GOOD, 全角空格 -> 半角
    text = text.lower()
    return re.sub(r"https?://\S+", " ", text)


def normalize_variants(text: str) -> str:
    """繁体异形字统一成简体: 滿意 -> 满意, 價格 -> 价格。"""
    for simp, trad in VARIANTS.items():
        text = text.replace(trad, simp)
    return text


def remove_special(text: str) -> str:
    """只留汉字/英文/数字/中文标点/空白，其余 (★, ♡, 【】 等) 一律去掉。"""
    return re.sub(rf"[^一-鿿a-z0-9\s{re.escape(PUNCT)}]", " ", text)


def collapse_repeats(text: str) -> str:
    """同一个字符连续 3 次以上压成 2 次: 好好好好好 -> 好好, 哈哈哈哈 -> 哈哈"""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def punct_to_space(text: str) -> str:
    """标点换成空格 (中文没有空格，标点是唯一现成的边界) + 空白规范化。"""
    text = re.sub(rf"[{re.escape(PUNCT)}]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def drop_stopwords(text: str) -> str:
    """删停用词。中文没有空格，所以只能按字符串替换来删 (长词优先)。"""
    for sw in sorted(STOPWORDS, key=len, reverse=True):
        text = text.replace(sw, "")
    return re.sub(r"\s+", " ", text).strip()


PIPELINE = [
    ("半角化·小写·去URL", normalize_width_and_strip_url),
    ("异形字统一", normalize_variants),
    ("特殊符号清理", remove_special),
    ("重复字符压缩", collapse_repeats),
    ("标点转空格", punct_to_space),
    ("停用词删除", drop_stopwords),
]


def clean(text: str) -> str:
    """让文本按顺序通过整条流水线。"""
    for _, step in PIPELINE:
        text = step(text)
    return text


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------

def demo_step_by_step(sample: str) -> None:
    print("[2] 让一条评论逐步通过流水线")
    print(f"    原文: {sample!r}")
    text = sample
    for i, (name, step) in enumerate(PIPELINE, start=1):
        text = step(text)
        print(f"    第{i}步 {name:14s} -> {text!r}")
    print()


def bigram_freq(texts: list[str], top: int = 8) -> list[tuple[str, int]]:
    """中文没有空格，所以先用"相邻两字"(二元组) 粗略代替词来统计。"""
    counter = Counter()
    for t in texts:
        for chunk in t.split():
            for i in range(len(chunk) - 1):
                pair = chunk[i:i + 2]
                if re.fullmatch(r"[一-鿿]{2}", pair):
                    counter[pair] += 1
    return counter.most_common(top)


if __name__ == "__main__":
    print("=" * 70)
    print("文本预处理 — 用流水线把脏评论洗干净")
    print("=" * 70 + "\n")

    dirty = make_dirty_reviews(n=8, seed=42)
    print(f"[1] 生成脏评论 {len(dirty)} 条 (含表情符号·URL·重复字符·全角·繁体·杂空白)")
    for d in dirty[:3]:
        print(f"    例: {d!r}")
    print()

    demo_step_by_step(dirty[0])

    print("[3] 全部一次性清洗 (原文 -> 结果)")
    cleaned = [clean(d) for d in dirty]
    for before, after in zip(dirty, cleaned):
        print(f"    {before.strip()[:24]:26s} -> {after}")
    print()

    print("[4] 预处理前后的二字词频 top8 对比")
    before_freq = bigram_freq(dirty)
    after_freq = bigram_freq(cleaned)
    print("    预处理前                  | 预处理后")
    print("    " + "-" * 46)
    for (bw, bc), (aw, ac) in zip(before_freq, after_freq):
        print(f"    {bw:10s} {bc}次           | {aw:10s} {ac}次")
    print()
    print("    异形字被拆成两份统计的现场 (为看清差距，这里用 100 条样本):")
    big_dirty = make_dirty_reviews(n=100, seed=42)
    big_clean = [clean(d) for d in big_dirty]
    for word, trad in [("满意", "滿意"), ("包装", "包裝")]:
        b_simp = sum(t.count(word) for t in big_dirty)
        b_trad = sum(t.count(trad) for t in big_dirty)
        a_simp = sum(t.count(word) for t in big_clean)
        print(f"      预处理前 {word}={b_simp} / {trad}={b_trad}  ->  预处理后 {word}={a_simp}")
    print()
    print("    -> 预处理前，'滿意'和'满意'各算一份、'哈哈'之类噪声还挤进前列；")
    print("       预处理后同一个概念合成一份，统计结果终于能写进周报了。")
    print("\n结论: 预处理是'按目的来处理食材'。下一关正式学正则表达式。")
