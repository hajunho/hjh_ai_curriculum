"""
数据质量、去重与污染检查实战。
- 用字符 n-gram (shingle) 集合与杰卡德相似度 (Jaccard) 检测并清除完全重复/近似重复文档。
- 用 n-gram 重叠检查"基准测试题混进了训练语料"这种污染 (考题泄露)，
  并用模拟实验看看污染把基准分数抬高了多少。
"""
import random
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

DUP_THRESHOLD = 0.7    # 达到这个值就判定为近似重复
NGRAM_N = 5            # 字符 n-gram 的长度
CONTAM_N = 10          # 污染检查用的 n-gram 长度 (越长越只抓"整段照搬")
CONTAM_THRESHOLD = 0.3 # 题目 n-gram 中有这个比例以上出现在语料里就怀疑泄露


def char_ngrams(text: str, n: int) -> set:
    """把字符串切成长度为 n 的连续碎片 (字符 n-gram) 集合。"""
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard(a: set, b: set) -> float:
    """杰卡德相似度 = |交集| / |并集|。完全相同为 1.0，毫不相干为 0.0。"""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def contamination_ratio(question: str, corpus_grams: set, n: int) -> float:
    """题目的 n-gram 中原封不动地存在于训练语料里的比例。"""
    q_grams = char_ngrams(question, n)
    if not q_grams:
        return 0.0
    return sum(1 for g in q_grams if g in corpus_grams) / len(q_grams)


def build_documents(rng: random.Random):
    """用 tiny_corpus 的句子造 24 份文档，并故意埋下重复/泄露的案例。"""
    sentences = [s.strip() + "。" for s in hjh_data.tiny_corpus().split("。 ") if s.strip()]
    docs = []
    for i in range(24):
        picked = rng.sample(sentences, 3)          # 3 句 = 1 份文档
        docs.append(" ".join(picked))
    docs[15] = docs[2]                             # 埋一份完全复制品
    docs[19] = docs[5].replace("做好了", "完成了", 1)  # 只改一个词的近似复制品
    return docs


BENCHMARK = [  # (题目, 正确答案) — 假设是一份四选一的常识考试
    ("中国的首都是哪里？ 选项: 上海, 北京, 广州, 成都", "北京"),
    ("水沸腾时的摄氏温度是多少？ 选项: 50, 80, 100, 120", "100"),
    ("一周有几天？ 选项: 5天, 6天, 7天, 8天", "7天"),
    ("三角形的内角和是多少度？ 选项: 90, 180, 270, 360", "180"),
    ("光和声音哪个更快？ 选项: 光, 声音, 一样快, 不确定", "光"),
    ("一年大约有多少天？ 选项: 300天, 330天, 365天, 400天", "365天"),
]
LEAKED_IDX = 3  # 把这道题泄露进训练文档


def memorizer_score(corpus_text: str, items, rng: random.Random):
    """"只会答背过的题"的假想模型: 题目整段出现在语料里就答对，
    否则四选一瞎猜 (正确率 25%)。返回: 每道题的 (见过?, 对错) 列表。"""
    results = []
    for question, _answer in items:
        seen = question in corpus_text          # 这题在"考试宝典"里见过吗？
        correct = True if seen else (rng.random() < 0.25)
        results.append((seen, correct))
    return results


def main():
    rng = random.Random(42)  # 固定 seed 以保证可复现

    # [1] 构建语料 ---------------------------------------------------------
    docs = build_documents(rng)
    q_leak, a_leak = BENCHMARK[LEAKED_IDX]
    docs[9] = docs[9] + f" 今日常识小测验。{q_leak} 答案是{a_leak}。" # 泄露文档
    print("[1] 训练语料构建: 文档", len(docs), "份")
    print("    - 埋下的问题: 完全复制品(2↔15)、近似复制品(5↔19)、基准题泄露(文档 9)")
    print("    - 文档示例:", docs[0][:44], "...")

    # [2] 近似重复检测 ------------------------------------------------------
    grams = [char_ngrams(d, NGRAM_N) for d in docs]
    dup_pairs = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            sim = jaccard(grams[i], grams[j])
            if sim >= DUP_THRESHOLD:
                dup_pairs.append((i, j, sim))
    print(f"\n[2] 重复检测 (字符 {NGRAM_N}-gram 杰卡德 >= {DUP_THRESHOLD})")
    for i, j, sim in dup_pairs:
        kind = "完全重复" if sim > 0.999 else "近似重复"
        print(f"    - 文档 {i:2d} ↔ 文档 {j:2d} : 杰卡德 {sim:.3f}  → {kind}")
    if not dup_pairs:
        print("    - 未发现重复")

    # [3] 去重 -------------------------------------------------------------
    drop = {j for _i, j, _s in dup_pairs}       # 丢掉后出现的那一份
    kept = [d for k, d in enumerate(docs) if k not in drop]
    print(f"\n[3] 去重: {len(docs)} 份 → {len(kept)} 份 (丢弃的文档: {sorted(drop)})")
    print("    省下了本会被浪费的训练 token，也降低了死记某份文档的风险。")

    # [4] 基准污染检查 ------------------------------------------------------
    corpus_text = " ".join(kept)
    corpus_grams = char_ngrams(corpus_text, CONTAM_N)
    print(f"\n[4] 污染检查 (题目 {CONTAM_N}-gram 出现在语料中的比例 >= {CONTAM_THRESHOLD:.0%})")
    contaminated = []
    for idx, (question, _a) in enumerate(BENCHMARK):
        ratio = contamination_ratio(question, corpus_grams, CONTAM_N)
        flag = ratio >= CONTAM_THRESHOLD
        if flag:
            contaminated.append(idx)
        mark = "★疑似泄露" if flag else "干净"
        print(f"    - 第 {idx} 题: 重叠 {ratio:5.1%}  [{mark}]  {question[:26]}...")

    # [5] 演示污染如何抬高分数 ----------------------------------------------
    print("\n[5] “只会答背过的题”的假想模型，其基准分数对比")
    results = memorizer_score(corpus_text, BENCHMARK, random.Random(6))  # 专用于瞎猜的 seed
    total = len(BENCHMARK)
    score_all = sum(c for _s, c in results) / total
    clean_items = [r for k, r in enumerate(results) if k not in contaminated]
    score_clean = sum(c for _s, c in clean_items) / max(1, len(clean_items))
    for k, (seen, correct) in enumerate(results):
        note = "宝典里见过 → 自动答对" if seen else ("瞎猜蒙对" if correct else "答错")
        print(f"    - 第 {k} 题: {'O' if correct else 'X'}  ({note})")
    print(f"    含污染的分数 : {score_all:.1%}  ← 最容易被写进汇报材料的数字")
    print(f"    干净分数     : {score_clean:.1%}  ← 更接近真实水平的数字")
    print("    → 一道泄露的题就能把分数吹起来。请记得问一句: “这个分数做过污染检查吗？”")


if __name__ == "__main__":
    main()
