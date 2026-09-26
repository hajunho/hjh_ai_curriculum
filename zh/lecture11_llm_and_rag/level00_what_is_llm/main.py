"""
把 LLM (大规模语言模型) 的原理当成"下一个词预测器"来体验。
1) 在迷你中文语料上统计 n-gram (词语接续的概率)。
2) 查看特定上下文之后出现各个词的概率分布。
3) 按概率一个词一个词地抽样，"生成"出句子。
真实的 LLM 就是把这个原理扩展成了数千亿参数的神经网络。
"""

import pathlib
import random
import sys
from collections import Counter, defaultdict

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data


def build_ngram_model(words: list[str], n: int) -> dict:
    """建立"前 (n-1) 个词 (上下文) -> 下一个词出现次数"的统计表。"""
    model = defaultdict(Counter)
    for i in range(len(words) - n + 1):
        context = tuple(words[i:i + n - 1])
        nxt = words[i + n - 1]
        model[context][nxt] += 1
    return model


def next_word_distribution(model: dict, context: tuple) -> list[tuple[str, float]]:
    """计算上下文之后各个候选词的概率分布。"""
    counter = model.get(context, Counter())
    total = sum(counter.values())
    if total == 0:
        return []
    return [(w, c / total) for w, c in counter.most_common()]


def generate(model: dict, start: tuple, rng: random.Random,
             max_words: int = 12, greedy: bool = False) -> str:
    """从上下文出发，反复抽出下一个词来造句。
    greedy=True 时永远只选第 1 名 (温度 0)，False 时按概率抽样。"""
    out = list(start)
    context = start
    for _ in range(max_words):
        dist = next_word_distribution(model, context)
        if not dist:
            break
        if greedy:
            word = dist[0][0]                      # 概率最高的词
        else:
            words, probs = zip(*dist)
            word = rng.choices(words, weights=probs, k=1)[0]
        out.append(word)
        if word.endswith("。"):                    # 出现句号就结束这句话
            break
        context = tuple(out[-(len(start)):])       # 上下文窗口向后滑一格
    return " ".join(out)


def main() -> None:
    rng = random.Random(42)                        # 固定种子保证可复现

    print("=" * 62)
    print("Level 00 | LLM 是什么 — 超大号'下一个词预测器'")
    print("=" * 62)

    # [1] 训练数据: 迷你中文语料
    corpus = hjh_data.tiny_corpus()
    words = corpus.split()
    print(f"\n[1] 训练语料: {len(corpus):,} 字符，{len(words):,} 个词")
    print(f"    开头示例: {' '.join(words[:12])} ...")

    # [2] 做统计 = "训练"。参数越多，能记住的上下文越长。
    bigram = build_ngram_model(words, n=2)
    trigram = build_ngram_model(words, n=3)
    print(f"\n[2] 训练完成 — bigram 上下文 {len(bigram):,} 种，trigram 上下文 {len(trigram):,} 种")
    print("    (真实 LLM 记住这些靠的不是'表'，而是数千亿个参数的神经网络)")

    # [3] 下一个词的概率分布 — LLM 每生成一个 token 都在做这件事
    print("\n[3] '上班族' 之后出现各个词的概率分布 (前 5 名)")
    for w, p in next_word_distribution(bigram, ("上班族",))[:5]:
        bar = "#" * int(p * 40)
        print(f"    {w:<10} {p:6.1%} {bar}")

    print("\n    '昨天 开发者' 之后的词 (trigram, 前 5 名)")
    for w, p in next_word_distribution(trigram, ("昨天", "开发者"))[:5]:
        bar = "#" * int(p * 40)
        print(f"    {w:<12} {p:6.1%} {bar}")

    # [4] 生成 — 一个词一个词地抽出来组成句子 (token 级生成)
    print("\n[4] 句子生成: 同一个开头，抽样每次得到不同的句子")
    for i in range(3):
        print(f"    样本 {i + 1}: {generate(trigram, ('今天', '厨师'), rng)}")
    print(f"    贪心式(永远选第1名): {generate(trigram, ('今天', '厨师'), rng, greedy=True)}")
    print("    -> ChatGPT/Claude 对同一个问题给出不同答案，靠的就是这个'抽样'。")

    # [5] 体验局限 — 看似合理却在语料里没出现过的组合 = 幻觉的种子
    print("\n[5] 局限: '一本正经的胡说八道'(幻觉)的原理")
    sentences = set(s.strip() for s in corpus.split("。") if s.strip())
    novel, sample_novel = 0, ""
    for i in range(20):
        g = generate(trigram, ("周末", "学生"), rng).rstrip("。")
        if g not in sentences:
            novel += 1
            sample_novel = g
    print(f"    生成的 20 句中有 {novel} 句是训练数据里'没有过'的新组合。")
    print(f"    例: \"{sample_novel}。\"")
    print("    语法很自然，但没有任何事实核查 — LLM 的幻觉也是同样的原理。")

    # [6] 知识截止时间 (knowledge cutoff)
    print("\n[6] 知识截止时间: 这个模型对语料里没有的词一无所知。")
    print(f"    '新员工' 的下一个词分布: {next_word_distribution(bigram, ('新员工',)) or '无 (训练中没见过这个词)'}")
    print("    -> 真实 LLM 也不知道训练结束之后发生的事。(所以才需要 RAG)")

    print("\n总结: LLM = 用海量文本学会'下一个 token 概率'的超大号自动补全。")
    print("      流畅和真实是两码事 — 这正是整门课的出发点。")


if __name__ == "__main__":
    main()
