"""
level00 — 计算机处理语言为什么难

用最简单的字符串匹配 (关键词搜索) 搭一个客服机器人，
再用 4 个失败案例演示它在 同义词 / 歧义 / 上下文(反话) /
"中文没有词边界" 这些特性面前是怎么塌掉的。
只使用标准库。
"""


def keyword_match(text: str, keywords: list[str]) -> bool:
    """最简单的做法: 只要任意一个关键词以"子字符串"形式出现就返回 True。"""
    return any(kw in text for kw in keywords)


# ---------------------------------------------------------------------------
# [1] 同义词问题 — 意思相同，字面不同
# ---------------------------------------------------------------------------

REFUND_REQUESTS = [  # 全部都是"想要退款"的客户咨询。
    "麻烦帮我办理退款",
    "把钱退给我吧",
    "我想取消这笔支付",
    "取消订单后请把款打回来",
    "退款什么时候到账",
    "可以走赔付流程吗",
]


def demo_synonym() -> None:
    print("[1] 同义词问题 — 搜关键词'退款'能找到几条?")
    keywords = ["退款"]
    hit = 0
    for text in REFUND_REQUESTS:
        found = keyword_match(text, keywords)
        hit += found
        mark = "命中  " if found else "漏掉 ×"
        print(f"    {mark} | {text}")
    print(f"    -> 实际 {len(REFUND_REQUESTS)} 条退款请求中只检出 {hit} 条 "
          f"(命中率 {hit / len(REFUND_REQUESTS):.0%})")
    print("    -> '把钱退给我'和'退款'意思相同这件事，字符串并不知道。\n")


# ---------------------------------------------------------------------------
# [2] 歧义问题 — "苹果"的两种意思
# ---------------------------------------------------------------------------

AMBIGUOUS_SENTENCES = [
    ("苹果又发布了新款手机", "company"),
    ("苹果洗干净放在果盘里", "fruit"),
    ("这苹果又脆又甜", "fruit"),
    ("苹果的系统更新了", "company"),
    ("削苹果的时候小心点", "fruit"),
    ("苹果的股价涨了", "company"),
]

# 周边词 (上下文) 提示: 区分意思的线索不在"苹果"本身，而在它的邻居。
CONTEXT_HINTS = {
    "company": ["手机", "发布", "系统", "股价", "门店"],
    "fruit": ["甜", "洗", "削", "果盘", "脆", "一斤"],
}


def guess_by_context(sentence: str) -> str:
    """靠邻居词的提示来猜词义的超迷你"上下文"分类器。"""
    for sense, hints in CONTEXT_HINTS.items():
        if any(h in sentence for h in hints):
            return sense
    return "?"


def demo_ambiguity() -> None:
    print("[2] 歧义问题 — 同样两个字'苹果'，在计算机眼里完全相同")
    correct = 0
    for sentence, answer in AMBIGUOUS_SENTENCES:
        naive = "苹果" in sentence        # 字符串匹配: 全部以"发现苹果"告终
        guess = guess_by_context(sentence)
        correct += guess == answer
        print(f"    字符串匹配={'发现苹果' if naive else '-'} | "
              f"上下文推测={guess:7s} | 正确答案={answer:7s} | {sentence}")
    print(f"    -> 字符串匹配无法区分两种意思，但只要看'周边的词'，"
          f"就能分对 {correct}/{len(AMBIGUOUS_SENTENCES)}")
    print("    -> '词的意思由邻居决定' — 这正是 level06 词嵌入的核心思想。\n")


# ---------------------------------------------------------------------------
# [3] 上下文·反话问题 — 好评关键词的背叛
# ---------------------------------------------------------------------------

REVIEWS = [  # (评论, 真实情感 1=好评 0=差评)
    ("质量真的很好", 1),
    ("物流也快，好评", 1),
    ("说是很好才买的，结果彻底失望", 0),
    ("都说最好用，结果一天就坏了", 0),
    ("性价比不高，不太满意", 0),
    ("包装很用心，满意", 1),
]


def demo_sarcasm() -> None:
    print("[3] 上下文问题 — 用'好/满意'关键词判定好评会怎样?")
    keywords = ["好", "满意", "赞"]
    wrong = 0
    for text, label in REVIEWS:
        pred = 1 if keyword_match(text, keywords) else 0
        ok = pred == label
        wrong += not ok
        mark = "判对  " if ok else "判错 ×"
        print(f"    {mark} | 预测={'好评' if pred else '差评'} "
              f"实际={'好评' if label else '差评'} | {text}")
    print(f"    -> {len(REVIEWS)} 条中判错 {wrong} 条。"
          f"'好'这个字和'好这个意思'是两回事。\n")


# ---------------------------------------------------------------------------
# [4] 无词边界问题 — 中文里词与词之间没有空格
# ---------------------------------------------------------------------------

DELIVERY_SENTENCES = [
    "配送速度快得惊人",
    "配送虽然慢但包装完好",
    "还在等配送",
    "配送和服务都满意",
    "请重新配送一次",      # 含"配送"但是另一个概念 (补发/重发)
    "免费配送真不错",      # 这也是复合词
]


def demo_no_boundary() -> None:
    print("[4] 无词边界问题 — 想找'配送'，可词与词之间没有缝")
    exact = [s for s in DELIVERY_SENTENCES if "配送" in s.split()]     # 按空格切的精确匹配
    substr = [s for s in DELIVERY_SENTENCES if "配送" in s]            # 子字符串包含
    print(f"    按空格切分后精确匹配('配送'单独成词): 检出 {len(exact)} 条 {exact}")
    print(f"    子字符串包含('配送' in s)            : 检出 {len(substr)} 条")
    for s in substr:
        note = " <- 重新配送/免费配送也被拽了进来" if ("重新配送" in s or "免费配送" in s) else ""
        print(f"        - {s}{note}")
    cross = "和服" in "配送和服务都满意"
    print(f"    再看一个经典事故: '和服' in '配送和服务都满意' -> {cross}")
    print("       ('配送|和|服务' 的词边界被子字符串横跨，凭空匹配出一件'和服'!)")
    print("    -> 中文没有空格，split() 把整句当成一个块，精确匹配 0 条；")
    print("       子字符串包含又会跨越词边界乱抓。这就是需要分词 (level03) 的原因。\n")


# ---------------------------------------------------------------------------
# [5] 路线图
# ---------------------------------------------------------------------------

def print_roadmap() -> None:
    print("[5] 今天遇到的这些失败，这门课会这样逐个解决")
    roadmap = [
        ("同义词·写法不一", "level01 预处理，level06 嵌入(意思相近则坐标也相近)"),
        ("歧义(上下文)", "level07 RNN，level08 注意力(看周边的词来定意思)"),
        ("反话·否定表达", "level05 分类模型 + level08 之后的上下文模型"),
        ("无词边界(分词)", "level03 分词·最大匹配法，子词(BPE)"),
        ("新词", "level03 子词(没见过的词也拆成碎片处理)"),
    ]
    for problem, solution in roadmap:
        print(f"    {problem:12s} -> {solution}")
    print("\n结论: 语言不是'字符'，而是'用法'。从下一关开始逐个攻克。")


if __name__ == "__main__":
    print("=" * 70)
    print("计算机处理语言为什么难 — 简单字符串匹配的崩塌实验")
    print("=" * 70 + "\n")
    demo_synonym()
    demo_ambiguity()
    demo_sarcasm()
    demo_no_boundary()
    print_roadmap()
