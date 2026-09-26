"""
lecture11 公用模拟 (mock) LLM / 嵌入模块。

这是一套基于规则的实现，让我们不需要 API 密钥和网络也能上 LLM 课。
- MockLLM       : 用模式匹配模仿总结、分类、写初稿等任务的语言模型。
                  收到检索出的上下文 (context) 时，会重新组装其中的句子来回答。
- MockEmbedding : 字符 n-gram 哈希 + 词语共现 (co-occurrence) 扩展向量。
                  同一主题的句子真的会算出较高的余弦相似度。
在真实服务中，这个位置放的就是 Claude API 等真正的模型。
"""

import re
import zlib

import numpy as np


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """非常简单的中文分句器：按句号/问号/叹号切分。
    中文标点是全角 (。？！) 且句间没有空格，所以直接在标点后切开。"""
    parts = re.split(r"(?<=[。？！.?!])\s*", text.strip())
    return [p.strip() for p in parts if p.strip()]


def tokenize(text: str) -> list[str]:
    """中文没有空格，不能按空格切词。这里把连续汉字切成字符 2-gram，
    英文/数字串整体保留为一个 token。(不用分词器也能保证课程可复现)"""
    tokens = []
    for run in re.findall(r"[一-鿿]+|[a-zA-Z0-9]+", text):
        if re.match(r"[a-zA-Z0-9]", run) or len(run) < 2:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度。向量若已归一化，则等价于点积。"""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# MockEmbedding — 字符 n-gram 哈希 + 共现扩展
# ---------------------------------------------------------------------------

class MockEmbedding:
    """
    文本 -> 固定长度向量。

    原理:
    1) 抽取字符 2~3-gram，用 CRC32 哈希把计数加到 dim 个槽位之一。
       "年假申请"和"申请年假"这类写法略有差异的文本，n-gram 仍会重叠，
       所以向量彼此接近。
    2) 用 fit() 学习语料后，会记住每个词的共现邻居；做嵌入时把邻居词的
       n-gram 也弱化地混进来。
       ("年假"句和"休假"句若经常出现在同一文档里，两者就会互相靠近)
    """

    def __init__(self, dim: int = 512, ngram_range: tuple = (2, 3),
                 neighbor_weight: float = 0.35, max_neighbors: int = 4):
        self.dim = dim
        self.ngram_range = ngram_range
        self.neighbor_weight = neighbor_weight
        self.max_neighbors = max_neighbors
        self._cooc: dict[str, list[str]] = {}   # 词 -> 共现邻居词列表

    # -- 内部: 抽取字符 n-gram ---------------------------------------------
    def _char_ngrams(self, text: str) -> list[str]:
        cleaned = re.sub(r"[^0-9A-Za-z一-鿿 ]", "", text).replace(" ", "_")
        grams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            grams.extend(cleaned[i:i + n] for i in range(len(cleaned) - n + 1))
        return grams

    def _add_ngrams(self, vec: np.ndarray, text: str, weight: float) -> None:
        for g in self._char_ngrams(text):
            idx = zlib.crc32(g.encode("utf-8")) % self.dim   # 稳定的哈希
            vec[idx] += weight

    # -- 学习: 共现统计 -------------------------------------------------------
    def fit(self, texts: list[str]) -> "MockEmbedding":
        """从句子列表统计"在同一句里一起出现的词"。"""
        counts: dict[str, dict[str, int]] = {}
        for text in texts:
            words = [w for w in tokenize(text) if len(w) >= 2]
            for w in words:
                bucket = counts.setdefault(w, {})
                for other in words:
                    if other != w:
                        bucket[other] = bucket.get(other, 0) + 1
        self._cooc = {
            w: [x for x, _ in sorted(nb.items(), key=lambda kv: -kv[1])[: self.max_neighbors]]
            for w, nb in counts.items()
        }
        return self

    # -- 嵌入 ------------------------------------------------------------------
    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        self._add_ngrams(vec, text, 1.0)
        # 弱化地混入共现邻居词 ("年假"提问 -> 与"休假"文档也拉近)
        for w in tokenize(text):
            for nb in self._cooc.get(w, []):
                self._add_ngrams(vec, nb, self.neighbor_weight)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.embed(t) for t in texts])


# ---------------------------------------------------------------------------
# MockLLM — 基于规则的模拟语言模型
# ---------------------------------------------------------------------------

# 客诉/咨询分类用关键词词典
_CATEGORY_LEXICON = {
    "物流配送": ["配送", "快递", "物流", "延迟", "到货", "发货", "运单"],
    "退款/支付": ["退款", "支付", "扣款", "账单", "取消", "重复"],
    "质量/故障": ["故障", "坏了", "破损", "噪音", "不能用", "维修", "质量"],
    "服务态度": ["店员", "客服", "态度", "冷淡", "接待", "排队", "等待"],
}


class MockLLM:
    """
    基于规则的模拟 LLM。

    complete(prompt)              : 根据提示词里的指令词 (总结/分类/邮件...) 生成
                                    模式化回复。角色、格式、示例写得越清楚的
                                    "好提示词"，得到的回答越结构化。
    answer_with_context(q, ctxs)  : RAG 专用。从检索到的上下文中挑出与问题
                                    重叠的句子，引用出处组装成回答。
    """

    def __init__(self, name: str = "mock-llm-v1"):
        self.name = name

    # -- 从提示词中分离正文 (payload) -----------------------------------------
    @staticmethod
    def _payload(prompt: str) -> str:
        for marker in ("正文：", "正文:", "内容：", "内容:", "---"):
            if marker in prompt:
                return prompt.split(marker, 1)[1].strip()
        return prompt.strip()

    # -- 检测提示词质量信号 ----------------------------------------------------
    @staticmethod
    def _quality_flags(prompt: str) -> dict[str, bool]:
        return {
            "role": ("你是" in prompt or "角色" in prompt),
            "format": any(k in prompt for k in ("格式", "要点", "JSON", "编号", "条列")),
            "example": "示例" in prompt,
            "context": any(k in prompt for k in ("情境", "目标读者", "背景")),
        }

    # -- 各子任务 ----------------------------------------------------------------
    def _summarize(self, payload: str, n: int, structured: bool) -> str:
        """抽取式总结: 挑出包含最多高频词的句子。"""
        sents = split_sentences(payload)
        freq: dict[str, int] = {}
        for s in sents:
            for w in tokenize(s):
                if len(w) >= 2:
                    freq[w] = freq.get(w, 0) + 1
        scored = [(sum(freq.get(w, 0) for w in tokenize(s)) / (len(tokenize(s)) + 1), i, s)
                  for i, s in enumerate(sents)]
        top = sorted(sorted(scored, key=lambda x: -x[0])[:n], key=lambda x: x[1])
        if structured:
            return "\n".join(f"- {s}" for _, _, s in top)
        return top[0][2] if top else "(没有可以总结的内容)"

    def _classify(self, payload: str) -> tuple[str, list[str]]:
        best, best_hits = "其他", []
        for cat, words in _CATEGORY_LEXICON.items():
            hits = [w for w in words if w in payload]
            if len(hits) > len(best_hits):
                best, best_hits = cat, hits
        return best, best_hits

    def _draft_email(self, prompt: str, structured: bool) -> str:
        def _field(key, default):
            m = re.search(key + r"\s*[:：]\s*(.+)", prompt)
            return m.group(1).strip() if m else default
        to = _field("收件人", "相关负责人")
        purpose = _field("目的", "工作协作事宜")
        deadline = _field("期限", "")
        body = [f"{to}：", "", f"您好。现就{purpose}与您联系。"]
        if deadline:
            body.append(f"如果方便，请在{deadline}之前回复。")
        body += ["如审阅时需要补充材料，请随时告知。", "", "谢谢！", "敬上"]
        if not structured:
            return f"就{purpose}发一封邮件应该就可以。开头写您好，结尾写谢谢。"
        return "\n".join(body)

    # -- 公开 API -----------------------------------------------------------------
    def complete(self, prompt: str) -> str:
        """看指令词把任务路由到对应子功能的模拟应答器。"""
        flags = self._quality_flags(prompt)
        structured = flags["format"] or flags["example"]
        payload = self._payload(prompt)

        if "分类" in prompt:
            cat, hits = self._classify(payload)
            if structured:
                return f"分类: {cat}\n依据关键词: {', '.join(hits) if hits else '无'}"
            return f"嗯，这看起来像是{cat}方面的内容。"
        if "总结" in prompt or "摘要" in prompt:
            n = 3 if structured else 1
            return self._summarize(payload, n, structured)
        if "邮件" in prompt:
            return self._draft_email(prompt, structured or flags["role"])
        if "翻译" in prompt:
            return "(模拟翻译) Hello, this is a mock translation of the given text."
        return "已理解您的请求。提供更具体的指令 (角色、格式、示例) 可以提高准确度。"

    def answer_with_context(self, question: str, contexts: list[tuple[str, str]],
                            max_evidence: int = 2) -> str:
        """
        RAG 回答组装器。contexts = [(出处, 正文), ...]
        挑出与问题字符 2-gram 重叠的句子作为依据并引用。找不到依据时
        直接说不知道 (这就是抑制幻觉的最小形态)。
        """
        q_words = {w for w in tokenize(question) if len(w) >= 2}
        evidence = []
        for source, text in contexts:
            for sent in split_sentences(text):
                s_words = tokenize(sent)
                overlap = sum(1 for w in s_words if w in q_words)
                if overlap > 0:
                    evidence.append((overlap, source, sent))
        if not evidence:
            return "在提供的文档里没有找到依据，这个问题我不知道。"
        evidence.sort(key=lambda x: -x[0])
        picked = evidence[:max_evidence]
        lines = ["根据文档回答如下。"]
        for _, source, sent in picked:
            lines.append(f"- {sent} (出处: {source})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# [参考] 真实 API 的样子 — 有密钥时调用 Claude API 的示例
# ---------------------------------------------------------------------------
# import anthropic                              # pip install anthropic
# client = anthropic.Anthropic()                # 使用 ANTHROPIC_API_KEY 环境变量
# response = client.messages.create(
#     model="claude-opus-5",
#     max_tokens=1024,
#     messages=[{"role": "user", "content": "请把会议纪要总结成3行: ..."}],
# )
# print(response.content[0].text)
# ---------------------------------------------------------------------------
