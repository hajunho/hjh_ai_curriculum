"""
lecture11 共通のモック (mock) LLM / 埋め込みモジュール。

API キーもインターネットも使わずに LLM の授業を進めるための、ルールベース実装です。
- MockLLM       : パターンマッチで要約・分類・下書き作成などを真似る言語モデル。
                  検索された文脈 (context) を渡すと、その文を組み替えて答えます。
- MockEmbedding : 文字 n-gram ハッシュ + 単語の共起 (co-occurrence) 拡張ベクトル。
                  同じ話題の文が実際にコサイン類似度で高く出ます。
実サービスでは、この場所に Claude API などの本物のモデルが入ります。

日本語対応のポイント: 日本語は単語の間に空白を置かないため、
「空白で切る」トークナイザーが使えません。そこで
  tokenize()       … 文字 2-gram を「単語」代わりに使う (検索エンジンの定番手法)
  tokenize_words() … ひらがなを落として漢字・カタカナの連なりを内容語として拾う
の 2 つを用意し、用途に応じて使い分けます。
"""

import re
import zlib

import numpy as np


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """句点「。」・感嘆符・疑問符を区切りとする、ごく素朴な日本語の文分割器。
    日本語は文と文の間に空白を入れないので、区切り文字の「直後」で切ります。"""
    parts = re.split(r"(?<=[。！？!?])", text.strip())
    return [p.strip() for p in parts if p.strip()]


_RUN = re.compile(r"[ぁ-んァ-ヶー一-龯々]+|[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """日本語向けトークン化: 文字 2-gram を「単語」の代わりに使います。
    分かち書きのない日本語で、形態素解析器なしに検索・類似度計算を
    成り立たせる定番の手法です (英数字の連なりはそのまま 1 トークン)。"""
    tokens = []
    for run in _RUN.findall(text):
        if re.match(r"[A-Za-z0-9]", run) or len(run) < 2:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


_CONTENT = re.compile(r"[ァ-ヶー]+|[一-龯々]+|[A-Za-z0-9]+")


def tokenize_words(text: str) -> list[str]:
    """「単語単位」の素朴なトークナイザー。
    ひらがな (助詞・活用語尾) を捨て、漢字・カタカナ・英数字の連なりを
    内容語として取り出します。形態素解析器の軽量な代用品で、
    共起統計と「キーワード完全一致検索」の実演に使います。"""
    return _CONTENT.findall(text)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """コサイン類似度。ベクトルが正規化済みなら内積と同じです。"""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# MockEmbedding — 文字 n-gram ハッシュ + 共起拡張
# ---------------------------------------------------------------------------

class MockEmbedding:
    """
    テキスト -> 固定長ベクトル。

    仕組み:
    1) 文字 2〜3 グラム (n-gram) を取り出し、CRC32 ハッシュで dim 個のマスの
       どれかにカウントを足します。「在宅勤務」と「在宅で」のように表記が
       少し違っても n-gram が重なるので、ベクトルが近づきます。
    2) fit() でコーパスを学習すると、内容語ごとの共起する隣人を覚えておき、
       埋め込むときに隣人語の n-gram も弱く混ぜます。
       (「在宅」の文と「勤務」の文が同じ文書に何度も現れるなら互いに近づく)
    """

    def __init__(self, dim: int = 512, ngram_range: tuple = (2, 3),
                 neighbor_weight: float = 0.35, max_neighbors: int = 4):
        self.dim = dim
        self.ngram_range = ngram_range
        self.neighbor_weight = neighbor_weight
        self.max_neighbors = max_neighbors
        self._cooc: dict[str, list[str]] = {}   # 内容語 -> 共起する隣人語

    # -- 内部: 文字 n-gram の取り出し ---------------------------------------
    def _char_ngrams(self, text: str) -> list[str]:
        cleaned = re.sub(r"[^0-9A-Za-zぁ-んァ-ヶー一-龯々]", "", text)
        grams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            grams.extend(cleaned[i:i + n] for i in range(len(cleaned) - n + 1))
        return grams

    def _add_ngrams(self, vec: np.ndarray, text: str, weight: float) -> None:
        for g in self._char_ngrams(text):
            idx = zlib.crc32(g.encode("utf-8")) % self.dim   # 安定したハッシュ
            vec[idx] += weight

    # -- 学習: 共起統計 ------------------------------------------------------
    def fit(self, texts: list[str]) -> "MockEmbedding":
        """文のリストから「同じ文に一緒に出た内容語」の統計を作ります。"""
        counts: dict[str, dict[str, int]] = {}
        for text in texts:
            words = [w for w in tokenize_words(text) if len(w) >= 2]
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

    # -- 埋め込み ------------------------------------------------------------
    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        self._add_ngrams(vec, text, 1.0)
        # 共起する隣人語を弱く混ぜる (「在宅」の質問 -> 「勤務」の文書にも近づく)
        for w in tokenize_words(text):
            for nb in self._cooc.get(w, []):
                self._add_ngrams(vec, nb, self.neighbor_weight)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.embed(t) for t in texts])


# ---------------------------------------------------------------------------
# MockLLM — ルールベースのモック言語モデル
# ---------------------------------------------------------------------------

# 問い合わせ分類用のキーワード辞書
_CATEGORY_LEXICON = {
    "配送": ["配送", "宅配", "遅延", "到着", "出荷", "追跡番号"],
    "返金/決済": ["返金", "決済", "カード", "請求", "キャンセル", "二重"],
    "品質/故障": ["故障", "不良", "破損", "異音", "動作", "修理"],
    "応対/サービス": ["店員", "オペレーター", "不親切", "応対", "つながら", "待た"],
}


class MockLLM:
    """
    ルールベースのモック LLM。

    complete(prompt)              : プロンプトの指示語 (要約/分類/メール...) を見て
                                    パターン応答を作ります。役割・形式・例が明記された
                                    「良いプロンプト」ほど構造化された答えを返します。
    answer_with_context(q, ctxs)  : RAG 用。検索された文脈から質問と重なる文を
                                    選び、根拠を引用した回答を組み立てます。
    """

    def __init__(self, name: str = "mock-llm-v1"):
        self.name = name

    # -- プロンプトから本文 (payload) を切り出す -----------------------------
    @staticmethod
    def _payload(prompt: str) -> str:
        for marker in ("本文:", "内容:", "---"):
            if marker in prompt:
                return prompt.split(marker, 1)[1].strip()
        return prompt.strip()

    # -- プロンプト品質のシグナル検出 ---------------------------------------
    @staticmethod
    def _quality_flags(prompt: str) -> dict[str, bool]:
        return {
            "role": ("あなたは" in prompt or "役割" in prompt),
            "format": any(k in prompt for k in ("形式", "箇条書き", "JSON", "番号を付け", "見出し")),
            "example": "例" in prompt,
            "context": any(k in prompt for k in ("状況", "読み手", "背景")),
        }

    # -- 下位タスク ----------------------------------------------------------
    def _summarize(self, payload: str, n: int, structured: bool) -> str:
        """抽出型要約: よく出てくる語を多く含む文を選びます。"""
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
        return top[0][2] if top else "(要約する内容がありません)"

    def _classify(self, payload: str) -> tuple[str, list[str]]:
        best, best_hits = "その他", []
        for cat, words in _CATEGORY_LEXICON.items():
            hits = [w for w in words if w in payload]
            if len(hits) > len(best_hits):
                best, best_hits = cat, hits
        return best, best_hits

    def _draft_email(self, prompt: str, structured: bool) -> str:
        def _field(key, default):
            m = re.search(key + r"\s*[:：]\s*(.+)", prompt)
            return m.group(1).strip() if m else default
        to = _field("宛先", "ご担当者")
        purpose = _field("目的", "業務のご協力のお願い")
        deadline = _field("期限", "")
        body = [f"{to} 様", "", f"いつもお世話になっております。{purpose}についてご連絡いたします。"]
        if deadline:
            body.append(f"お差し支えなければ、{deadline}までにご返信いただけますと幸いです。")
        body += ["ご確認に必要な資料がございましたらお知らせください。", "",
                 "よろしくお願いいたします。", "(署名)"]
        if not structured:
            return (f"{purpose}についてメールを送ればよいと思います。"
                    "お世話になっておりますで始めて、よろしくお願いいたしますで終えてください。")
        return "\n".join(body)

    # -- 公開 API ------------------------------------------------------------
    def complete(self, prompt: str) -> str:
        """指示語を見てタスクを振り分けるモック応答器。"""
        flags = self._quality_flags(prompt)
        structured = flags["format"] or flags["example"]
        payload = self._payload(prompt)

        if "分類" in prompt:
            cat, hits = self._classify(payload)
            if structured:
                return f"分類: {cat}\n根拠キーワード: {', '.join(hits) if hits else 'なし'}"
            return f"うーん、これは {cat} に関する内容のようです。"
        if "要約" in prompt:
            n = 3 if structured else 1
            return self._summarize(payload, n, structured)
        if "メール" in prompt:
            return self._draft_email(prompt, structured or flags["role"])
        if "翻訳" in prompt:
            return "(モック翻訳) Hello, this is a mock translation of the given text."
        return "ご依頼は理解しました。役割・形式・例など、より具体的な指示をいただけると精度が上がります。"

    def answer_with_context(self, question: str, contexts: list[tuple[str, str]],
                            max_evidence: int = 2) -> str:
        """
        RAG 用の回答組み立て器。contexts = [(出典, 本文), ...]
        質問とトークンが重なる文を根拠として選び、引用します。根拠がなければ
        「わかりません」と答えます (ハルシネーション抑制の最小形)。
        """
        q_words = {w for w in tokenize(question) if len(w) >= 2}
        # 内容語だけの集合も併用し、ひらがなだけの偶然の一致に引っぱられないようにします。
        q_content = {w for w in tokenize_words(question) if len(w) >= 2}
        evidence = []
        for source, text in contexts:
            for sent in split_sentences(text):
                s_words = tokenize(sent)
                overlap = sum(1 for w in s_words if w in q_words)
                overlap += 2 * sum(1 for w in tokenize_words(sent) if w in q_content)
                if overlap > 0:
                    evidence.append((overlap, source, sent))
        if not evidence:
            return "提供された文書の中に根拠が見つかりませんでした。申し訳ありませんが、わかりません。"
        evidence.sort(key=lambda x: -x[0])
        picked = evidence[:max_evidence]
        lines = ["文書に基づいてお答えします。"]
        for _, source, sent in picked:
            lines.append(f"- {sent} (出典: {source})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# [参考] 本物の API ならこう書きます — キーがある場合の Claude API 呼び出し例
# ---------------------------------------------------------------------------
# import anthropic                              # pip install anthropic
# client = anthropic.Anthropic()                # ANTHROPIC_API_KEY 環境変数を使用
# response = client.messages.create(
#     model="claude-opus-5",
#     max_tokens=1024,
#     messages=[{"role": "user", "content": "議事録を3行で要約して: ..."}],
# )
# print(response.content[0].text)
# ---------------------------------------------------------------------------
