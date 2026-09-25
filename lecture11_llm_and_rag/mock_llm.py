"""
lecture11 공용 모의(mock) LLM / 임베딩 모듈.

API 키·인터넷 없이 LLM 수업을 진행하기 위한 규칙 기반 구현입니다.
- MockLLM       : 패턴 매칭으로 요약·분류·초안 작성 등을 흉내 내는 언어모델.
                  검색된 문맥(context)을 받으면 그 문장을 재조립해 답합니다.
- MockEmbedding : 문자 n-gram 해시 + 단어 동시출현(co-occurrence) 확장 벡터.
                  같은 주제의 문장이 실제로 코사인 유사도가 높게 나옵니다.
실제 서비스에서는 이 자리에 Claude API 등 진짜 모델이 들어갑니다.
"""

import re
import zlib

import numpy as np


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """마침표/물음표 기준의 아주 단순한 한국어 문장 분리기."""
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def tokenize(text: str) -> list[str]:
    """공백 기준 토큰화 + 구두점 제거. (형태소 분석기 없이 최대한 단순하게)"""
    return [w.strip(".,?!():;\"'") for w in text.split() if w.strip(".,?!():;\"'")]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """코사인 유사도. 벡터가 이미 정규화돼 있으면 내적과 같습니다."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# MockEmbedding — 문자 n-gram 해시 + 동시출현 확장
# ---------------------------------------------------------------------------

class MockEmbedding:
    """
    텍스트 -> 고정 길이 벡터.

    원리:
    1) 문자 2~3그램(n-gram)을 뽑아 CRC32 해시로 dim개의 칸 중 하나에 카운트를
       더합니다. "연차휴가"와 "연차 휴가"처럼 표기가 조금 달라도 n-gram 이
       겹치므로 벡터가 가까워집니다.
    2) fit()으로 코퍼스를 학습하면 단어별 동시출현 이웃을 기억해 두었다가,
       임베딩할 때 이웃 단어의 n-gram 도 약하게 섞어 줍니다.
       ("연차" 문장과 "휴가" 문장이 같은 문서에 자주 함께 나오면 서로 가까워짐)
    """

    def __init__(self, dim: int = 512, ngram_range: tuple = (2, 3),
                 neighbor_weight: float = 0.35, max_neighbors: int = 4):
        self.dim = dim
        self.ngram_range = ngram_range
        self.neighbor_weight = neighbor_weight
        self.max_neighbors = max_neighbors
        self._cooc: dict[str, list[str]] = {}   # 단어 -> 동시출현 이웃 단어들

    # -- 내부: 문자 n-gram 추출 ---------------------------------------------
    def _char_ngrams(self, text: str) -> list[str]:
        cleaned = re.sub(r"[^0-9A-Za-z가-힣 ]", "", text).replace(" ", "_")
        grams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            grams.extend(cleaned[i:i + n] for i in range(len(cleaned) - n + 1))
        return grams

    def _add_ngrams(self, vec: np.ndarray, text: str, weight: float) -> None:
        for g in self._char_ngrams(text):
            idx = zlib.crc32(g.encode("utf-8")) % self.dim   # 안정적 해시
            vec[idx] += weight

    # -- 학습: 동시출현 통계 -------------------------------------------------
    def fit(self, texts: list[str]) -> "MockEmbedding":
        """문장 리스트에서 '한 문장에 함께 나온 단어' 통계를 만듭니다."""
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

    # -- 임베딩 --------------------------------------------------------------
    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        self._add_ngrams(vec, text, 1.0)
        # 동시출현 이웃 단어를 약하게 섞기 ("연차" 질문 -> "휴가" 문서와도 가까워짐)
        for w in tokenize(text):
            for nb in self._cooc.get(w, []):
                self._add_ngrams(vec, nb, self.neighbor_weight)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.embed(t) for t in texts])


# ---------------------------------------------------------------------------
# MockLLM — 규칙 기반 모의 언어모델
# ---------------------------------------------------------------------------

# 민원/문의 분류용 키워드 사전
_CATEGORY_LEXICON = {
    "배송": ["배송", "택배", "지연", "도착", "출고", "운송장"],
    "환불/결제": ["환불", "결제", "카드", "청구", "취소", "이중"],
    "품질/고장": ["고장", "불량", "파손", "소음", "작동", "수리"],
    "응대/서비스": ["직원", "상담", "불친절", "응대", "연결", "대기"],
}


class MockLLM:
    """
    규칙 기반 모의 LLM.

    complete(prompt)              : 프롬프트의 지시어(요약/분류/이메일...)를 보고
                                    패턴 응답을 만듭니다. 역할·형식·예시가 명시된
                                    "좋은 프롬프트"일수록 구조화된 답을 냅니다.
    answer_with_context(q, ctxs)  : RAG용. 검색된 문맥에서 질문과 겹치는 문장을
                                    골라 근거를 인용한 답변을 조립합니다.
    """

    def __init__(self, name: str = "mock-llm-v1"):
        self.name = name

    # -- 프롬프트에서 본문(payload) 분리 -------------------------------------
    @staticmethod
    def _payload(prompt: str) -> str:
        for marker in ("본문:", "내용:", "---"):
            if marker in prompt:
                return prompt.split(marker, 1)[1].strip()
        return prompt.strip()

    # -- 프롬프트 품질 신호 감지 ---------------------------------------------
    @staticmethod
    def _quality_flags(prompt: str) -> dict[str, bool]:
        return {
            "role": ("당신은" in prompt or "역할" in prompt),
            "format": any(k in prompt for k in ("형식", "개조식", "JSON", "번호를 붙여", "글머리")),
            "example": "예시" in prompt,
            "context": any(k in prompt for k in ("상황", "대상 독자", "맥락")),
        }

    # -- 하위 작업들 ----------------------------------------------------------
    def _summarize(self, payload: str, n: int, structured: bool) -> str:
        """추출 요약: 자주 나오는 단어를 많이 포함한 문장을 고릅니다."""
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
        return top[0][2] if top else "(요약할 내용이 없습니다)"

    def _classify(self, payload: str) -> tuple[str, list[str]]:
        best, best_hits = "기타", []
        for cat, words in _CATEGORY_LEXICON.items():
            hits = [w for w in words if w in payload]
            if len(hits) > len(best_hits):
                best, best_hits = cat, hits
        return best, best_hits

    def _draft_email(self, prompt: str, structured: bool) -> str:
        def _field(key, default):
            m = re.search(key + r"\s*[:：]\s*(.+)", prompt)
            return m.group(1).strip() if m else default
        to = _field("받는 사람", "관계자")
        purpose = _field("목적", "업무 협조 요청")
        deadline = _field("기한", "")
        body = [f"{to}께,", "", f"안녕하세요. {purpose} 관련하여 연락드립니다."]
        if deadline:
            body.append(f"가능하시다면 {deadline}까지 회신 부탁드립니다.")
        body += ["검토에 필요한 자료가 있으면 말씀해 주세요.", "", "감사합니다.", "드림"]
        if not structured:
            return f"{purpose}에 대해 메일을 보내면 될 것 같습니다. 안녕하세요로 시작해서 감사합니다로 끝내세요."
        return "\n".join(body)

    # -- 공개 API -------------------------------------------------------------
    def complete(self, prompt: str) -> str:
        """지시어를 보고 작업을 라우팅하는 모의 응답기."""
        flags = self._quality_flags(prompt)
        structured = flags["format"] or flags["example"]
        payload = self._payload(prompt)

        if "분류" in prompt:
            cat, hits = self._classify(payload)
            if structured:
                return f"분류: {cat}\n근거 키워드: {', '.join(hits) if hits else '없음'}"
            return f"음, 이건 {cat} 관련 내용인 것 같습니다."
        if "요약" in prompt:
            n = 3 if structured else 1
            return self._summarize(payload, n, structured)
        if "이메일" in prompt or "메일" in prompt:
            return self._draft_email(prompt, structured or flags["role"])
        if "번역" in prompt:
            return "(모의 번역) Hello, this is a mock translation of the given text."
        return "요청을 이해했습니다. 더 구체적인 지시(역할·형식·예시)를 주시면 정확도가 올라갑니다."

    def answer_with_context(self, question: str, contexts: list[tuple[str, str]],
                            max_evidence: int = 2) -> str:
        """
        RAG 답변 조립기. contexts = [(출처, 본문), ...]
        질문과 단어가 겹치는 문장을 근거로 골라 인용합니다. 근거가 없으면
        모른다고 답합니다(환각 억제의 최소 형태).
        """
        q_words = {w for w in tokenize(question) if len(w) >= 2}
        # 조사 차이를 흡수하기 위해 앞 2글자 어간도 함께 봅니다.
        q_stems = {w[:2] for w in q_words}
        evidence = []
        for source, text in contexts:
            for sent in split_sentences(text):
                s_words = tokenize(sent)
                overlap = sum(1 for w in s_words
                              if w in q_words or (len(w) >= 2 and w[:2] in q_stems))
                if overlap > 0:
                    evidence.append((overlap, source, sent))
        if not evidence:
            return "제공된 문서에서 근거를 찾지 못했습니다. 이 질문에는 답할 수 없습니다."
        evidence.sort(key=lambda x: -x[0])
        picked = evidence[:max_evidence]
        lines = ["문서에 근거해 답변드립니다."]
        for _, source, sent in picked:
            lines.append(f"- {sent} (출처: {source})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# [참고] 실제 API 라면 이렇게 — 키가 있을 때의 Claude API 호출 예시
# ---------------------------------------------------------------------------
# import anthropic                              # pip install anthropic
# client = anthropic.Anthropic()                # ANTHROPIC_API_KEY 환경변수 사용
# response = client.messages.create(
#     model="claude-opus-5",
#     max_tokens=1024,
#     messages=[{"role": "user", "content": "회의록을 3줄로 요약해 줘: ..."}],
# )
# print(response.content[0].text)
# ---------------------------------------------------------------------------
