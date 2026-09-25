"""
level03 — 토큰화와 한국어의 특수성

같은 문장을 공백 / 음절 / n-gram / 미니 규칙 기반 형태소 분석의
4가지 전략으로 토큰화해 비교하고, 어휘 크기 실험과
미니 BPE(Byte Pair Encoding) 학습까지 직접 구현합니다.
NLP 전용 패키지 없이 표준 라이브러리로만 동작합니다.
"""

import re
import sys
import pathlib
from collections import Counter

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

# ---------------------------------------------------------------------------
# 토큰화 전략 4종
# ---------------------------------------------------------------------------

def tokenize_space(text: str) -> list[str]:
    """전략1: 공백 분리 — 영어의 기본, 한국어에서는 조사가 붙은 채로 나옵니다."""
    return text.split()


def tokenize_syllable(text: str) -> list[str]:
    """전략2: 음절(문자) 분리 — 미등록 단어에 강하지만 의미가 옅어집니다."""
    return [ch for ch in text if not ch.isspace()]


def tokenize_ngram(text: str, n: int = 2) -> list[str]:
    """전략3: 음절 n-gram — 어절 안에서 연속 n 음절을 겹쳐 가며 묶습니다."""
    grams = []
    for word in text.split():
        if len(word) < n:
            grams.append(word)
        else:
            grams.extend(word[i:i + n] for i in range(len(word) - n + 1))
    return grams


# --- 전략4: 미니 규칙 기반 형태소 분석기 ------------------------------------
# 실제 분석기의 뼈대(사전 + 규칙 + 가장 긴 일치 우선)를 축소 구현한 교육용입니다.

NOUNS = ["배송", "포장", "가격", "품질", "직원", "매장", "사진", "성능",
         "설치", "재구매", "분위기", "의사", "생각", "문의", "색상", "설명서",
         "답", "채", "번"]
STEMS = ["빠르", "빨라", "느리", "좋", "좋았", "훌륭하", "꼼꼼하", "만족하",
         "튼튼하", "친절하", "쾌적했", "놀랐", "간단해", "편했", "떨어져",
         "망가졌", "다르", "달라", "실망했", "헤맸", "걸렸", "찢어진", "약해"]
JOSA = ["이나", "에서", "보다", "이랑", "이", "가", "은", "는", "을", "를",
        "도", "에", "의", "로", "과", "와"]
EOMI = ["습니다", "았습니다", "었습니다", "네요", "어요", "아요", "여서",
        "아서", "어서", "하고", "지만", "다", "요", "고"]


def analyze_word(word: str) -> list[str]:
    """어절 하나를 [명사+조사] 또는 [어간+어미]로 해체 (가장 긴 일치 우선)."""
    for noun in sorted(NOUNS, key=len, reverse=True):        # 긴 명사부터
        if word == noun:
            return [noun]
        if word.startswith(noun):
            rest = word[len(noun):]
            for josa in sorted(JOSA, key=len, reverse=True):
                if rest == josa:
                    return [noun, rest + "(조사)"]
    for stem in sorted(STEMS, key=len, reverse=True):        # 긴 어간부터
        if word.startswith(stem):
            rest = word[len(stem):]
            if rest == "":
                return [stem + "-"]
            for eomi in sorted(EOMI, key=len, reverse=True):
                if rest == eomi:
                    return [stem + "-", rest + "(어미)"]
    return [word]                                            # 미등록어는 그대로


def tokenize_morph(text: str) -> list[str]:
    """전략4: 미니 형태소 분석 — 어절마다 analyze_word 적용."""
    tokens = []
    for word in text.split():
        tokens.extend(analyze_word(word))
    return tokens


# ---------------------------------------------------------------------------
# 미니 BPE — 자주 붙는 글자 쌍을 반복 병합
# ---------------------------------------------------------------------------

def bpe_train(corpus_words: list[str], n_merges: int) -> list[tuple[str, str]]:
    """말뭉치에서 병합 규칙을 학습합니다. 반환: [(왼쪽조각, 오른쪽조각), ...]"""
    # 각 단어를 음절 리스트로 초기화
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
        if freq < 2:                       # 2회 미만이면 병합 가치 없음
            break
        merges.append((a, b))
        merged = a + b
        for w in words:                    # 모든 단어에 병합 적용
            i = 0
            while i < len(w) - 1:
                if w[i] == a and w[i + 1] == b:
                    w[i:i + 2] = [merged]
                else:
                    i += 1
    return merges


def bpe_encode(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """학습된 병합 규칙을 순서대로 적용해 단어를 토큰화합니다."""
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
# 시연
# ---------------------------------------------------------------------------

def demo_compare(sentence: str) -> None:
    print("[1] 같은 문장, 4가지 토큰화 전략")
    print(f"    문장: {sentence!r}\n")
    strategies = [
        ("공백 분리", tokenize_space(sentence)),
        ("음절 분리", tokenize_syllable(sentence)),
        ("2-gram", tokenize_ngram(sentence, 2)),
        ("미니 형태소", tokenize_morph(sentence)),
    ]
    for name, tokens in strategies:
        print(f"    {name:8s} ({len(tokens):2d}개): {tokens}")
    print()


def demo_morph_inside() -> None:
    print("[2] 미니 형태소 분석기 내부 — 어절을 '가장 긴 일치'로 해체")
    for word in ["배송이", "포장은", "빨라서", "좋았습니다", "친절하네요", "신제품이"]:
        print(f"    {word:8s} -> {analyze_word(word)}")
    print("    -> '신제품이'처럼 사전에 없는 말은 통째로 남습니다 (미등록어 문제).\n")


def demo_vocab_size() -> None:
    print("[3] 어휘 크기 실험 — 리뷰 60건을 전략별로 토큰화하면")
    reviews = [r["text"] for r in hjh_data.review_corpus(60, seed=3)]
    strategies = {
        "공백 분리": tokenize_space,
        "음절 분리": tokenize_syllable,
        "2-gram": tokenize_ngram,
        "미니 형태소": tokenize_morph,
    }
    for name, fn in strategies.items():
        vocab = Counter()
        for r in reviews:
            vocab.update(fn(r))
        delivery = sorted(t for t in vocab if t.startswith("배송"))
        print(f"    {name:8s}: 어휘 {len(vocab):4d}종 | '배송' 계열 토큰: {delivery}")
    print("    -> 형태소 방식은 '배송이/배송을'을 '배송'+조사로 모아 어휘를 압축합니다.\n")


def demo_bpe() -> None:
    print("[4] 미니 BPE 학습 — 자주 붙는 글자 쌍을 병합해 토큰을 '발명'하기")
    reviews = [r["text"] for r in hjh_data.review_corpus(200, seed=3)]
    corpus_words = [w for r in reviews for w in r.split()]
    merges = bpe_train(corpus_words, n_merges=40)
    print(f"    병합 규칙 {len(merges)}개 학습. 처음 10개:")
    for i, (a, b) in enumerate(merges[:10], start=1):
        print(f"      {i:2d}. '{a}' + '{b}' -> '{a + b}'")
    print("\n    학습된 규칙으로 단어 토큰화:")
    for word in ["배송이", "품질이", "만족합니다", "초고속배송"]:
        note = "  <- 처음 보는 단어도 조각으로 표현!" if word == "초고속배송" else ""
        print(f"      {word:8s} -> {bpe_encode(word, merges)}{note}")
    print("    -> 자주 나온 '배송이'는 통째로 한 토큰, 처음 보는 '초고속배송'은")
    print("       학습된 조각('배송' 포함)의 조합으로 표현됩니다.")
    print("       GPT 계열 토크나이저(BPE)의 원리이며, 한국어 데이터가 적으면")
    print("       병합 규칙이 덜 만들어져 토큰 수(=API 요금)가 늘어납니다.")


if __name__ == "__main__":
    print("=" * 70)
    print("토큰화와 한국어의 특수성 — 쪼개는 단위가 품질을 결정한다")
    print("=" * 70 + "\n")
    demo_compare("배송이 빨라서 좋았습니다")
    demo_morph_inside()
    demo_vocab_size()
    demo_bpe()
