"""
level01 — 텍스트 전처리 기초

지저분한 한국어 리뷰(이모티콘, URL, 반복 문자, 잡공백, 조사)를
단계별 파이프라인으로 정제하는 과정을 시연합니다.
전처리 전후의 단어 빈도 집계를 비교해 '왜 전처리가 필수인지' 확인합니다.
"""

import random
import re
import sys
import pathlib
from collections import Counter

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

# ---------------------------------------------------------------------------
# [1] 오염된 리뷰 만들기 — 현실의 텍스트는 깨끗하지 않습니다
# ---------------------------------------------------------------------------

NOISE_PREFIX = ["★★★ ", "[포토리뷰] ", "", "♡♡ "]
NOISE_SUFFIX = [" ㅋㅋㅋㅋㅋ", " ㅠㅠㅠㅠ", "!!!!!", " 좋아요오오오",
                " 참고: http://blog.example.com/review123", ""]


def make_dirty_reviews(n: int = 8, seed: int = 42) -> list[str]:
    """hjh_data 리뷰에 일부러 잡음을 입혀 '현실적인' 리뷰를 만듭니다."""
    rng = random.Random(seed)
    base = hjh_data.review_corpus(n * 3, seed=3)[:n]
    dirty = []
    for row in base:
        text = (rng.choice(NOISE_PREFIX) + row["text"].replace(" ", "  ", 1)
                + rng.choice(NOISE_SUFFIX))
        if rng.random() < 0.5:
            text = "  " + text + "   "          # 앞뒤 잡공백
        if rng.random() < 0.4:
            text = text.replace("습니다", "습니다요 THANKS")  # 영문 대문자 섞기
        dirty.append(text)
    return dirty


# ---------------------------------------------------------------------------
# [2] 파이프라인 단계 정의 — 문자열 in, 문자열 out 인 작은 함수들
# ---------------------------------------------------------------------------

STOPWORDS = {"그리고", "그런데", "다만", "참고"}
# 어절 끝에서 떼어 볼 간이 조사 목록 (완전하지 않음 — 형태소 분석은 level03)
JOSA = ["이라", "에서", "부터", "까지", "이", "가", "은", "는", "을", "를",
        "도", "에", "의", "로", "와", "과", "요"]


def lowercase_and_strip_url(text: str) -> str:
    """소문자화 + URL 제거. URL 을 먼저 지워야 뒤 단계가 편해집니다."""
    text = text.lower()
    return re.sub(r"https?://\S+", " ", text)


def remove_special(text: str) -> str:
    """한글/영문/숫자/공백만 남기고 제거 (★, ♡, !, [] 등)."""
    return re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-z0-9\s]", " ", text)


def collapse_repeats(text: str) -> str:
    """같은 글자 3회 이상 반복을 2회로 축약: 좋아요오오오 -> 좋아요오, ㅋㅋㅋㅋ -> ㅋㅋ"""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def normalize_space(text: str) -> str:
    """연속 공백을 하나로, 앞뒤 공백 제거."""
    return re.sub(r"\s+", " ", text).strip()


def drop_stopwords_and_josa(text: str) -> str:
    """불용어 제거 + 어절 끝 간이 조사 떼기 (2글자 이상 남을 때만)."""
    words = []
    for word in text.split():
        if word in STOPWORDS:
            continue
        for josa in JOSA:                      # 긴 조사부터 시도
            if word.endswith(josa) and len(word) - len(josa) >= 2:
                word = word[: -len(josa)]
                break
        words.append(word)
    return " ".join(words)


PIPELINE = [
    ("소문자화·URL 제거", lowercase_and_strip_url),
    ("특수문자 정리", remove_special),
    ("반복 문자 축약", collapse_repeats),
    ("공백 정규화", normalize_space),
    ("불용어·간이 조사 제거", drop_stopwords_and_josa),
]


def clean(text: str) -> str:
    """파이프라인 전체를 순서대로 통과시킵니다."""
    for _, step in PIPELINE:
        text = step(text)
    return text


# ---------------------------------------------------------------------------
# 시연
# ---------------------------------------------------------------------------

def demo_step_by_step(sample: str) -> None:
    print("[2] 한 건을 단계별로 통과시키기")
    print(f"    원본: {sample!r}")
    text = sample
    for i, (name, step) in enumerate(PIPELINE, start=1):
        text = step(text)
        print(f"    {i}단계 {name:14s} -> {text!r}")
    print()


def word_freq(texts: list[str], top: int = 8) -> list[tuple[str, int]]:
    counter = Counter()
    for t in texts:
        counter.update(w for w in t.split() if len(w) >= 2)
    return counter.most_common(top)


if __name__ == "__main__":
    print("=" * 70)
    print("텍스트 전처리 — 지저분한 리뷰를 파이프라인으로 정제하기")
    print("=" * 70 + "\n")

    dirty = make_dirty_reviews(n=8, seed=42)
    print(f"[1] 오염된 리뷰 {len(dirty)}건 생성 (이모티콘·URL·반복문자·잡공백 포함)")
    for d in dirty[:3]:
        print(f"    예시: {d!r}")
    print()

    demo_step_by_step(dirty[0])

    print("[3] 전체 일괄 정제 (원본 -> 결과)")
    cleaned = [clean(d) for d in dirty]
    for before, after in zip(dirty, cleaned):
        print(f"    {before.strip()[:34]:36s} -> {after}")
    print()

    print("[4] 전처리 전후 단어 빈도 top8 비교")
    before_freq = word_freq(dirty)
    after_freq = word_freq(cleaned)
    print("    전처리 전                | 전처리 후")
    print("    " + "-" * 50)
    for (bw, bc), (aw, ac) in zip(before_freq, after_freq):
        print(f"    {bw:14s} {bc}회      | {aw:12s} {ac}회")
    print()
    print("    -> 전처리 전에는 '배송이/배송은'이 따로 세어지고 잡음 어절이 섞이지만,")
    print("       전처리 후에는 같은 개념이 하나로 합쳐져 집계가 보고서에 쓸 만해집니다.")
    print("\n결론: 전처리는 '목적에 맞는 재료 손질'입니다. 다음 레벨에서 정규표현식을 정식으로 배웁니다.")
