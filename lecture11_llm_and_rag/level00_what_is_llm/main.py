"""
LLM(대규모 언어모델)의 원리를 '다음 단어 예측기'로 체험합니다.
1) 미니 한국어 코퍼스에서 n-gram 통계(단어가 이어질 확률)를 셉니다.
2) 특정 문맥 뒤에 올 단어의 확률 분포를 확인합니다.
3) 확률에 따라 단어를 하나씩 뽑아 문장을 '생성'해 봅니다.
실제 LLM은 이 원리를 수천억 파라미터의 신경망으로 확장한 것입니다.
"""

import pathlib
import random
import sys
from collections import Counter, defaultdict

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data


def build_ngram_model(words: list[str], n: int) -> dict:
    """앞 (n-1)개 단어(문맥) -> 다음 단어 등장 횟수 테이블을 만듭니다."""
    model = defaultdict(Counter)
    for i in range(len(words) - n + 1):
        context = tuple(words[i:i + n - 1])
        nxt = words[i + n - 1]
        model[context][nxt] += 1
    return model


def next_word_distribution(model: dict, context: tuple) -> list[tuple[str, float]]:
    """문맥 뒤에 올 단어들의 확률 분포를 계산합니다."""
    counter = model.get(context, Counter())
    total = sum(counter.values())
    if total == 0:
        return []
    return [(w, c / total) for w, c in counter.most_common()]


def generate(model: dict, start: tuple, rng: random.Random,
             max_words: int = 12, greedy: bool = False) -> str:
    """문맥에서 시작해 다음 단어를 반복해서 뽑아 문장을 만듭니다.
    greedy=True 면 항상 1등 단어만 선택(온도 0), False 면 확률대로 샘플링."""
    out = list(start)
    context = start
    for _ in range(max_words):
        dist = next_word_distribution(model, context)
        if not dist:
            break
        if greedy:
            word = dist[0][0]                      # 가장 확률 높은 단어
        else:
            words, probs = zip(*dist)
            word = rng.choices(words, weights=probs, k=1)[0]
        out.append(word)
        if word.endswith("."):                     # 마침표가 나오면 문장 종료
            break
        context = tuple(out[-(len(start)):])       # 문맥 창을 한 칸 밀기
    return " ".join(out)


def main() -> None:
    rng = random.Random(42)                        # 재현성을 위한 시드 고정

    print("=" * 62)
    print("Level 00 | LLM이란 — 초대형 '다음 단어 예측기'")
    print("=" * 62)

    # [1] 학습 데이터: 미니 한국어 코퍼스
    corpus = hjh_data.tiny_corpus()
    words = corpus.split()
    print(f"\n[1] 학습 코퍼스: {len(corpus):,}자, {len(words):,}단어")
    print(f"    예시 앞부분: {' '.join(words[:12])} ...")

    # [2] 통계 만들기 = '학습'. 파라미터가 많을수록 더 긴 문맥을 기억합니다.
    bigram = build_ngram_model(words, n=2)
    trigram = build_ngram_model(words, n=3)
    print(f"\n[2] 학습 완료 — bigram 문맥 {len(bigram):,}종, trigram 문맥 {len(trigram):,}종")
    print("    (실제 LLM 은 이 '표'가 아니라 수천억 개 파라미터의 신경망으로 기억)")

    # [3] 다음 단어 확률 분포 — LLM 이 매 토큰마다 하는 일
    print("\n[3] '회사원이' 다음에 올 단어의 확률 분포 (상위 5개)")
    for w, p in next_word_distribution(bigram, ("회사원이",))[:5]:
        bar = "#" * int(p * 40)
        print(f"    {w:<10} {p:6.1%} {bar}")

    print("\n    '어제 개발자가' 다음에 올 단어 (trigram, 상위 5개)")
    for w, p in next_word_distribution(trigram, ("어제", "개발자가"))[:5]:
        bar = "#" * int(p * 40)
        print(f"    {w:<12} {p:6.1%} {bar}")

    # [4] 생성 — 한 단어씩 뽑아 문장을 만든다 (토큰 단위 생성)
    print("\n[4] 문장 생성: 같은 시작이라도 샘플링하면 매번 다른 문장")
    for i in range(3):
        print(f"    샘플 {i + 1}: {generate(trigram, ('오늘', '요리사가'), rng)}")
    print(f"    탐욕적(항상 1등만): {generate(trigram, ('오늘', '요리사가'), rng, greedy=True)}")
    print("    -> ChatGPT/Claude 가 같은 질문에 다른 답을 주는 이유가 이 '샘플링'입니다.")

    # [5] 한계 체험 — 그럴듯하지만 코퍼스에 없던 조합 = 환각의 씨앗
    print("\n[5] 한계: 그럴듯한 거짓말(환각)의 원리")
    sentences = set(s.strip() for s in corpus.split(".") if s.strip())
    novel, sample_novel = 0, ""
    for i in range(20):
        g = generate(trigram, ("주말에", "학생이"), rng).rstrip(".")
        if g not in sentences:
            novel += 1
            sample_novel = g
    print(f"    생성 20문장 중 {novel}문장은 학습 데이터에 '없던' 새 조합입니다.")
    print(f"    예: \"{sample_novel}.\"")
    print("    문법은 자연스럽지만 사실 확인은 없습니다 — LLM 환각도 같은 원리입니다.")

    # [6] 지식 시점(knowledge cutoff)
    print("\n[6] 지식 시점: 이 모델은 코퍼스에 없는 단어를 전혀 모릅니다.")
    print(f"    '신입사원이' 다음 단어 분포: {next_word_distribution(bigram, ('신입사원이',)) or '없음(학습에 없던 단어)'}")
    print("    -> 실제 LLM 도 학습 종료 시점 이후의 일은 모릅니다. (그래서 RAG 가 필요)")

    print("\n요약: LLM = 방대한 텍스트로 '다음 토큰 확률'을 배운 초대형 자동완성.")
    print("      유창함과 사실성은 별개라는 것이 이번 강의 전체의 출발점입니다.")


if __name__ == "__main__":
    main()
