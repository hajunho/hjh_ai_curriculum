"""
BPE(Byte Pair Encoding) 토크나이저를 밑바닥부터 구현합니다.
1) tiny_corpus 에서 "가장 자주 붙어 나오는 글자쌍"을 반복 병합해 vocab 을 만들고
2) 병합 횟수(vocab 크기)에 따라 문장이 몇 토큰이 되는지 비교하며
3) 숫자 분리(digit split)가 있고 없을 때 숫자 토큰화가 어떻게 달라져
   수학 능력에 영향을 주는지 확인합니다.
"""

import sys
import pathlib
import random
from collections import Counter

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

END = "</w>"  # 단어 끝 표시 — "학생"과 "학생이"를 구분하게 해 줍니다.


def word_freqs(text: str) -> Counter:
    """공백 기준으로 단어를 세어 {글자 튜플: 빈도} 사전을 만듭니다."""
    freqs = Counter()
    for word in text.split():
        freqs[tuple(list(word) + [END])] += 1
    return freqs


def count_pairs(freqs: Counter, digit_split: bool) -> Counter:
    """이웃한 심볼 쌍의 등장 빈도를 셉니다. digit_split=True 면 숫자가
    포함된 심볼은 병합 후보에서 빼서, 숫자를 항상 한 자리씩 유지합니다."""
    pairs = Counter()
    for symbols, f in freqs.items():
        for a, b in zip(symbols, symbols[1:]):
            if digit_split and (any(c.isdigit() for c in a) or
                                any(c.isdigit() for c in b)):
                continue
            pairs[(a, b)] += f
    return pairs


def merge_pair(freqs: Counter, pair) -> Counter:
    """모든 단어에서 해당 글자쌍을 하나의 심볼로 붙입니다."""
    a, b = pair
    new_freqs = Counter()
    for symbols, f in freqs.items():
        merged, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        new_freqs[tuple(merged)] += f
    return new_freqs


def train_bpe(text: str, num_merges: int, digit_split: bool = False,
              verbose: bool = False):
    """BPE 학습: 가장 잦은 쌍을 num_merges 번 병합하고 병합 규칙을 반환합니다."""
    freqs = word_freqs(text)
    merges = []
    for step in range(num_merges):
        pairs = count_pairs(freqs, digit_split)
        if not pairs:            # 더 병합할 쌍이 없으면 조기 종료
            break
        best, best_n = pairs.most_common(1)[0]
        freqs = merge_pair(freqs, best)
        merges.append(best)
        if verbose and (step < 8 or (step + 1) % 20 == 0):
            print(f"    병합 {step + 1:3d}: '{best[0]}' + '{best[1]}'"
                  f" -> '{best[0] + best[1]}'  ({best_n}회 등장)")
    return merges


def encode(text: str, merges) -> list:
    """학습된 병합 규칙을 순서대로 적용해 새 문장을 토큰화합니다."""
    tokens = []
    for word in text.split():
        symbols = list(word) + [END]
        for pair in merges:  # 학습 때 배운 순서대로 적용 (우선순위)
            symbols = list(next(iter(
                merge_pair(Counter({tuple(symbols): 1}), pair))))
        tokens.extend(symbols)
    return [t.replace(END, "") for t in tokens if t != END]


def make_price_corpus(n: int = 300, seed: int = 42) -> str:
    """숫자 분리 실험용 매출 보고 문장 코퍼스 (끝자리 0이 많은 금액)."""
    rng = random.Random(seed)
    lines = [f"매출 {rng.randrange(1, 999) * 100}원 기록"
             for _ in range(n)]
    return " ".join(lines)


if __name__ == "__main__":
    corpus = hjh_data.tiny_corpus()
    print(f"[1] 코퍼스 로딩: {len(corpus):,}자, 고유 글자 "
          f"{len(set(corpus)):,}종 (글자 단위에서 출발)")

    # ---- BPE 병합 과정 관찰 ----
    print("\n[2] BPE 학습 — 가장 자주 붙는 글자쌍부터 차례로 병합합니다.")
    merges = train_bpe(corpus, num_merges=60, verbose=True)
    print(f"    총 {len(merges)}회 병합 완료 (요청 60회, 쌍이 소진되면 조기 종료)")

    # ---- vocab 크기 트레이드오프 ----
    sample = "어제 학생이 보고서를 만들었다."
    print(f"\n[3] vocab 크기 트레이드오프 — 같은 문장이 몇 토큰이 되나")
    print(f"    문장: \"{sample}\"")
    base_vocab = len(set(corpus)) + 1  # 글자 종수 + </w>
    for n in [0, 10, 30, 60]:
        m = train_bpe(corpus, num_merges=n)
        toks = encode(sample, m)
        print(f"    병합 {n:3d}회 (vocab~{base_vocab + len(m):3d}):"
              f" {len(toks):2d}토큰 -> {toks}")
    print("    -> vocab 이 클수록 문장은 짧아지지만(추론 비용 절감),")
    print("       임베딩 표가 커지고 희귀 토큰은 학습 기회가 줄어듭니다.")

    # ---- 숫자 분리(digit split) 실험 ----
    print("\n[4] 숫자 분리 실험 — 금액이 많은 코퍼스로 BPE 를 두 번 학습")
    price_corpus = make_price_corpus()
    m_free = train_bpe(price_corpus, num_merges=40, digit_split=False)
    m_split = train_bpe(price_corpus, num_merges=40, digit_split=True)
    digit_merges = [a + b for a, b in m_free
                    if any(c.isdigit() for c in a + b)]
    print(f"    분리 없음: 숫자가 낀 병합 {len(digit_merges)}개 생김"
          f" 예: {digit_merges[:6]}")

    for test in ["98700원", "12500원"]:
        t_free = encode(test, m_free)
        t_split = encode(test, m_split)
        print(f"    \"{test}\"  분리 없음: {t_free}")
        print(f"    {'':>9}  분리 있음: {t_split}")

    print("\n[5] 왜 수학에 중요한가")
    print("    분리 없음: '00', '500' 같은 덩어리 토큰이 생겨 987+13 같은")
    print("      자리올림 계산을 '자릿수' 단위로 볼 수 없게 됩니다.")
    print("    분리 있음: 모든 수가 한 자리씩 일관되게 쪼개져, 모델이")
    print("      자릿수 규칙(받아올림 등)을 배울 기회가 생깁니다.")
    print("    실제 최신 LLM 들이 숫자를 1~3자리 단위로 강제 분리하는 이유입니다.")
