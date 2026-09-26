"""
データ品質・重複排除・汚染チェックの実習。
- 文字 n-gram(シングル)集合とジャッカード類似度で、完全/近似重複の文書を検出・除去します。
- ベンチマークの問題が学習コーパスに混ざった「汚染(試験問題の流出)」を n-gram の重なりで検査し、
  汚染がベンチマークスコアをどれだけ水増しするかをシミュレーションで確認します。
"""
import random
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

DUP_THRESHOLD = 0.7    # この値以上なら近似重複と判定
NGRAM_N = 5            # 文字 n-gram の長さ
CONTAM_N = 10          # 汚染チェック用 n-gram の長さ (長いほど「丸ごと流出」だけを捕まえる)
CONTAM_THRESHOLD = 0.3 # 問題の n-gram のうちこの比率以上がコーパスにあれば流出の疑い


def char_ngrams(text: str, n: int) -> set:
    """文字列を長さ n の連続した断片(文字 n-gram)の集合に切る。"""
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard(a: set, b: set) -> float:
    """ジャッカード類似度 = |積集合| / |和集合|。完全に同じなら 1.0、無関係なら 0.0。"""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def contamination_ratio(question: str, corpus_grams: set, n: int) -> float:
    """問題の n-gram のうち、学習コーパスにそのまま存在する比率。"""
    q_grams = char_ngrams(question, n)
    if not q_grams:
        return 0.0
    return sum(1 for g in q_grams if g in corpus_grams) / len(q_grams)


def build_documents(rng: random.Random):
    """tiny_corpus の文で文書24件を作り、重複/流出の事例をわざと仕込む。"""
    sentences = [s.strip() + "。" for s in hjh_data.tiny_corpus().split("。 ") if s.strip()]
    docs = []
    for i in range(24):
        picked = rng.sample(sentences, 3)          # 文3つ = 文書1件
        docs.append(" ".join(picked))
    docs[15] = docs[2]                             # 完全な複製を仕込む
    docs[19] = docs[5].replace("確認した", "チェックした", 1)  # 一語だけ変えた近似複製
    return docs


BENCHMARK = [  # (問題, 正解) — 4択の一般常識テストだと想定
    ("日本の首都はどこか? 選択肢: 大阪, 東京, 名古屋, 福岡", "東京"),
    ("水が沸騰する摂氏の温度は何度か? 選択肢: 50, 80, 100, 120", "100"),
    ("一週間は何日か? 選択肢: 5日, 6日, 7日, 8日", "7日"),
    ("三角形の内角の和は何度か? 選択肢: 90, 180, 270, 360", "180"),
    ("光と音のうち速いのはどちらか? 選択肢: 光, 音, 同じ, 分からない", "光"),
    ("1年はおよそ何日か? 選択肢: 300日, 330日, 365日, 400日", "365日"),
]
LEAKED_IDX = 3  # この問題を学習文書に流出させる


def memorizer_score(corpus_text: str, items, rng: random.Random):
    """「暗記したものだけ当てる」仮想モデル: 問題がコーパスに丸ごとあれば正解、
    なければ4択の当てずっぽう(正答率25%)。戻り値: 問題ごとの正誤リスト。"""
    results = []
    for question, _answer in items:
        seen = question in corpus_text          # 過去問集で見た問題か?
        correct = True if seen else (rng.random() < 0.25)
        results.append((seen, correct))
    return results


def main():
    rng = random.Random(42)  # 再現性のためのシード固定

    # [1] コーパスの構成 ---------------------------------------------------
    docs = build_documents(rng)
    q_leak, a_leak = BENCHMARK[LEAKED_IDX]
    docs[9] = docs[9] + f" 今日の常識クイズ。{q_leak} 正解は {a_leak}。" # 流出文書
    print("[1] 学習コーパスの構成: 文書", len(docs), "件")
    print("    - 仕込んだ問題: 完全な複製(2↔15)、近似の複製(5↔19)、ベンチマーク流出(文書9)")
    print("    - 文書の例:", docs[0][:44], "...")

    # [2] 近似重複の検出 ----------------------------------------------------
    grams = [char_ngrams(d, NGRAM_N) for d in docs]
    dup_pairs = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            sim = jaccard(grams[i], grams[j])
            if sim >= DUP_THRESHOLD:
                dup_pairs.append((i, j, sim))
    print(f"\n[2] 重複の検出 (文字 {NGRAM_N}-gram のジャッカード >= {DUP_THRESHOLD})")
    for i, j, sim in dup_pairs:
        kind = "完全な重複" if sim > 0.999 else "近似の重複"
        print(f"    - 文書 {i:2d} ↔ 文書 {j:2d} : ジャッカード {sim:.3f}  → {kind}")
    if not dup_pairs:
        print("    - 重複は見つかりませんでした")

    # [3] 重複の除去 --------------------------------------------------------
    drop = {j for _i, j, _s in dup_pairs}       # 後から出てきた方を捨てる
    kept = [d for k, d in enumerate(docs) if k not in drop]
    print(f"\n[3] 重複の除去: {len(docs)}件 → {len(kept)}件 (捨てた文書: {sorted(drop)})")
    print("    無駄になりかけた学習トークンを節約し、特定文書を暗記するリスクも減りました。")

    # [4] ベンチマーク汚染のチェック ----------------------------------------
    corpus_text = " ".join(kept)
    corpus_grams = char_ngrams(corpus_text, CONTAM_N)
    print(f"\n[4] 汚染チェック (問題の {CONTAM_N}-gram がコーパスに存在する比率 >= {CONTAM_THRESHOLD:.0%})")
    contaminated = []
    for idx, (question, _a) in enumerate(BENCHMARK):
        ratio = contamination_ratio(question, corpus_grams, CONTAM_N)
        flag = ratio >= CONTAM_THRESHOLD
        if flag:
            contaminated.append(idx)
        mark = "★流出の疑い" if flag else "クリーン"
        print(f"    - 問題 {idx}: 重なり {ratio:5.1%}  [{mark}]  {question[:26]}...")

    # [5] 汚染がスコアを水増しする実演 --------------------------------------
    print("\n[5] 「暗記したものだけ当てる」仮想モデルのベンチマークスコア比較")
    results = memorizer_score(corpus_text, BENCHMARK, random.Random(6))  # 当てずっぽう専用シード
    total = len(BENCHMARK)
    score_all = sum(c for _s, c in results) / total
    clean_items = [r for k, r in enumerate(results) if k not in contaminated]
    score_clean = sum(c for _s, c in clean_items) / max(1, len(clean_items))
    for k, (seen, correct) in enumerate(results):
        note = "過去問で見た → 自動的に正解" if seen else ("当てずっぽうで正解" if correct else "不正解")
        print(f"    - 問題 {k}: {'O' if correct else 'X'}  ({note})")
    print(f"    汚染込みのスコア : {score_all:.1%}  ← 発表資料に載りやすい数字")
    print(f"    クリーンスコア   : {score_clean:.1%}  ← 実力に近い数字")
    print("    → 流出した1問がスコアを水増しします。「このスコアは汚染チェック済みですか?」と聞いてみましょう。")


if __name__ == "__main__":
    main()
