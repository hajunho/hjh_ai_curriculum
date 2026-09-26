"""
level00 — コンピュータが言語を扱う難しさ

単純な文字列マッチング (キーワード検索) でサポートボットを作ってみて、
同義語 / 曖昧性 / 文脈 (皮肉) / 「分かち書きのない日本語」という特性の前で
どのように崩れるかを 4 つの失敗事例で実演します。
標準ライブラリだけを使います。
"""


def keyword_match(text: str, keywords: list[str]) -> bool:
    """最も単純なアプローチ: キーワードが 1 つでも「部分文字列」として含まれれば True。"""
    return any(kw in text for kw in keywords)


# ---------------------------------------------------------------------------
# [1] 同義語問題 — 同じ意味、違う文字
# ---------------------------------------------------------------------------

REFUND_REQUESTS = [  # すべて「返金を求めている」お客様からの問い合わせです。
    "返金対応をお願いします",
    "お金を返してください",
    "決済をキャンセルしたいです",
    "購入キャンセル後の振り込みをお願いします",
    "返金はいつになりますか",
    "ペイバックはできますか",
]


def demo_synonym() -> None:
    print("[1] 同義語問題 — 「返金」のキーワード検索は何件見つけられるか?")
    keywords = ["返金"]
    hit = 0
    for text in REFUND_REQUESTS:
        found = keyword_match(text, keywords)
        hit += found
        mark = "検出  " if found else "見逃し ×"
        print(f"    {mark} | {text}")
    print(f"    -> 実際の返金依頼 {len(REFUND_REQUESTS)}件のうち {hit}件しか検出できず "
          f"(的中率 {hit / len(REFUND_REQUESTS):.0%})")
    print("    -> 「お金を返してください」と「返金」が同じ意味だと、文字列は知りません。\n")


# ---------------------------------------------------------------------------
# [2] 曖昧性問題 — 「あめ」の 2 つの意味
# ---------------------------------------------------------------------------

AMBIGUOUS_SENTENCES = [
    ("あめが激しく降って道が混んだ", "rain"),
    ("あめをなめたら喉が楽になった", "candy"),
    ("あめがやんだので傘を閉じた", "rain"),
    ("あめを配ったら子どもたちが喜んだ", "candy"),
    ("あめに濡れて服がびしょびしょだ", "rain"),
    ("いちご味のあめを買った", "candy"),
]

# 周辺の単語 (文脈) のヒント: 意味を分ける手がかりは「あめ」自体ではなく隣の単語です。
CONTEXT_HINTS = {
    "rain": ["降っ", "やん", "傘", "濡れ", "天気"],
    "candy": ["なめ", "味", "配っ", "甘", "口"],
}


def guess_by_context(sentence: str) -> str:
    """隣の単語のヒントで意味を推定する、超ミニ「文脈」分類器。"""
    for sense, hints in CONTEXT_HINTS.items():
        if any(h in sentence for h in hints):
            return sense
    return "?"


def demo_ambiguity() -> None:
    print("[2] 曖昧性問題 — 同じ文字「あめ」、コンピュータには完全に同一")
    correct = 0
    for sentence, answer in AMBIGUOUS_SENTENCES:
        naive = "あめ" in sentence          # 文字列マッチング: 全部「あめ発見」で終わり
        guess = guess_by_context(sentence)
        correct += guess == answer
        print(f"    文字列一致={'あめ発見' if naive else '-'} | "
              f"文脈推定={guess:5s} | 正解={answer:5s} | {sentence}")
    print(f"    -> 文字列マッチングは 2 つの意味を区別できませんが、「周辺の単語」を見れば "
          f"{correct}/{len(AMBIGUOUS_SENTENCES)} 区別に成功")
    print("    -> 「単語の意味は隣人が決める」 — level06 埋め込みの中核アイデアです。\n")


# ---------------------------------------------------------------------------
# [3] 文脈・皮肉問題 — ポジティブキーワードの裏切り
# ---------------------------------------------------------------------------

REVIEWS = [  # (レビュー, 実際の感情 1=ポジティブ 0=ネガティブ)
    ("品質が本当に良い", 1),
    ("配送も速くて最高です", 1),
    ("良いと聞いて買ったのに完全にがっかりしました", 0),
    ("最高だと言われたのに一日で故障", 0),
    ("値段の割に良いとは言えない", 0),
    ("梱包が丁寧で満足しています", 1),
]


def demo_sarcasm() -> None:
    print("[3] 文脈問題 — 「良い/最高」キーワードでポジティブ判定すると?")
    keywords = ["良い", "最高", "満足"]
    wrong = 0
    for text, label in REVIEWS:
        pred = 1 if keyword_match(text, keywords) else 0
        ok = pred == label
        wrong += not ok
        mark = "正解  " if ok else "不正解 ×"
        print(f"    {mark} | 予測={'ポジ' if pred else 'ネガ'} "
              f"実際={'ポジ' if label else 'ネガ'} | {text}")
    print(f"    -> {len(REVIEWS)}件中 {wrong}件を誤分類。"
          f"「良い」という文字と「良いという意味」は別物です。\n")


# ---------------------------------------------------------------------------
# [4] 分かち書き問題 — 日本語には単語の切れ目がない
# ---------------------------------------------------------------------------

DELIVERY_SENTENCES = [
    "配送が速くて驚きました",
    "配送は遅かったけれど梱包は良い",
    "配送を待っているところです",
    "配送も対応もすべて満足",
    "再配送をお願いします",      # 「配送」を含むが別の概念 (再配送)
    "無料配送で嬉しかったです",  # これも複合語
]


def demo_agglutinative() -> None:
    print("[4] 分かち書き問題 — 「配送」を探したいのに単語の境界がない")
    exact = [s for s in DELIVERY_SENTENCES if "配送" in s.split()]     # 空白区切りの完全一致
    substr = [s for s in DELIVERY_SENTENCES if "配送" in s]            # 部分文字列の包含
    print(f"    単語の完全一致(空白区切りで「配送」単独): {len(exact)}件検出 {exact}")
    print(f"    部分文字列の包含(「配送」 in s)        : {len(substr)}件検出")
    for s in substr:
        note = " <- 再配送/無料配送まで釣れてくる" if ("再配送" in s or "無料配送" in s) else ""
        print(f"        - {s}{note}")
    print("    -> 日本語には空白がないため、split() では文全体が 1 つの塊になり")
    print("       完全一致は 0 件。部分包含は別の単語まで巻き込みます。形態素解析 (level03) が必要な理由です。\n")


# ---------------------------------------------------------------------------
# [5] ロードマップ
# ---------------------------------------------------------------------------

def print_roadmap() -> None:
    print("[5] 今日出会った失敗は、この講義でこう解決されます")
    roadmap = [
        ("同義語・表記ゆれ", "level01 前処理、level06 埋め込み (意味が近ければ座標も近く)"),
        ("曖昧性 (文脈)", "level07 RNN、level08 アテンション (周辺の単語を見て意味を決める)"),
        ("皮肉・否定表現", "level05 分類モデル + level08 以降の文脈モデル"),
        ("分かち書き・活用", "level03 トークン化・形態素解析、サブワード (BPE)"),
        ("新語", "level03 サブワード (知らない単語も断片として処理)"),
    ]
    for problem, solution in roadmap:
        print(f"    {problem:12s} -> {solution}")
    print("\n結論: 言語は「文字」ではなく「使われ方」です。次のレベルから一つずつ攻略していきます。")


if __name__ == "__main__":
    print("=" * 70)
    print("コンピュータが言語を扱う難しさ — 単純な文字列マッチング崩壊実験")
    print("=" * 70 + "\n")
    demo_synonym()
    demo_ambiguity()
    demo_sarcasm()
    demo_agglutinative()
    print_roadmap()
