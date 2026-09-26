"""
level02 — ビジネスの質問 -> ML 問題定義書チェックリストエンジン

「解約を減らしたい」のような願いを 5 要素の定義書 (対象/単位/時点/指標/アクション)
として書くと、空欄・リーク (leakage) の危険・ML への適合性を自動点検してくれる
ツールを作ります。モデルを作らない唯一のレベルですが、実務では最もよく使う技術です。
"""

from dataclasses import dataclass, field


@dataclass
class Feature:
    """予測の材料 (特徴量) 1 つ。available_at: この値が分かるようになる時点。"""
    name: str
    available_at: str  # 「予測時点より前」または「結果確定後」


@dataclass
class MLSpec:
    """ML 問題定義書 — 会議室のホワイトボードに書く 5 要素 + ベースライン。"""
    business_goal: str = ""    # ビジネス目標 (願い)
    target: str = ""           # 1. 予測対象: 測定可能な定義
    unit: str = ""             # 2. 予測単位: 1 行が何を表すか
    timing: str = ""           # 3. 予測時点
    metric: str = ""           # 4. 評価指標
    action: str = ""           # 5. 予測後のアクション
    baseline: str = ""         # モデルなし・今のやり方の成績
    features: list[Feature] = field(default_factory=list)


def check_completeness(spec: MLSpec) -> list[str]:
    """5 要素 + ベースラインが埋まっているか点検。"""
    problems = []
    checks = [
        (spec.target, "予測対象 (target) が空です。『解約』ではなく『次の 30日以内に解約 (1/0)』のように測定可能に。"),
        (spec.unit, "予測単位 (unit) が空です。1 行が顧客なのかアカウントなのか注文なのかを決めましょう。"),
        (spec.timing, "予測時点 (timing) が空です。いつ予測ボタンを押すのかを決めましょう。"),
        (spec.metric, "評価指標 (metric) が空です。ビジネスの損益とつながる指標を選びましょう。"),
        (spec.action, "アクション (action) が空です。アクションのない予測は飾り物です。"),
        (spec.baseline, "ベースライン (baseline) が空です。モデルなし・今のやり方の成績を先に測りましょう。"),
    ]
    for value, msg in checks:
        if not value.strip():
            problems.append(msg)
    return problems


def check_leakage(spec: MLSpec) -> list[str]:
    """リーク点検: 「予測ボタンを押すその瞬間、この値を知り得るか?」"""
    return [
        f"リークの疑い: '{f.name}' は{f.available_at}に生まれる値です。"
        f"予測時点 ({spec.timing}) には存在しないため、材料から外す必要があります。"
        for f in spec.features if f.available_at != "予測時点より前"
    ]


def validate(title: str, spec: MLSpec) -> None:
    """定義書を 1 つ点検して結果を出力。"""
    print(f"  ■ 定義書: {title}")
    print(f"    目標: {spec.business_goal}")
    issues = check_completeness(spec) + check_leakage(spec)
    if not issues:
        print(f"    [合格] 5 要素完備 — 対象: {spec.target} / 単位: {spec.unit}")
        print(f"           時点: {spec.timing} / 指標: {spec.metric}")
        print(f"           アクション: {spec.action}")
    else:
        for i, msg in enumerate(issues, 1):
            print(f"    [指摘 {i}] {msg}")
    print(f"    点数: {6 + len(spec.features) - len(issues)} / {6 + len(spec.features)}\n")


def judge_ml_fitness() -> None:
    """[4] 解ける問題 / 解けない問題の自動分類。
    基準: 反復して起きるか、十分な正解データ、パターンの存在可能性、誤差の許容。"""
    candidates = [
        ("店舗別の明日のサンドイッチ需要予測", dict(repeats=True, labels=3000, pattern=True, error_ok=True)),
        ("消費税 10% の自動計算", dict(repeats=True, labels=100000, pattern=True, error_ok=False)),
        ("自社の M&A 成功可否の予測", dict(repeats=False, labels=12, pattern=True, error_ok=True)),
        ("カード不正取引のリアルタイム検知", dict(repeats=True, labels=50000, pattern=True, error_ok=True)),
    ]
    for name, c in candidates:
        reasons = []
        if not c["repeats"]:
            reasons.append("反復して起きる出来事ではない (事例が蓄積しない)")
        if c["labels"] < 500:
            reasons.append(f"正解データが {c['labels']}件しかない (学習不可能な水準)")
        if not c["error_ok"]:
            reasons.append("誤差の許容ゼロ -> すでに明確なルールがあるならルールで (level00)")
        verdict = "ML で解く価値あり" if not reasons else "ML には不向き"
        print(f"    {name:28s} -> {verdict}")
        for r in reasons:
            print(f"        理由: {r}")


if __name__ == "__main__":
    # [1] 悪い定義書: 願いだけで全部空欄 --------------------------------
    print("[1] 悪い定義書 — 『解約を減らしたい』をそのまま提出した場合")
    validate("願いをそのまま", MLSpec(business_goal="解約を減らしたい"))

    # [2] 良い定義書: 5 要素を埋めた翻訳完了版 -----------------------------
    print("[2] 良い定義書 — 同じ願いを ML の問題に翻訳した場合")
    good = MLSpec(
        business_goal="サブスクの解約を減らしたい",
        target="今月末を基準に、次の 30日以内にサブスクを解約する (1/0)",
        unit="アクティブなサブスク顧客 1 名 (毎月 1日のスナップショット)",
        timing="毎月 1日の午前、前月末までに確定したデータのみ使用",
        metric="リスク上位 10% リストの適合率/再現率 (正解率ではない、level06 参照)",
        action="リスク上位 10% の顧客に CS チームが 1 週間以内にリテンション面談 + 特典提案",
        baseline="現在は『3 か月未ログインの全員に SMS』 — 反応率 2%",
        features=[
            Feature("直近 30日の利用日数", "予測時点より前"),
            Feature("直近 30日のサポート問い合わせ数", "予測時点より前"),
            Feature("契約月数", "予測時点より前"),
        ],
    )
    validate("解約予測 v1", good)

    # [3] リークの罠: 結果確定後に生まれる列が材料に混ざった場合 -----------
    print("[3] リーク点検 — 未来を盗み見る材料が混ざった定義書")
    leaky = MLSpec(
        business_goal="サブスクの解約を減らしたい",
        target=good.target, unit=good.unit, timing=good.timing,
        metric=good.metric, action=good.action, baseline=good.baseline,
        features=[
            Feature("直近 30日の利用日数", "予測時点より前"),
            Feature("解約違約金の請求額", "結果確定後"),   # <- 解約して初めて生まれる値!
            Feature("解約理由アンケートの回答", "結果確定後"),
        ],
    )
    validate("解約予測 v2 (罠)", leaky)
    print("    -> リーク列は試験の成績だけを完璧にし、実戦では使えません。\n")

    # [4] 解ける問題 / 解けない問題の分類 -----------------------------------
    print("[4] 候補問題 4 つの ML 適合性判定")
    judge_ml_fitness()
    print()
    print("[5] まとめ: 翻訳の順序 = 願い -> (対象/単位/時点/指標/アクション) -> リーク点検 -> ベースライン")
    print("    このチェックリストを通過して初めて、モデリング (level03〜) を始める価値が生まれます。")
