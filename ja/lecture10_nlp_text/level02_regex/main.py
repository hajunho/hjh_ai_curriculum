"""
level02 — 正規表現

re モジュールで架空の社内文書から電話番号・メールアドレス・金額を抽出し、
個人情報のマスキングと貪欲 (greedy) マッチングの罠まで実習します。
標準ライブラリだけを使います。
"""

import re

# ---------------------------------------------------------------------------
# 実習用の社内文書 (この講義のために創作した架空のテキスト)
# ---------------------------------------------------------------------------

DOCUMENTS = {
    "発注確認_メール.txt": (
        "お世話になっております。総務部の田中誠です。9月の発注の件、確認いたします。"
        "オフィスチェア20脚、総額1,200,000円で、請求書は "
        "tanaka.m@example.co.jp までお送りください。お急ぎの場合は 03-3122-4567 "
        "または 090-2345-6789 までご連絡ください。"
    ),
    "取引先_連絡網.txt": (
        "アオゾラ物流 担当 佐藤美咲 090-8765-4321 / misaki@aozora-logi.jp、"
        "ナゴヤ印刷 代表番号 052-777-0912、見積もりのお問い合わせは quote@nagoya-print.co.jp まで。"
        "夜間の緊急配車は 09055551234 へ SMS をお願いします。"
    ),
    "精算_お知らせ.txt": (
        "第3四半期イベント費精算のご案内: ブース賃借料 350万円、販促物として 9900円の"
        "エコバッグ 300個 (総額 2,970,000円) です。証憑漏れのお問い合わせは財務部 "
        "fin.help@example.co.jp (内線 03-3122-9999) までお願いします。"
    ),
}

# 名前付きパターン集 — メンテナンスしやすいよう 1 か所にまとめておきます。
PATTERNS = {
    "電話番号": r"0\d{1,2}-?\d{3,4}-?\d{4}",
    "メール": r"[\w.]+@[\w.-]+\.[a-z]{2,}",
    "金額": r"\d{1,3}(?:,\d{3})+円?|\d+万\s?円|\d+円",
}


def demo_basics() -> None:
    """[1] 基本部品のウォーミングアップ — 何を / いくつ / どこで"""
    print("[1] 文法ウォーミングアップ: パターンの部品がそれぞれ何を捕まえるか")
    sample = "注文番号 A-2093、数量 15個、担当 田中、メモ: 9月26日出荷"
    drills = [
        (r"\d+", "数字の塊"),
        (r"[一-龯]+", "漢字の塊"),
        (r"[A-Z]-\d{4}", "大文字-数字4桁 (注文番号の形式)"),
        (r"\d+個", "数字+「個」 (数量表現)"),
    ]
    print(f"    対象: {sample}")
    for pattern, meaning in drills:
        found = re.findall(pattern, sample)
        print(f"    {pattern:12s} ({meaning:20s}) -> {found}")
    print()


def demo_extraction() -> None:
    """[2] 文書 3 件から電話番号・メール・金額を一括抽出"""
    print("[2] 情報抽出: 文書の山から連絡先と金額だけを回収する")
    for name, text in DOCUMENTS.items():
        print(f"    -- {name}")
        for label, pattern in PATTERNS.items():
            found = re.findall(pattern, text)
            print(f"       {label:4s}: {found}")
    print("    -> 人が蛍光ペンでやっていた作業が findall 3 回で終わります。\n")


def mask_phone(text: str) -> str:
    """電話番号の真ん中の桁を **** にマスキング。グループ参照 \\1, \\3 を使用。"""
    return re.sub(r"(0\d{1,2}-?)(\d{3,4})(-?\d{4})", r"\1****\3", text)


def demo_masking() -> None:
    """[3] 個人情報のマスキング — re.sub とグループ参照"""
    print("[3] 個人情報マスキング: 公開資料を作るときの必須作業")
    original = DOCUMENTS["取引先_連絡網.txt"]
    masked = mask_phone(original)
    print(f"    原文    : {original[:40]}...")
    print(f"    マスキング: {masked[:40]}...")
    n = len(re.findall(r"\*{4}", masked))
    print(f"    -> 電話番号 {n}件の真ん中の桁が **** に変わりました。\n")


def demo_greedy() -> None:
    """[4] 貪欲マッチング事故の再現 — .* vs .*?"""
    print("[4] 貪欲マッチングの罠: アスタリスクは基本的に「できるだけ長く」つかむ")
    text = "参加者: <田中誠> <佐藤美咲> <伊藤大輝>"
    greedy = re.findall(r"<.*>", text)
    lazy = re.findall(r"<.*?>", text)
    print(f"    対象         : {text}")
    print(f"    <.*>  (貪欲) : {greedy}   <- 丸ごと 1 つの塊!")
    print(f"    <.*?> (控えめ): {lazy}")
    # 置換事故: タグだけ消すつもりが名前まで全部削除されるケース
    broken = re.sub(r"<.*>", "", text)
    fixed = re.sub(r"<.*?>", "", text)
    print(f"    タグ削除(貪欲)  : {broken!r}  <- データが塊ごと蒸発")
    print(f"    タグ削除(控えめ): {fixed!r}")
    print("    -> 置換結果が不自然に消えていたら、十中八九、貪欲マッチングです。\n")


def demo_summary_table() -> None:
    """[5] 抽出結果を CSV スタイルの要約表に — 実務の最終成果物の形"""
    print("[5] 最終整理: 文書別の抽出結果の要約表 (Excel 貼り付け用)")
    print("    文書,電話番号数,メール数,金額数")
    for name, text in DOCUMENTS.items():
        counts = [len(re.findall(p, text)) for p in PATTERNS.values()]
        print(f"    {name},{counts[0]},{counts[1]},{counts[2]}")
    print("\n結論: 正規表現は「見た目で探す」技術です。「意味で探す」方法は次のレベルから。")


if __name__ == "__main__":
    print("=" * 70)
    print("正規表現 — ビジネス文書から情報を自動抽出する")
    print("=" * 70 + "\n")
    demo_basics()
    demo_extraction()
    demo_masking()
    demo_greedy()
    demo_summary_table()
