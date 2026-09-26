"""
hjh_data.py — カリキュラム専用の合成データジェネレーター

このモジュールは hjh_ai_curriculum の全講義で使う練習用データを「自前で生成」します。
外部データセットをダウンロードしないため、著作権・ライセンスの問題が一切なく、
インターネットのつながらない教室でもまったく同じように動作します。

すべての関数は seed を受け取り、常に同じ結果を再現します。
講義ノートに書かれた数値と実行結果がずれないようにするためです。

作成: ハ・ジュノ (hajunho) · MIT License
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# numpy はあれば使い、なければ純粋な Python で動作します。
# (lecture01〜02 はまだ numpy を学ぶ前なので、依存関係なしで動く必要があります。)
try:
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None


# ---------------------------------------------------------------------------
# 0. 共通ユーティリティ
# ---------------------------------------------------------------------------

def _rng(seed: int) -> random.Random:
    """独立した乱数生成器。グローバルな random の状態には触れません。"""
    return random.Random(seed)


def to_csv(rows: list[dict], path: str) -> str:
    """辞書のリストを CSV ファイルに保存し、そのパスを返します。"""
    import csv
    if not rows:
        raise ValueError("空のデータは保存できません。")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def head(rows: list[dict], n: int = 5) -> None:
    """先頭部分だけを表の形できれいに出力します (pandas なしでも動作)。"""
    if not rows:
        print("(データなし)")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(str(c)), *(len(str(r[c])) for r in rows[:n])) for c in cols}
    line = " | ".join(str(c).ljust(widths[c]) for c in cols)
    print(line)
    print("-" * len(line))
    for r in rows[:n]:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))
    print(f"... 全 {len(rows)}行")


# ---------------------------------------------------------------------------
# 1. 売上データ — lecture03, 05, 07
# ---------------------------------------------------------------------------

STORES = ["渋谷店", "新宿店", "大阪店", "名古屋店", "福岡店"]
CATEGORIES = ["コーヒー", "ベーカリー", "サンドイッチ", "デザート", "ドリンク"]


def sales_table(n_days: int = 365, seed: int = 42) -> list[dict]:
    """
    架空のカフェチェーンの日次売上データ。

    次の性質をわざと仕込んであります (講義の中で一つずつ発見していきます):
      - 週末の売上が平日より高い            -> groupby / 曜日効果
      - 夏にドリンクの売上が跳ね上がる       -> 季節性 / 時系列
      - 欠損値と負の外れ値が混ざっている     -> データクレンジング
      - 広告費と売上に相関がある            -> 回帰 / 相関 vs 因果
    """
    rng = _rng(seed)
    rows: list[dict] = []
    for day in range(n_days):
        weekday = day % 7                      # 0=月 ... 6=日
        is_weekend = weekday >= 5
        season = math.sin(2 * math.pi * day / 365.0)   # -1(冬) 〜 +1(夏)

        for store in STORES:
            store_power = 1.0 + 0.15 * STORES.index(store)
            ad_cost = round(rng.uniform(50_000, 400_000), -3)

            for cat in CATEGORIES:
                base = 300_000 * store_power
                if cat == "ドリンク":
                    base *= 1.0 + 0.45 * season          # 夏に急増
                if cat == "コーヒー":
                    base *= 1.3
                if is_weekend:
                    base *= 1.25
                noise = rng.gauss(1.0, 0.18)
                revenue = base * noise + ad_cost * 0.35

                # データクレンジング練習用の汚染: 1% 欠損、0.5% 負の値
                u = rng.random()
                if u < 0.010:
                    revenue_out = None
                elif u < 0.015:
                    revenue_out = -abs(round(revenue))
                else:
                    revenue_out = round(revenue)

                rows.append({
                    "date": f"2025-{day // 31 + 1:02d}-{day % 31 + 1:02d}",
                    "day_index": day,
                    "weekday": ["月", "火", "水", "木", "金", "土", "日"][weekday],
                    "store": store,
                    "category": cat,
                    "ad_cost": int(ad_cost),
                    "revenue": revenue_out,
                })
    return rows


# ---------------------------------------------------------------------------
# 2. 顧客チャーン(解約)データ — lecture06, 07
# ---------------------------------------------------------------------------

def churn_table(n: int = 2000, seed: int = 7) -> list[dict]:
    """
    サブスクリプションサービスの解約データ (二値分類用)。

    本物のシグナル:  利用日数↓、サポート問い合わせ↑、プラン変更履歴↑  -> 解約確率↑
    偽物のシグナル:  customer_id には何の意味もない (リーク変数の練習用)
    解約率は約 18% で、不均衡データの実習に適しています。
    """
    rng = _rng(seed)
    rows = []
    for i in range(n):
        tenure = rng.randint(1, 60)                       # 契約月数
        monthly_fee = rng.choice([9900, 14900, 19900, 29900])
        usage_days = max(0, min(30, int(rng.gauss(18, 7))))
        support_calls = max(0, int(rng.expovariate(1 / 1.3)))
        plan_changes = max(0, int(rng.expovariate(1 / 0.6)))
        is_auto_pay = rng.random() < 0.7

        # ログオッズを直接設計 -> 正解のあるデータ
        z = (-1.2
             - 0.05 * usage_days
             + 0.45 * support_calls
             + 0.40 * plan_changes
             - 0.020 * tenure
             + 0.00004 * monthly_fee
             - (0.7 if is_auto_pay else 0.0))
        p = 1 / (1 + math.exp(-z))
        churned = 1 if rng.random() < p else 0

        rows.append({
            "customer_id": f"C{100000 + i}",
            "tenure_months": tenure,
            "monthly_fee": monthly_fee,
            "usage_days_30d": usage_days,
            "support_calls_30d": support_calls,
            "plan_changes": plan_changes,
            "auto_pay": int(is_auto_pay),
            "churned": churned,
        })
    return rows


# ---------------------------------------------------------------------------
# 3. 不正取引データ — lecture07
# ---------------------------------------------------------------------------

def fraud_table(n: int = 5000, seed: int = 11) -> list[dict]:
    """カードの不正取引データ。不正の比率を約 1.5% にして極端な不均衡を作ります。"""
    rng = _rng(seed)
    rows = []
    for i in range(n):
        is_fraud = rng.random() < 0.015
        if is_fraud:
            amount = rng.lognormvariate(12.5, 1.1)     # 異常に大きい金額
            hour = rng.choice([0, 1, 2, 3, 4, 23])     # 深夜・未明の時間帯
            foreign = rng.random() < 0.55
            n_recent = rng.randint(5, 20)              # 短時間の連続決済
        else:
            amount = rng.lognormvariate(10.2, 0.9)
            hour = int(max(0, min(23, rng.gauss(14, 4))))
            foreign = rng.random() < 0.05
            n_recent = rng.randint(0, 4)
        rows.append({
            "tx_id": f"T{i:06d}",
            "amount": round(amount),
            "hour": hour,
            "is_foreign": int(foreign),
            "tx_count_1h": n_recent,
            "is_fraud": int(is_fraud),
        })
    return rows


# ---------------------------------------------------------------------------
# 4. 日本語テキストコーパス — lecture10, 11, 12
# ---------------------------------------------------------------------------

REVIEW_POSITIVE = [
    "配送がとても速くて助かりました", "値段の割に品質が素晴らしいです",
    "リピート購入する気満々です", "梱包が丁寧で満足しています",
    "思ったよりずっと丈夫で良いです", "店員さんが親切に対応してくれました",
    "写真と同じで気に入りました", "店内の雰囲気がとても快適でした",
    "性能が期待以上で驚きました", "設置が簡単で楽でした",
]
REVIEW_NEGATIVE = [
    "配送に一週間もかかりました", "値段の割に品質が悪すぎます",
    "二度と買わないと思います", "梱包が破れた状態で届きました",
    "思ったより弱くてすぐ壊れました", "問い合わせをしたのに返事がありません",
    "写真と色が全く違います", "店内が狭くてうるさかったです",
    "性能が説明と違ってがっかりしました", "説明書が分かりにくくて迷いました",
]


def review_corpus(n: int = 600, seed: int = 3) -> list[dict]:
    """感情分類用の日本語レビューデータ。label 1=ポジティブ、0=ネガティブ。"""
    rng = _rng(seed)
    fillers = ["", " ただ、少し残念な点もありました。", " また利用したいと思います。",
               " ご参考になれば幸いです。", " 星の数は正直につけています。"]
    rows = []
    for i in range(n):
        label = i % 2
        pool = REVIEW_POSITIVE if label == 1 else REVIEW_NEGATIVE
        text = rng.choice(pool) + rng.choice(fillers)
        rows.append({"id": i, "text": text, "label": label})
    rng.shuffle(rows)
    return rows


SAMPLE_DOCS = {
    "社内規定_休暇.txt": (
        "第1条 年次有給休暇は、入社日を基準として勤続1年で15日付与される。"
        "勤続3年以上の社員には2年ごとに1日ずつ加算され、最大で25日を超えない。"
        "年次休暇の取得は、遅くとも3営業日前までに決裁システムで申請しなければならない。"
        "未使用の年次休暇は、会計年度終了後に手当として精算する。"
    ),
    "社内規定_経費.txt": (
        "第2条 出張経費は、事前承認を受けた案件に限り精算する。"
        "出張日当は国内出張が3万ウォン、海外出張が8万ウォンを基準とする。"
        "領収書は出張終了後7日以内に提出しなければならず、"
        "提出期限を過ぎた経費は原則として精算されない。"
    ),
    "社内規定_在宅勤務.txt": (
        "第3条 在宅勤務は週2回まで認められ、チームリーダーの承認が必要である。"
        "在宅勤務日であっても、コアタイムである午前10時から午後4時までは連絡が取れる状態を保つこと。"
        "会社の資産を社外へ持ち出す場合は、情報セキュリティ部門へ事前に届け出る。"
    ),
    "製品マニュアル_設置.txt": (
        "本製品を設置する前に、必ず電源を切ってください。"
        "壁面から最低10センチメートル以上の間隔を空け、通風を確保する必要があります。"
        "初期設定は、電源を入れた後に画面の案内に従えば5分以内に完了します。"
        "設置後に異音が聞こえた場合は、直ちに使用を中止してカスタマーセンターへお問い合わせください。"
    ),
    "製品マニュアル_保証.txt": (
        "製品の保証期間は購入日から2年間です。"
        "お客様の過失による破損、水没、落下は保証の対象外となります。"
        "保証修理を受けるには、購入時のレシートまたは注文番号が必要です。"
        "消耗品は保証期間にかかわらず有償で交換となります。"
    ),
}


def tiny_corpus(seed: int = 5) -> str:
    """
    ミニ言語モデルの事前学習用日本語テキスト (lecture12)。
    文法パターンが繰り返されるため、小さなモデルでも学習シグナルを
    つかめるように設計してあります。
    """
    rng = _rng(seed)
    subjects = ["学生が", "会社員が", "料理人が", "開発者が", "先生が"]
    objects = ["報告書を", "カレーライスを", "プログラムを", "手紙を", "計画書を"]
    verbs = ["作った", "直した", "確認した", "整理した", "準備した"]
    times = ["昨日", "今日", "朝に", "夕方に", "週末に"]
    lines = []
    for _ in range(2000):
        lines.append(f"{rng.choice(times)} {rng.choice(subjects)} "
                     f"{rng.choice(objects)} {rng.choice(verbs)}。")
    return " ".join(lines)


# ---------------------------------------------------------------------------
# 5. 画像データ — lecture09
# ---------------------------------------------------------------------------

def shape_images(n: int = 800, size: int = 16, seed: int = 13):
    """
    図形分類用の白黒画像。0=四角形、1=円、2=三角形。
    MNIST をダウンロードしなくても CNN の実習ができるよう自前で描画します。
    numpy が必要です。戻り値: (X, y) — X は (n, size, size) の float32 配列。
    """
    if _np is None:
        raise ImportError("shape_images() には numpy が必要です。pip install numpy")
    rng = _np.random.default_rng(seed)
    X = _np.zeros((n, size, size), dtype="float32")
    y = _np.zeros(n, dtype="int64")
    for i in range(n):
        label = i % 3
        y[i] = label
        img = _np.zeros((size, size), dtype="float32")
        cy, cx = rng.integers(5, size - 5, size=2)
        r = int(rng.integers(3, 5))
        yy, xx = _np.mgrid[0:size, 0:size]
        if label == 0:                                    # 四角形
            img[max(0, cy - r):cy + r, max(0, cx - r):cx + r] = 1.0
        elif label == 1:                                  # 円
            img[((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r] = 1.0
        else:                                             # 三角形
            mask = (yy >= cy - r) & (yy <= cy + r) & (_np.abs(xx - cx) <= (yy - cy + r) / 2)
            img[mask] = 1.0
        img += rng.normal(0, 0.08, img.shape).astype("float32")   # ノイズ
        X[i] = _np.clip(img, 0.0, 1.0)
    idx = rng.permutation(n)
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# 6. リレーショナル DB スキーマ — lecture04
# ---------------------------------------------------------------------------

def build_sqlite(path: str = "hjh_shop.db", seed: int = 21) -> str:
    """
    SQL 実習用の SQLite データベースを作ります。
    テーブル: customers, products, orders, order_items, employees
    Python 標準ライブラリだけを使うので、DB のインストールは不要です。
    """
    import os
    import sqlite3

    if os.path.exists(path):
        os.remove(path)
    rng = _rng(seed)
    con = sqlite3.connect(path)
    cur = con.cursor()

    cur.executescript("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
            city TEXT, grade TEXT, joined_at TEXT);
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
            category TEXT, price INTEGER, cost INTEGER);
        CREATE TABLE employees (
            employee_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
            dept TEXT, salary INTEGER, manager_id INTEGER);
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY, customer_id INTEGER,
            employee_id INTEGER, ordered_at TEXT, status TEXT);
        CREATE TABLE order_items (
            order_id INTEGER, product_id INTEGER, quantity INTEGER,
            PRIMARY KEY (order_id, product_id));
    """)

    cities = ["東京", "大阪", "名古屋", "福岡", "札幌", "横浜"]
    grades = ["VIP", "GOLD", "SILVER", "BASIC"]
    surnames = ["佐藤", "鈴木", "高橋", "田中", "伊藤", "渡辺", "山本", "中村", "小林", "加藤"]
    givens = ["翔太", "陽菜", "大輝", "結衣", "蓮", "美咲", "悠斗", "さくら", "健太", "葵"]

    customers = [(i, rng.choice(surnames) + rng.choice(givens), rng.choice(cities),
                  rng.choice(grades), f"202{rng.randint(0, 5)}-{rng.randint(1, 12):02d}-15")
                 for i in range(1, 201)]
    cur.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", customers)

    pnames = [("ノートパソコン", "電子機器"), ("ワイヤレスマウス", "電子機器"), ("メカニカルキーボード", "電子機器"),
              ("モニター", "電子機器"), ("コーヒー豆", "食品"), ("チョコレート", "食品"),
              ("タンブラー", "生活雑貨"), ("ノート", "文具"), ("万年筆", "文具"), ("チェア", "家具")]
    products = []
    for pid, (nm, cat) in enumerate(pnames, start=1):
        price = rng.choice([4900, 12900, 29000, 89000, 350000, 1200000])
        products.append((pid, nm, cat, price, int(price * rng.uniform(0.45, 0.75))))
    cur.executemany("INSERT INTO products VALUES (?,?,?,?,?)", products)

    depts = ["営業", "マーケティング", "開発", "CS"]
    employees = [(1, "山田美月", "営業", 9000, None)]
    for eid in range(2, 21):
        employees.append((eid, rng.choice(surnames) + rng.choice(givens),
                          rng.choice(depts), rng.randint(3200, 8500),
                          1 if eid <= 5 else rng.randint(2, 5)))
    cur.executemany("INSERT INTO employees VALUES (?,?,?,?,?)", employees)

    orders, items = [], []
    for oid in range(1, 1001):
        orders.append((oid, rng.randint(1, 200), rng.randint(1, 20),
                       f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                       rng.choices(["完了", "キャンセル", "配送中"], weights=[8, 1, 2])[0]))
        for pid in rng.sample(range(1, 11), rng.randint(1, 3)):
            items.append((oid, pid, rng.randint(1, 5)))
    cur.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", orders)
    cur.executemany("INSERT INTO order_items VALUES (?,?,?)", items)

    con.commit()
    con.close()
    return path


if __name__ == "__main__":
    print("=== hjh_data セルフチェック ===\n")
    print("[1] 売上データ"); head(sales_table(n_days=10), 3)
    print("\n[2] チャーンデータ"); head(churn_table(200), 3)
    print("\n[3] 不正取引データ"); head(fraud_table(500), 3)
    print("\n[4] レビューコーパス"); head(review_corpus(20), 3)
    print(f"\n[5] 文書 {len(SAMPLE_DOCS)}件、ミニコーパス {len(tiny_corpus()):,}文字")
    print(f"\n[6] SQLite 生成 -> {build_sqlite('/tmp/hjh_check.db')}")
    print("\nすべてのジェネレーターが正常に動作しました。")
