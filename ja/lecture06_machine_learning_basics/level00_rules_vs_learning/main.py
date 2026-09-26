"""
level00 — ルール vs 学習

同じ分類問題 (カードの不正取引検知) を 2 つの方式で解いて比較します。
  A. 手書きルール: 人が勘で決めた IF 文 (「50万ウォン以上 + 深夜なら不正」)
  B. 学習: しきい値の候補をすべて試し、データが最適値を選ぶ探索 (自前で実装)
核心メッセージ: 従来のプログラミングは「ルール+データ->答え」、機械学習は「データ+答え->ルール」。
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

NIGHT_HOURS = {0, 1, 2, 3, 4, 23}          # 深夜・未明の時間帯


def score(y_true: list[int], y_pred: list[int]) -> dict:
    """ルールの成績表。不正は 1.4% しかないため単純な正解率は錯覚を生む。
    そこで「不正の検知率(再現率)」と「正常の検知率」の平均(バランススコア)を基準にする。"""
    n_fraud = sum(y_true)
    hit_fraud = sum(p == 1 for t, p in zip(y_true, y_pred) if t == 1) / n_fraud
    hit_normal = sum(p == 0 for t, p in zip(y_true, y_pred) if t == 0) / (len(y_true) - n_fraud)
    n_alarm = sum(y_pred)
    precision = (sum(t == 1 for t, p in zip(y_true, y_pred) if p == 1) / n_alarm) if n_alarm else 0.0
    return {"balanced": (hit_fraud + hit_normal) / 2, "recall": hit_fraud,
            "precision": precision, "alarms": n_alarm}


def hand_rule(row: dict) -> int:
    """[手書きルール] 会議室で出てきそうな勘ベースのルール。
    「50万ウォンを超える大金が深夜に引き落とされたら不正だろう。」"""
    return 1 if (row["amount"] >= 500_000 and row["hour"] in NIGHT_HOURS) else 0


def learn_threshold(rows: list[dict], feature: str) -> tuple[float, float]:
    """[学習] 「値 >= T なら不正」というルールのしきい値 T をすべて試す。
    人はルールの「形」だけを決め、「数字」T はデータが選ぶ。"""
    y_true = [r["is_fraud"] for r in rows]
    best_t, best_s = None, -1.0
    for t in sorted({r[feature] for r in rows}):
        y_pred = [1 if r[feature] >= t else 0 for r in rows]
        s = score(y_true, y_pred)["balanced"]
        if s > best_s:
            best_t, best_s = t, s
    return best_t, best_s


def learn_combo(rows: list[dict]) -> tuple[float, float]:
    """[学習の拡張] 「金額 >= T かつ深夜時間帯」の T を探索。
    人のルールと形は同じだが、数字はデータが選ぶ。"""
    y_true = [r["is_fraud"] for r in rows]
    best_t, best_s = None, -1.0
    for t in sorted({round(r["amount"], -4) for r in rows}):    # 1万ウォン単位の候補
        y_pred = [1 if (r["amount"] >= t and r["hour"] in NIGHT_HOURS) else 0
                  for r in rows]
        s = score(y_true, y_pred)["balanced"]
        if s > best_s:
            best_t, best_s = t, s
    return best_t, best_s


def report(name: str, s: dict) -> None:
    print(f"    {name}")
    print(f"      バランススコア {s['balanced']:.1%} / 不正検知率 {s['recall']:.1%} / "
          f"警報の適合率 {s['precision']:.1%} (警報 {s['alarms']}件)")


if __name__ == "__main__":
    # [1] データの準備 -----------------------------------------------------
    rows = hjh_data.fraud_table(n=5000, seed=11)   # seed 固定 -> 常に同じデータ
    y_true = [r["is_fraud"] for r in rows]
    print("[1] データ: カード取引", len(rows), "件 (スパムフィルターと同じ構造の分類問題)")
    hjh_data.head(rows, 3)
    print(f"    不正の比率: {sum(y_true) / len(y_true):.1%}  (不正=1, 正常=0)\n")

    # [2] 方式 A — 手書きルール ---------------------------------------------
    pred_hand = [hand_rule(r) for r in rows]
    s_hand = score(y_true, pred_hand)
    print("[2] 方式 A — 手書きルール (人が数字を勘で決める)")
    print("    ルール: 金額 >= 500,000ウォン かつ深夜時間帯なら不正")
    report("成績:", s_hand)
    print("      -> 捕まえたものは全部本物ですが、不正の 70% を見逃します。基準が高すぎたのです。\n")

    # [3] 方式 B — 学習: 最適なしきい値をデータが選ぶ --------------------
    print("[3] 方式 B — 学習 (しきい値 T の候補をすべて試し、データが選択)")
    best_feat, best_t, best_s = None, None, -1.0
    for feat in ["amount", "hour", "is_foreign"]:
        t, s = learn_threshold(rows, feat)
        print(f"    特徴量 {feat:12s}: 最適 T={t:>10,} -> バランススコア {s:.1%}")
        if s > best_s:
            best_feat, best_t, best_s = feat, t, s
    pred_learn = [1 if r[best_feat] >= best_t else 0 for r in rows]
    s_learn = score(y_true, pred_learn)
    print(f"    => データが選んだルール: \"{best_feat} >= {best_t:,} なら不正\"")
    report("成績:", s_learn)
    print("      -> 人が知らなかった数字をデータが見つけましたが、空振り警報 (適合率↓) が多めです。\n")

    # [4] 方式 B の拡張 — 特徴量 2 個の組み合わせ ---------------------------
    t2, s2 = learn_combo(rows)
    pred_two = [1 if (r["amount"] >= t2 and r["hour"] in NIGHT_HOURS) else 0 for r in rows]
    s_combo = score(y_true, pred_two)
    print("[4] 方式 B の拡張 — 人のルールと同じ形、数字だけデータが決める")
    print(f"    データが選んだルール: \"金額 >= {t2:,}ウォン かつ深夜なら不正\"")
    report("成績:", s_combo)
    print("      -> 人の勘 (50万ウォン) とデータの答え (約 2万ウォン) はこれほど違います。\n")

    # [5] 総合比較 -------------------------------------------------------
    print("[5] 総合比較 (バランススコア = 不正検知率と正常検知率の平均)")
    print("    方式                          バランス   不正検知率   警報適合率")
    print(f"    A. 手書きルール                {s_hand['balanced']:6.1%}     {s_hand['recall']:6.1%}      {s_hand['precision']:6.1%}")
    print(f"    B. 学習(特徴量 1 個)           {s_learn['balanced']:6.1%}     {s_learn['recall']:6.1%}      {s_learn['precision']:6.1%}")
    print(f"    B. 学習(特徴量 2 個の結合)     {s_combo['balanced']:6.1%}     {s_combo['recall']:6.1%}      {s_combo['precision']:6.1%}")
    print()
    print("    核心: 従来のプログラミング  ルール + データ -> 答え")
    print("          機械学習            データ + 答え  -> ルール")
    print("    今日の「学習」は for 文によるしきい値探索でしたが、ニューラルネットも本質は同じです。")
    print("    (注意: 今回は訓練データで採点したため、成績は楽観的です -> level04)")
