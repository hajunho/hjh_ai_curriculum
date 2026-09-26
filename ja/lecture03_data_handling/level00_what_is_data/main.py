"""
データとは何か — 表の構造を辞書のリストで直接解剖します。
行 (row)=事例、列 (column)=属性という原理を目で確認し、
列の抽出・行の検索・型/欠損の観察・ミニスキーマ要約を標準ライブラリだけで行います。
最後に非構造化テキストと比較して、「表」がなぜ集計に有利なのかを見ます。
"""

import pathlib
import sys

# 共用データモジュール (hjh_data) を読み込むためのパス設定
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data


def extract_column(rows: list[dict], col: str) -> list:
    """表から列を1つ、リストとして抜き出します。(列 = 同じ属性の値の集まり)"""
    return [r[col] for r in rows]


def summarize_schema(rows: list[dict]) -> list[dict]:
    """列ごとに型の構成と欠損の数を数えて「ミニスキーマ要約」を作ります。"""
    summary = []
    for col in rows[0].keys():
        values = extract_column(rows, col)
        none_count = sum(1 for v in values if v is None)
        # None を除いた値たちの型の名前を集めます
        type_names = sorted({type(v).__name__ for v in values if v is not None})
        summary.append({
            "column": col,
            "types": "/".join(type_names),
            "missing": none_count,
            "example": next(v for v in values if v is not None),
        })
    return summary


def main() -> None:
    # 30日分の架空カフェ売上表を生成 (seed 固定 → 常に同じ結果)
    rows = hjh_data.sales_table(n_days=30, seed=42)

    print("[1] 表の大きさとスキーマ (列の名前)")
    print(f"    行 (事例) 数: {len(rows)}")
    print(f"    列 (属性) の一覧: {list(rows[0].keys())}")
    print()

    print("[2] 表のプレビュー — 受け取った表は何はともあれまず目で見ます")
    hjh_data.head(rows, n=5)
    print()

    print("[3] 行1つの解剖 — この表の1行は「何の1件」なのか?")
    first = rows[0]
    for key, value in first.items():
        print(f"    {key:>10} = {value!r}  ({type(value).__name__})")
    print("    → 行1つ = 特定の日付・店舗・カテゴリの「1日の売上」1件です。")
    print()

    print("[4] 列の抽出 — revenue 列だけを抜き出してみる")
    revenues = extract_column(rows, "revenue")
    print(f"    revenue 列の長さ: {len(revenues)} (行数と同じです)")
    print(f"    先頭8個の値: {revenues[:8]}")
    print()

    print("[5] 条件で行を探す — 渋谷店のコーヒーの売上だけ")
    gangnam_coffee = [r for r in rows
                     if r["store"] == "渋谷店" and r["category"] == "コーヒー"]
    print(f"    条件に合う行: {len(gangnam_coffee)}個 (30日なので30個が正常)")
    hjh_data.head(gangnam_coffee, n=3)
    print()

    print("[6] ミニスキーマ要約 — 列ごとの型と欠損 (None) の数")
    schema = summarize_schema(rows)
    for s in schema:
        print(f"    {s['column']:>10} | 型: {s['types']:<8} | "
              f"欠損: {s['missing']:>2}個 | 例: {s['example']!r}")
    missing_total = sum(s["missing"] for s in schema)
    negative_count = sum(1 for v in revenues if v is not None and v < 0)
    print(f"    → 全体で欠損 {missing_total}マス、マイナスの売上 {negative_count}件が隠れています。")
    print("      (わざと仕込んだ汚染です。level06 で処理法を学びます)")
    print()

    print("[7] 構造化 vs 非構造化 — 同じ情報、違う形")
    unstructured = "昨日渋谷店に行ったらコーヒーがおいしかった。混んでいたから売上もかなり出たはず!"
    structured = {"date": "2025-01-01", "store": "渋谷店",
                  "category": "コーヒー", "revenue": 512000}
    print(f"    非構造化 (自由な文章): {unstructured!r}")
    print(f"    構造化 (表の行1つ): {structured!r}")
    print("    → 文章には SUM をかけられませんが、表の revenue 列はすぐに合計できます。")
    print("      非構造化データも、分析するには結局、表へ変換する過程を経ます。")


if __name__ == "__main__":
    main()
