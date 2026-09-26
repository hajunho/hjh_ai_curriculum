"""
大規模データ処理の戦略の実習。
8年分の売上データ (73,000行) を相手に
  - memory_usage(deep=True) で列ごとのメモリ測定
  - dtype ダイエット (category / int32 / float32) の前後比較
  - read_csv(chunksize=...) によるチャンク・ストリーミング集計と結果の検証
  - usecols/dtype を指定した読み込み最適化の時間・メモリ比較
を行い、「いつ pandas を離れるか」の基準を整理します。
"""

import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

# 共通データモジュール (hjh_data) を読み込むためのパス設定
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
CSV_PATH = OUT_DIR / "big_sales.csv"


def fmt_bytes(n: float) -> str:
    """バイト数を人が読みやすい単位に変えます。"""
    return f"{n / 1024 / 1024:.2f} MB" if n >= 1024 * 1024 else f"{n / 1024:.1f} KB"


def step1_measure(df: pd.DataFrame) -> int:
    """[1] 体重を測る: 列ごとのメモリ使用量を deep=True で測定します。"""
    print(f"\n[1] memory_usage(deep=True) — データ {len(df):,}行の列ごとのメモリ")
    mem = df.memory_usage(deep=True)
    for col, n_bytes in mem.items():
        dtype = "-" if col == "Index" else str(df[col].dtype)
        print(f"    {str(col):10s} ({dtype:7s}): {fmt_bytes(n_bytes):>10s}")
    total = int(mem.sum())
    print(f"    合計: {fmt_bytes(total)}")
    print("    -> 文字列 (object) の列が数値の列より何倍も重い「メモリのカバ」です。")
    return total


def step2_dtype_diet(df: pd.DataFrame, before: int) -> pd.DataFrame:
    """[2] dtype ダイエット: category / int32 / float32 に変えて削減率を計算します。"""
    print("\n[2] dtype ダイエット — 同じデータを、もっと小さな服で")
    opt = df.copy()
    # 繰り返しの多い文字列 -> category (辞書 + 整数コードに圧縮)
    for col in ["date", "weekday", "store", "category"]:
        opt[col] = opt[col].astype("category")
    # 値の範囲が int32 に十分収まる整数の列 -> 半分のサイズに
    opt["day_index"] = opt["day_index"].astype("int32")
    opt["ad_cost"] = opt["ad_cost"].astype("int32")
    # 分析用の実数の列 -> float32 (会計報告用なら float64 の維持が安全)
    opt["revenue"] = opt["revenue"].astype("float32")

    after = int(opt.memory_usage(deep=True).sum())
    saving = (1 - after / before) * 100
    print(f"    最適化の前: {fmt_bytes(before)}")
    print(f"    最適化の後: {fmt_bytes(after)}  (削減率 {saving:.1f}%)")

    # 値が損なわれていないかを検証 (category は元の文字列に戻して比較)
    assert (opt["store"].astype(str) == df["store"]).all()
    assert (opt["day_index"].astype("int64") == df["day_index"]).all()
    ok = np.allclose(opt["revenue"].astype("float64"), df["revenue"], equal_nan=True)
    print(f"    検証: 文字列・整数の列は完全一致、revenue は float32 の精度内で一致 ({ok})")
    print("    -> category の原理: 「渋谷店」を73,000回書く代わりに、辞書に1回書いて番号だけを保存")
    return opt


def step3_chunked_aggregation(df: pd.DataFrame) -> None:
    """[3] チャンク・ストリーミング集計: 分けて読んでも全体の集計と同じことを検証します。"""
    print("\n[3] read_csv(chunksize) — メモリより大きいファイルを扱うパターン")
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"    CSV を保存: {CSV_PATH} ({fmt_bytes(CSV_PATH.stat().st_size)})")

    # チャンクを巡回: 部分集計 (合計) を累積して合わせます。
    total_by_store = None
    n_chunks = 0
    for chunk in pd.read_csv(CSV_PATH, chunksize=10_000):
        part = chunk.groupby("store")["revenue"].sum()
        total_by_store = part if total_by_store is None else total_by_store.add(part, fill_value=0)
        n_chunks += 1
    print(f"    10,000行ずつ {n_chunks}個のチャンクに分けて、店舗別の合計を累積集計しました。")

    # 一度に読んだ結果と比較して検証
    full = pd.read_csv(CSV_PATH).groupby("store")["revenue"].sum()
    match = np.allclose(total_by_store.sort_index(), full.sort_index())
    print(f"    検証: チャンク集計 == 全体集計 ? {match}")
    top = total_by_store.sort_values(ascending=False)
    print("    店舗別の売上合計(億ウォン): "
          + ", ".join(f"{s} {v / 1e8:.0f}" for s, v in top.items()))
    print("    -> 合計・件数・最大最小はこのパターンで OK、中央値のように全体が必要な統計は不可。")


def step4_read_optimized() -> None:
    """[4] 読み込みの最適化: usecols + dtype 指定の時間・メモリの効果を比較します。"""
    print("\n[4] 読み込みの最適化 — 必要な列だけを、ふさわしい型で")
    t0 = time.perf_counter()
    naive = pd.read_csv(CSV_PATH)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    smart = pd.read_csv(CSV_PATH, usecols=["store", "revenue"],
                        dtype={"store": "category", "revenue": "float32"})
    t_smart = time.perf_counter() - t0

    mem_naive = naive.memory_usage(deep=True).sum()
    mem_smart = smart.memory_usage(deep=True).sum()
    print(f"    全体を読み込み  : {t_naive * 1000:6.0f} ms / メモリ {fmt_bytes(mem_naive)}")
    print(f"    usecols+dtype   : {t_smart * 1000:6.0f} ms / メモリ {fmt_bytes(mem_smart)}"
          f"  (メモリ {(1 - mem_smart / mem_naive) * 100:.0f}% 削減)")
    print("    -> ファイルが小さいと時間差はわずかですが、メモリの削減はいつでも確実です。")
    # 参考: Parquet (列指向フォーマット) なら下の1行で済みます (pyarrow が必要)。
    #   df.to_parquet("big_sales.parquet")
    #   pd.read_parquet("big_sales.parquet", columns=["store", "revenue"])
    # 列だけを選んで読む・型の保存・圧縮が内蔵されており、繰り返し使うデータの標準です。


def step5_when_to_leave_pandas() -> None:
    """[5] 判断基準: いつ pandas を離れて DB/Spark へ行くか。"""
    print("\n[5] いつ pandas を離れるか — 判断基準のまとめ")
    rules = [
        ("データがメモリの1/3以下", "そのまま pandas"),
        ("メモリにぎりぎり収まる", "dtype ダイエット + Parquet で保存"),
        ("メモリより大きいが集計が目的", "chunksize ストリーミング、または Polars/DuckDB"),
        ("複数人が同時に照会・更新", "データベース(DB) — lecture04"),
        ("数億行以上、サーバーが複数台必要", "Spark などの分散処理"),
    ]
    for cond, action in rules:
        print(f"    {cond:26s} -> {action}")
    print("    -> 基準は「何GBか」ではなく作業の性格です。分散システムはそれ自体がコストです。")


def main() -> None:
    print("=" * 60)
    print("Level 11 — 大規模データ処理の戦略")
    print("=" * 60)
    # 8年分(2,920日) x 5店舗 x 5カテゴリ = 73,000行。seed 固定で再現可能。
    df = pd.DataFrame(hjh_data.sales_table(n_days=2920, seed=42))
    before = step1_measure(df)
    step2_dtype_diet(df, before)
    step3_chunked_aggregation(df)
    step4_read_optimized()
    step5_when_to_leave_pandas()
    print("\n完了! lecture03 を終えました。次は lecture04 — データベースと SQL です。")


if __name__ == "__main__":
    main()
