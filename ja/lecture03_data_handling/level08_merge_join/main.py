"""
結合 (merge)・連結 (concat) の実習。
売上表 + 店舗情報表 + 目標表という3つの表をつなぎ合わせて
店舗別の目標達成率ランキングを作ります。
inner/left/outer による行数の違い、indicator による検証、
キーの重複が引き起こす行数爆発の落とし穴まで自分の目で確かめます。
"""

import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data


def build_performance() -> pd.DataFrame:
    """1年分の売上をクレンジングして、店舗別の実績合計の表を作ります。"""
    df = pd.DataFrame(hjh_data.sales_table(n_days=365, seed=42))  # seed 固定
    df.loc[df["revenue"] < 0, "revenue"] = np.nan  # ドメインルールでクレンジング
    df = df.dropna(subset=["revenue"])
    perf = (df.groupby("store", as_index=False)
              .agg(actual=("revenue", "sum")))
    return df, perf


def main() -> None:
    # ------------------------------------------------------------------
    print("[1] 店舗別の実績集計 (level07 の復習)")
    df, perf = build_performance()
    perf["actual_mil"] = (perf["actual"] / 1e6).round(1)  # 百万ウォン単位
    print(perf[["store", "actual_mil"]].to_string(index=False))

    # ------------------------------------------------------------------
    print("\n[2] 補助の表を準備 — 総務部の名簿と企画部の目標表 (コードで直接生成)")
    # わざと大阪店を抜き、売上表には存在しない新規の「横浜店」を入れてあります。
    store_info = pd.DataFrame({
        "store": ["渋谷店", "新宿店", "名古屋店", "福岡店", "横浜店"],
        "region": ["関東", "関東", "中部", "九州", "関東"],
        "open_year": [2015, 2017, 2019, 2021, 2025],
        "manager": ["田中彩香", "鈴木健太", "佐藤悠斗", "高橋沙織", "渡辺翔太"],
    })
    targets = pd.DataFrame({
        "store": ["渋谷店", "新宿店", "大阪店", "名古屋店", "福岡店"],
        "target_mil": [800, 850, 900, 1000, 1100],  # 年間目標(百万ウォン)
    })
    print(f"  store_info {len(store_info)}行 (大阪店が欠落、横浜店が新規)")
    print(f"  targets    {len(targets)}行")

    # ------------------------------------------------------------------
    print("\n[3] how 4種の比較 — 同じ2つの表、違う行数")
    for how in ["inner", "left", "outer"]:
        merged = pd.merge(perf, store_info, on="store", how=how)
        print(f"  how='{how:5s}' -> {len(merged)}行")
    print("  (right は左右を入れ替えた left と同じなので省略)")
    audit = pd.merge(perf, store_info, on="store", how="outer", indicator=True)
    print("  indicator による検証 (_merge 列):")
    print(audit["_merge"].value_counts().to_string())
    only_left = audit.loc[audit["_merge"] == "left_only", "store"].tolist()
    only_right = audit.loc[audit["_merge"] == "right_only", "store"].tolist()
    print(f"  -> 売上はあるのに名簿にない: {only_left} / 名簿にしかない: {only_right}")

    # ------------------------------------------------------------------
    print("\n[4] 目標表を結合 -> 達成率の計算とランキング")
    result = pd.merge(perf, targets, on="store", how="left")
    result["achieve_pct"] = (result["actual_mil"] / result["target_mil"] * 100).round(1)
    result = result.sort_values("achieve_pct", ascending=False).reset_index(drop=True)
    result.index = result.index + 1  # 1位から表示する
    print(result[["store", "actual_mil", "target_mil", "achieve_pct"]].to_string())
    best = result.iloc[0]
    print(f"  -> 1位 {best['store']}: 目標に対して {best['achieve_pct']}% 達成")

    # ------------------------------------------------------------------
    print("\n[5] キー重複の落とし穴 — 行が音もなく増えていく")
    dup_targets = pd.concat(
        [targets, targets.iloc[[0]]], ignore_index=True)  # 渋谷店の目標が2行に!
    print(f"  汚染された目標表: {len(dup_targets)}行 "
          f"(渋谷店の重複 {int(dup_targets['store'].duplicated().sum())}件)")
    boom = pd.merge(perf, dup_targets, on="store", how="left")
    print(f"  結合の結果: {len(perf)}行 -> {len(boom)}行 (渋谷店が2行に複製!)")
    total_ok = perf["actual_mil"].sum()
    total_boom = boom["actual_mil"].sum()
    print(f"  実績合計: 正常 {total_ok:,.1f} vs 汚染 {total_boom:,.1f} 百万ウォン"
          f" (+{total_boom - total_ok:,.1f} の水増し)")
    safe = pd.merge(perf, dup_targets.drop_duplicates(subset="store"),
                    on="store", how="left")
    print(f"  予防: drop_duplicates してから結合 -> {len(safe)}行 (正常に復帰)")
    print("  習慣: 結合の前に duplicated() の確認 + 結合の前後で len() の比較!")

    # ------------------------------------------------------------------
    print("\n[6] concat — 横につなぐのではなく「下に積む」")
    first_half = (df[df["day_index"] < 182].groupby("store", as_index=False)
                  .agg(revenue_mil=("revenue", "sum")))
    second_half = (df[df["day_index"] >= 182].groupby("store", as_index=False)
                   .agg(revenue_mil=("revenue", "sum")))
    first_half["half"], second_half["half"] = "上半期", "下半期"
    stacked = pd.concat([first_half, second_half], ignore_index=True)
    stacked["revenue_mil"] = (stacked["revenue_mil"] / 1e6).round(1)
    print(f"  上半期 {len(first_half)}行 + 下半期 {len(second_half)}行"
          f" = concat {len(stacked)}行 (同じ列構造を縦に積む)")
    print(stacked.head(3).to_string(index=False))
    print("\nまとめ: 基準となる表があるなら left、監査するときは outer+indicator、")
    print("      結合の前に重複キーを点検 — この3つが結合事故を防ぎます。")


if __name__ == "__main__":
    main()
