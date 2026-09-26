"""
从 Excel 到 Python — 把销售表存成 CSV 再重新读回来，
然后用标准库代码重现你在 Excel 里做的 SUM、SUMIF、自动筛选和排序。
让代码自己汇报跳过了多少条缺失、负数污染，
从而体会"可验证的流程"这种代码形态，以及可复现性的价值。
"""

import csv
import os
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

BASE_DIR = pathlib.Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"


def load_clean_rows(csv_path: str) -> tuple[list[dict], int, int]:
    """读取 CSV，并把 revenue 还原成数字。
    缺失 (空字符串)、负数的行会被跳过，跳过的条数一并返回。"""
    clean, n_missing, n_negative = [], 0, 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            raw = r["revenue"]
            if raw == "" or raw == "None":          # 缺失 → 跳过
                n_missing += 1
                continue
            revenue = int(raw)
            if revenue < 0:                          # 负数污染 → 跳过
                n_negative += 1
                continue
            r["revenue"] = revenue                   # 类型还原 (str → int)
            r["ad_cost"] = int(r["ad_cost"])
            clean.append(r)
    return clean, n_missing, n_negative


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1] 把 60 天的销售表存成 CSV (Excel 也能打开的格式)")
    rows = hjh_data.sales_table(n_days=60, seed=42)
    csv_path = hjh_data.to_csv(rows, str(OUT_DIR / "sales.csv"))
    print(f"    保存完成: {csv_path} ({len(rows)} 行)")
    print()

    print("[2] 重新读取 CSV — 所有值都是以\"字符串\"进来的")
    with open(csv_path, newline="", encoding="utf-8") as f:
        first = next(csv.DictReader(f))
    print(f"    第一行: {first}")
    print(f"    revenue 的类型: {type(first['revenue']).__name__} ← 这不是数字!")
    print()

    print("[3] 类型还原 + 跳过污染行 (丢掉的东西一定要汇报)")
    clean, n_missing, n_negative = load_clean_rows(csv_path)
    print(f"    使用 {len(clean)} 行 / 跳过缺失 {n_missing} 条 / 跳过负数 {n_negative} 条")
    print()

    print("[4] SUM — 相当于 Excel 的 =SUM(G:G)")
    total = sum(r["revenue"] for r in clean)
    print(f"    60 天销售额总合计: {total:,} 韩元")
    print()

    print("[5] SUMIF — 各门店合计 (用字典累加)")
    by_store: dict[str, int] = {}
    for r in clean:
        by_store[r["store"]] = by_store.get(r["store"], 0) + r["revenue"]
    for store, subtotal in sorted(by_store.items(), key=lambda kv: kv[1]):
        print(f"    {store}: {subtotal:>15,} 韩元")
    print()

    print("[6] 自动筛选 — 只挑\"朝阳店 & 周末\"的行，和工作日做对比")
    gangnam = [r for r in clean if r["store"] == "朝阳店"]
    weekend = [r for r in gangnam if r["weekday"] in ("周六", "周日")]
    weekday_rows = [r for r in gangnam if r["weekday"] not in ("周六", "周日")]
    avg_weekend = sum(r["revenue"] for r in weekend) / len(weekend)
    avg_weekday = sum(r["revenue"] for r in weekday_rows) / len(weekday_rows)
    print(f"    朝阳店周末平均: {avg_weekend:>12,.0f} 韩元 ({len(weekend)} 行)")
    print(f"    朝阳店工作日平均: {avg_weekday:>12,.0f} 韩元 ({len(weekday_rows)} 行)")
    print(f"    → 周末是工作日的 {avg_weekend / avg_weekday:.2f} 倍。")
    print()

    print("[7] 排序 — 销售额前 5 名 (相当于 Excel 的降序排序)")
    top5 = sorted(clean, key=lambda r: r["revenue"], reverse=True)[:5]
    for i, r in enumerate(top5, 1):
        print(f"    第{i}名 | {r['date']} {r['weekday']} | {r['store']} "
              f"{r['category']} | {r['revenue']:,} 韩元")
    print()

    print("[8] 可复现性 — 这个脚本真正的价值")
    print("    你刚才看到的所有数字，都出自\"固定 seed + 被记录下来的流程\"。")
    print("    明天再跑一次、换别人来跑，结果都完全一样。")
    print("    Excel 的点击只留在记忆里，而代码会以文档的形式留下来。")


if __name__ == "__main__":
    main()
