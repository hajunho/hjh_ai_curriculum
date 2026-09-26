"""
数据到底是什么 — 用字典列表亲手解剖表格的结构。
先用眼睛确认"行 (row)=事例、列 (column)=属性"这条原理，
再只用标准库完成取列、找行、观察类型与缺失、做迷你 schema 摘要。
最后与非结构化文本做对比，看看"表"为什么在聚合上更有优势。
"""

import pathlib
import sys

# 为了导入公用数据模块 (hjh_data) 而设置路径
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data


def extract_column(rows: list[dict], col: str) -> list:
    """从表里把一列抽成列表。(列 = 同一个属性的值的集合)"""
    return [r[col] for r in rows]


def summarize_schema(rows: list[dict]) -> list[dict]:
    """逐列统计类型构成和缺失个数，做出一份"迷你 schema 摘要"。"""
    summary = []
    for col in rows[0].keys():
        values = extract_column(rows, col)
        none_count = sum(1 for v in values if v is None)
        # 收集除 None 以外的那些值的类型名
        type_names = sorted({type(v).__name__ for v in values if v is not None})
        summary.append({
            "column": col,
            "types": "/".join(type_names),
            "missing": none_count,
            "example": next(v for v in values if v is not None),
        })
    return summary


def main() -> None:
    # 生成 30 天的虚构咖啡店销售表 (固定 seed → 每次结果都一样)
    rows = hjh_data.sales_table(n_days=30, seed=42)

    print("[1] 表的大小与 schema (列名)")
    print(f"    行 (事例) 数: {len(rows)}")
    print(f"    列 (属性) 清单: {list(rows[0].keys())}")
    print()

    print("[2] 表格预览 — 拿到表先用眼睛看一遍，没有例外")
    hjh_data.head(rows, n=5)
    print()

    print("[3] 解剖一行 — 这张表的一行到底是\"一件什么事\"?")
    first = rows[0]
    for key, value in first.items():
        print(f"    {key:>10} = {value!r}  ({type(value).__name__})")
    print("    → 一行 = 某个日期、某家门店、某个品类的\"当日销售额\"1 条。")
    print()

    print("[4] 取列 — 只把 revenue 这一列抽出来")
    revenues = extract_column(rows, "revenue")
    print(f"    revenue 列的长度: {len(revenues)} (和行数相同)")
    print(f"    前 8 个值: {revenues[:8]}")
    print()

    print("[5] 按条件找行 — 只要朝阳店的咖啡销售额")
    gangnam_coffee = [r for r in rows
                     if r["store"] == "朝阳店" and r["category"] == "咖啡"]
    print(f"    符合条件的行: {len(gangnam_coffee)} 个 (30 天，所以 30 个才正常)")
    hjh_data.head(gangnam_coffee, n=3)
    print()

    print("[6] 迷你 schema 摘要 — 各列的类型与缺失 (None) 个数")
    schema = summarize_schema(rows)
    for s in schema:
        print(f"    {s['column']:>10} | 类型: {s['types']:<8} | "
              f"缺失: {s['missing']:>2} 个 | 示例: {s['example']!r}")
    missing_total = sum(s["missing"] for s in schema)
    negative_count = sum(1 for v in revenues if v is not None and v < 0)
    print(f"    → 全表藏着 {missing_total} 个缺失格、{negative_count} 条负数销售额。")
    print("      (这是故意埋下的污染。level06 会学怎么处理)")
    print()

    print("[7] 结构化 vs 非结构化 — 同样的信息，不同的形状")
    unstructured = "昨天去了朝阳店，咖啡挺好喝的。人很多，估计营业额不错!"
    structured = {"date": "2025-01-01", "store": "朝阳店",
                  "category": "咖啡", "revenue": 512000}
    print(f"    非结构化 (自由句子): {unstructured!r}")
    print(f"    结构化 (表的 1 行): {structured!r}")
    print("    → 句子上没法套 SUM，但表的 revenue 列可以直接加起来。")
    print("      非结构化数据要分析，最终也得经过变成表的这一步。")


if __name__ == "__main__":
    main()
