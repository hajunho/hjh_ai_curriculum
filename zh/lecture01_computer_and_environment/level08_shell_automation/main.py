"""
Lecture 01 / Level 08 — 用 Shell 脚本自动化重复工作

自动整理乱成一团的共享文件夹 (在临时文件夹里复现) 的实战。
沿着自动化的标准模式"收集 -> 判断 -> (空跑) -> 执行 -> 汇报"，
用正则表达式解析文件名，按门店、年月批量移动进文件夹，
并把文件名改成统一规则。
"""

import random
import re
import tempfile
from pathlib import Path

# 用来复现乱糟糟共享文件夹的文件们 — 命名规则五花八门!
MESSY_FILES = [
    "销售报告_朝阳_2024-03.csv", "销售报告_朝阳_2024-04.csv",
    "销售报告_海淀_2024-03.csv", "sales_pudong_2024-03.csv",
    "sales_pudong_2024-04.csv", "销售报告_天河_2024-04.csv",
    "sales_tianhe_2024-03.csv", "销售报告_海淀_2024-04.csv",
    "备忘录.txt", "午餐投票.txt", "演示文稿.pptx", "旧备份.zip",
    "销售报告_朝阳_2024-05.csv", "sales_pudong_2024-05.csv",
]

# 文件名解析规则: 抽取'门店'和'年-月'的两条正则表达式
PATTERNS = [
    re.compile(r"销售报告_(?P<store>.+)_(?P<ym>\d{4}-\d{2})\.csv"),
    re.compile(r"sales_(?P<store>.+)_(?P<ym>\d{4}-\d{2})\.csv"),
]
# 英文门店名 -> 统一为中文
STORE_ZH = {"pudong": "浦东", "tianhe": "天河"}


def create_mess(folder: Path):
    """[1] 制造烂摊子: 用固定 seed 的随机内容生成五花八门的文件。"""
    rng = random.Random(42)  # 固定 seed — 跑多少遍内容都一样
    for name in MESSY_FILES:
        revenue = rng.randint(500, 2000)
        (folder / name).write_text(f"revenue,{revenue}000\n", encoding="utf-8")


def plan_moves(folder: Path):
    """[2] 收集与判断: 只为每个文件制定'送去哪、改什么名'的计划。

    和执行分开写，空跑 (彩排) 就白送了。"""
    plans = []  # (源 Path, 目的地 Path, 分类理由)
    for path in sorted(folder.glob("*")):        # 收集: 目标清单
        if path.is_dir():
            continue
        for pattern in PATTERNS:                 # 判断: 应用规则
            matched = pattern.match(path.name)
            if matched:
                store = STORE_ZH.get(matched["store"], matched["store"])
                ym = matched["ym"]
                # 目的地: 门店文件夹/门店_年-月.csv，连名字一起统一
                dest = folder / store / f"{store}_{ym}.csv"
                plans.append((path, dest, f"销售文件 -> {store}/{ym}"))
                break
        else:  # 哪条规则都不匹配的，进人工核查箱
            dest = folder / "_待核查" / path.name
            plans.append((path, dest, "规则不匹配 -> 核查箱"))
    return plans


def execute_moves(plans):
    """[4] 执行: 按计划表建文件夹、批量移动。"""
    moved = 0
    for src, dest, _ in plans:
        dest.parent.mkdir(parents=True, exist_ok=True)  # Shell 的 mkdir -p
        if dest.exists():                                # 防止覆盖事故!
            dest = dest.with_name(dest.stem + "_重复" + dest.suffix)
        src.rename(dest)                                 # Shell 的 mv
        moved += 1
    return moved


def draw_tree(folder: Path):
    """把整理结果打印成树状图。"""
    for p in sorted(folder.rglob("*")):
        depth = len(p.relative_to(folder).parts) - 1
        tag = "/" if p.is_dir() else ""
        print("      " + "  " * depth + f"- {p.name}{tag}")


def main():
    print("=" * 60)
    print("文件整理自动化 — 收集 -> 判断 -> 空跑 -> 执行 -> 汇报")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        shared = Path(tmp) / "shared_folder"
        shared.mkdir()

        create_mess(shared)
        print(f"\n[1] 复现烂摊子: 生成命名规则五花八门的文件 {len(MESSY_FILES)} 个")
        print("    " + ", ".join(MESSY_FILES[:6]) + " ...")

        plans = plan_moves(shared)
        print(f"\n[2] 收集与判断: 用正则表达式解析文件名，制定计划 {len(plans)} 件")

        # [3] 空跑: 执行前只用眼睛检查计划表 — 自动化的第一道安全装置
        print("\n[3] 空跑(彩排) — 目前什么都还没搬")
        for src, dest, reason in plans:
            print(f"    {src.name:<28} -> {dest.relative_to(shared)}  ({reason})")

        moved = execute_moves(plans)
        print(f"\n[4] 真正执行: 完成 {moved} 件移动与文件名统一")

        print("\n[5] 汇报结果 — 整理后的文件夹结构:")
        draw_tree(shared)
        review = sum(1 for _, dest, _ in plans if "_待核查" in str(dest))
        print(f"\n    摘要: 自动分类 {moved - review} 件 / 需人工核查 {review} 件")
        print("    手工做的话每个文件 30 秒 x 14 个 = 约 7 分钟,")
        print("    脚本 0.1 秒 — 而且每周再让它做一遍也是免费的。")

    print("\n小结: 把计划 (plan) 和执行 (execute) 分开，空跑就白送了。")
    print("      批量操作务必按'检查计划表 -> 执行'的顺序!")


if __name__ == "__main__":
    main()
