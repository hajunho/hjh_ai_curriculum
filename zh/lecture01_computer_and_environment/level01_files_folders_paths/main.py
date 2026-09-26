"""
Lecture 01 / Level 01 — 文件、文件夹与路径的概念

用 pathlib 创建并探索虚构公司"晨光商社"的文档文件夹树。
在一个临时文件夹里安全地练习: 绝对路径/相对路径的差别、
按扩展名统计文件、解剖路径 (parent/stem/suffix)。
结束时临时文件夹会自动删除。
"""

import os
import tempfile
from collections import Counter
from pathlib import Path


def build_company_tree(root: Path):
    """[创建树] 建立部门文件夹和示例文档文件。"""
    # 文件夹结构: (相对路径) — 用 mkdir(parents=True) 连中间层级一次建好
    folders = [
        "行政部",
        "销售部/合同",
        "销售部/报表",
        "研发部/代码",
    ]
    for name in folders:
        (root / name).mkdir(parents=True, exist_ok=True)

    # 文件: (相对路径, 内容) — 扩展名是"文档类型印章"。
    files = [
        ("行政部/办公用品申请单.txt", "圆珠笔 10 支, A4 纸 2 箱"),
        ("行政部/停车规定.txt", "地下 2 层为访客专用。"),
        ("销售部/合同/A公司_合同.txt", "甲方: A公司 / 乙方: 晨光商社"),
        ("销售部/合同/B公司_合同.txt", "甲方: B公司 / 乙方: 晨光商社"),
        ("销售部/报表/3月_业绩.csv", "store,revenue\n朝阳,1512000\n海淀,1098000"),
        ("销售部/报表/4月_业绩.csv", "store,revenue\n朝阳,1620000\n海淀,1150000"),
        ("研发部/代码/hello.py", "print('hello')"),
    ]
    for rel_path, content in files:
        (root / rel_path).write_text(content, encoding="utf-8")
    return len(folders), len(files)


def draw_tree(folder: Path, indent: int = 0):
    """[画出树] 用递归调用把文件夹结构打印成带缩进的树状图。"""
    marker = "📁" if indent == 0 else "└─"
    print("    " + "   " * indent + f"{marker} {folder.name}/")
    for child in sorted(folder.iterdir()):
        if child.is_dir():
            draw_tree(child, indent + 1)  # 是文件夹就再调用自己 (递归)
        else:
            print("    " + "   " * (indent + 1) + f"└─ {child.name}")


def main():
    print("=" * 60)
    print("文件、文件夹与路径实战 — 探秘晨光商社档案库")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "chenguang_corp"
        root.mkdir()

        # [1] 创建文件夹树
        n_folders, n_files = build_company_tree(root)
        print(f"\n[1] 创建树: 建立了 {n_folders} 个文件夹(+中间层级)、{n_files} 个文件")
        print(f"    位置(临时文件夹): {root}")

        # [2] 画出树 — 用 Python 模仿终端的 tree 命令
        print("\n[2] 档案库整体结构:")
        draw_tree(root)

        # [3] 绝对路径 vs 相对路径
        target = root / "销售部" / "报表" / "3月_业绩.csv"
        print("\n[3] 指向同一个文件的两种地址:")
        print(f"    绝对路径: {target.resolve()}")
        print(f"    相对路径(以公司正门为基准): {target.relative_to(root)}")
        # '..' 表示上一层 (父文件夹)。
        sibling = target.parent.parent / "合同" / "A公司_合同.txt"
        print(f"    从报表文件夹走 '../合同/A公司_合同.txt' 会到:")
        print(f"      -> {sibling.relative_to(root)} (存在? {sibling.exists()})")
        print(f"    当前工作目录(CWD): {os.getcwd()}")
        print("      -> 相对路径总是以这个 CWD(或指定的基准)来解析")

        # [4] 按扩展名统计文件 — rglob 会把子文件夹也翻个遍
        counts = Counter(p.suffix for p in root.rglob("*") if p.is_file())
        print("\n[4] 按扩展名(文档类型印章)统计:")
        for ext, count in sorted(counts.items()):
            print(f"    {ext:<6} {count}个")

        # [5] 解剖路径 — 把一条路径按部位拆开
        print("\n[5] 解剖路径: 销售部/报表/3月_业绩.csv")
        print(f"    parent (所在文件夹)   : {target.parent.name}/")
        print(f"    name   (完整文件名)   : {target.name}")
        print(f"    stem   (去掉扩展名)   : {target.stem}")
        print(f"    suffix (扩展名)       : {target.suffix}")

    print("\n[6] 实战结束: 临时文件夹已自动删除 — 电脑上没有留下痕迹")
    print("小结: 绝对路径是从正门写全的完整地址，相对路径是以当前位置为基准的简写地址。")


if __name__ == "__main__":
    main()
