"""
Lecture 01 / Level 04 — 虚拟环境与包管理

诊断正在运行这段代码的 Python 环境。
通过比较 sys.prefix 和 sys.base_prefix 判定是否在虚拟环境里，
用 importlib.metadata 列出已安装的包 (工具箱里的东西)，
并揭示 pip freeze 生成的 requirements.txt 的真面目。
"""

import importlib.metadata
import importlib.util
import sys


def check_which_toolbox():
    """[1] 是哪只工具箱: 判定在虚拟环境里还是系统 Python。"""
    prefix = sys.prefix            # 当前 Python 查找工具的空间
    base = sys.base_prefix         # 原始 (系统) Python 的空间
    in_venv = prefix != base       # 两者不同 = 系着专用工具箱
    print("[1] 现在系的是哪只工具箱")
    print(f"    sys.prefix      (正在使用的空间): {prefix}")
    print(f"    sys.base_prefix (原始空间)      : {base}")
    if in_venv:
        print("    判定: 在虚拟环境里 — 正佩戴项目专用工具箱")
    else:
        print("    判定: 系统 Python — 正在用公司公用工具柜")
        print("    (改用仓库的 .venv/bin/python 运行，判定就会不一样)")
    print(f"    可执行文件: {sys.executable}")
    return in_venv


def list_installed_packages(limit=15):
    """[2] 工具箱里的东西: 列出已安装的包名和版本。"""
    dists = sorted(
        ((d.metadata["Name"] or "?", d.version) for d in importlib.metadata.distributions()),
        key=lambda pair: pair[0].lower(),
    )
    print(f"\n[2] 此环境已安装的包: 共 {len(dists)} 个 (和 pip list 相同的信息)")
    for name, version in dists[:limit]:
        print(f"    - {name:<28} {version}")
    if len(dists) > limit:
        print(f"    ... 另有 {len(dists) - limit} 个")
    return dists


def check_standard_tools():
    """[3] 检查基本工具 (标准库): 不用安装就有的工具们。"""
    basics = ["json", "csv", "datetime", "sqlite3", "pathlib", "random"]
    print("\n[3] 基本工具检查 — 标准库不用安装、直接可用")
    for name in basics:
        found = importlib.util.find_spec(name) is not None
        print(f"    {'[OK]' if found else '[缺失]'} {name}")
    print("    -> 本课程的 lecture01~02 全程只用这些基本工具")


def preview_requirements(dists, limit=8):
    """[4] requirements.txt 预览: pip freeze 生成的物品清单。"""
    print("\n[4] requirements.txt 预览 — '名字==版本' 格式的物品清单")
    if not dists:
        print("    (没有已安装的包，清单是空的)")
        return
    for name, version in dists[:limit]:
        print(f"    {name}=={version}")
    if len(dists) > limit:
        print(f"    ... 另有 {len(dists) - limit} 行")
    print("    -> 只要递出这个文件，同事用 'pip install -r requirements.txt'")
    print("       就能在自己电脑上原样重现同样配置的工具箱")


def main():
    print("=" * 60)
    print("虚拟环境与包诊断 — 我现在系的是哪只工具箱")
    print("=" * 60 + "\n")

    in_venv = check_which_toolbox()
    dists = list_installed_packages()
    check_standard_tools()
    preview_requirements(dists)

    # [5] 摘要
    print("\n[5] 诊断摘要")
    print(f"    是否虚拟环境 : {'是 (.venv 一类)' if in_venv else '否 (系统 Python)'}")
    print(f"    已安装包     : {len(dists)}个")
    print("\n小结: 给每个项目建专用工具箱 (venv)，")
    print("      用 pip freeze 清单 (requirements.txt) 共享配置。")
    print("      '在我电脑上明明能跑'事故的一半在这里就被预防了。")


if __name__ == "__main__":
    main()
