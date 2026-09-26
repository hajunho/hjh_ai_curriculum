"""
Lecture 01 / Level 03 — 安装 Python 与第一次运行

给自己电脑的 Python 环境做"体检"的自检脚本。
确认版本、可执行文件位置、操作系统和 PATH，
并用输出对比 REPL (交互式) 与脚本运行的区别。
"""

import os
import platform
import sys

# 本课程要求的最低 Python 版本 ("动手试试"第 2 题请改这里)
REQUIRED = (3, 10)


def check_python_identity():
    """[1] 核对翻译官身份: 版本与可执行文件位置。"""
    version = sys.version_info  # (major, minor, micro, ...) 可以按数字比较
    ok = version >= REQUIRED
    print("[1] Python 身份核对")
    print(f"    版本            : {platform.python_version()}")
    print(f"    可执行文件位置  : {sys.executable}")
    print(f"    要求版本        : {REQUIRED[0]}.{REQUIRED[1]} 以上")
    print(f"    判定            : {'通过 — 可以进行本课程' if ok else '不足 — 需要升级'}")
    return ok


def check_workplace():
    """[2] 确认工作地点: 操作系统与当前工作目录。"""
    os_name = platform.system()  # 'Darwin'(macOS) / 'Windows' / 'Linux'
    friendly = {"Darwin": "macOS", "Windows": "Windows", "Linux": "Linux"}.get(os_name, os_name)
    print("\n[2] 工作环境确认")
    print(f"    操作系统        : {friendly} ({platform.release()})")
    print(f"    处理器          : {platform.machine()}")
    print(f"    当前工作文件夹  : {os.getcwd()}")
    print("    -> 相对路径全都以这个文件夹为基准解析 (level01 复习)")
    return True


def check_path():
    """[3] 扫一眼 PATH: Shell 查找命令的文件夹清单。"""
    raw = os.environ.get("PATH", "")
    entries = [e for e in raw.split(os.pathsep) if e]
    print("\n[3] PATH — Shell 查找 'python3' 这类命令时翻的文件夹清单")
    print(f"    登记的文件夹数  : {len(entries)}个 (从前往后按顺序查找)")
    for i, entry in enumerate(entries[:5], start=1):
        print(f"    第{i}位: {entry}")
    if len(entries) > 5:
        print(f"    ... 另有 {len(entries) - 5} 个")
    print("    -> 'command not found' = 这份清单里哪儿都没有那个名字")
    return len(entries) > 0


def compare_repl_vs_script():
    """[4] 对比 REPL (即时对话) vs 脚本 (书面指示)。"""
    print("\n[4] 两种运行方式对比")
    print("    (a) REPL — 在终端只敲 python3 就出现的 '>>>' 里即时对话:")
    print("        >>> 120 * 12")
    print(f"        {120 * 12}")
    print("        >>> '报告' + '_最终版.xlsx'")
    print(f"        '{'报告' + '_最终版.xlsx'}'")
    print("    (b) 脚本 — 就是现在! 你看到的这整份输出，正是把 main.py 这份文件")
    print("        整个递给 python3、让它从上到下执行的结果。")
    print("    -> 做实验用 REPL，干正事用脚本 (可保存、共享、重跑)")
    return True


def main():
    print("=" * 60)
    print("Python 环境自检 — 我的翻译官准备好了吗")
    print("=" * 60 + "\n")

    results = {
        "Python 版本": check_python_identity(),
        "操作系统确认": check_workplace(),
        "PATH 设置": check_path(),
        "理解运行方式": compare_repl_vs_script(),
    }

    # [5] 综合判定表
    print("\n[5] 综合诊断摘要")
    for item, ok in results.items():
        mark = "[OK] " if ok else "[注意]"
        print(f"    {mark} {item}")
    if all(results.values()):
        print("\n    恭喜! 这台电脑已经做好进行本课程的准备。")
    else:
        print("\n    请解决 [注意] 项目后再运行一次。")

    print("\n小结: 环境不对劲时，永远先查 sys.executable (是哪个 Python)")
    print("      和 python3 --version (是什么版本)，把这变成习惯。")


if __name__ == "__main__":
    main()
