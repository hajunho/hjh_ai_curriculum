"""
Lecture 01 / Level 02 — 终端第一步

用 Python 写的"迷你 Shell 模拟器"，在安全的临时文件夹里
体验 pwd/ls/cd/mkdir/cp/mv/rm 命令的工作原理。
通过代码结构验证: Shell 归根结底是
"记住当前位置 (cwd)、代你完成文件操作的翻译官"。
"""

import shutil
import tempfile
from pathlib import Path


class MiniShell:
    """模仿真 Shell 核心行为 (管理当前位置 + 文件命令) 的类。"""

    def __init__(self, root: Path):
        # resolve(): 解析掉符号链接后的绝对路径 (应对 macOS 的 /var -> /private/var)
        self.root = root.resolve()   # 练习场的最顶层 (出不了这里)
        self.cwd = self.root      # 当前工作目录 — Shell 的核心状态!

    def run(self, line: str) -> str:
        """解析 'cp a.txt backup' 这样的一行，调用对应的方法。"""
        parts = line.split()
        command, args = parts[0], parts[1:]
        handler = getattr(self, f"cmd_{command}", None)  # cmd_ls, cmd_cd ...
        if handler is None:
            return f"(错误) '{command}': 找不到该命令"
        return handler(*args)

    # ---- 各命令的实现: 归根结底都是调用 pathlib 函数 ----
    def cmd_pwd(self):
        rel = self.cwd.relative_to(self.root)
        return "/" + str(rel) if str(rel) != "." else "/ (练习场最顶层)"

    def cmd_ls(self):
        names = sorted(p.name + ("/" if p.is_dir() else "") for p in self.cwd.iterdir())
        return "  ".join(names) if names else "(空)"

    def cmd_cd(self, name):
        target = (self.cwd / name).resolve()
        if not target.is_dir():
            return f"(错误) '{name}': 没有这个文件夹"
        self.cwd = target  # 所谓"移动"，只是改一下当前位置这个变量!
        return f"移动完成 -> {self.cmd_pwd()}"

    def cmd_mkdir(self, name):
        (self.cwd / name).mkdir()
        return f"已创建文件夹 '{name}'"

    def cmd_cp(self, src, dst):
        dst_path = self.cwd / dst
        if dst_path.is_dir():
            dst_path = dst_path / src  # 复制进文件夹时保持同名
        shutil.copy(self.cwd / src, dst_path)
        return f"'{src}' -> '{dst}' 复制完成 (原件不动)"

    def cmd_mv(self, src, dst):
        dst_path = self.cwd / dst
        if dst_path.is_dir():
            dst_path = dst_path / src
        (self.cwd / src).rename(dst_path)
        return f"'{src}' -> '{dst}' 移动完成 (同一文件夹内即为重命名)"

    def cmd_rm(self, name):
        (self.cwd / name).unlink()
        return f"'{name}' 已删除 — 不进回收站，直接销毁!"


# 练习场景: (命令, 这条命令做的事的解说)
SCENARIO = [
    ("pwd", "先确认我现在站在哪 — 终端操作的第一习惯"),
    ("ls", "看看这个房间(文件夹)里有什么"),
    ("mkdir backup", "装一个新的备份文件柜(文件夹)"),
    ("cp 周报.txt backup", "把报告复印一份存进 backup 文件夹"),
    ("cd backup", "走进 backup 文件夹"),
    ("ls", "确认副本是否放好了"),
    ("mv 周报.txt 周报_备份版.txt", "同一文件夹内的 mv = 重命名"),
    ("cd ..", "'..' = 上一层(父文件夹)"),
    ("rm 临时便签.txt", "删掉不再需要的便签 (立即·永久!)"),
    ("ls", "看看整理完的房间的最终状态"),
]


def main():
    print("=" * 60)
    print("终端第一步 — 迷你 Shell 模拟器")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        office = Path(tmp) / "practice_office"
        office.mkdir()

        # [1] 准备练习场: 铺好示例文件 (临时文件夹，失手也安全)
        for name, content in [
            ("周报.txt", "本周销售摘要"),
            ("会议纪要.txt", "9月例会"),
            ("临时便签.txt", "午饭菜单候选"),
        ]:
            (office / name).write_text(content, encoding="utf-8")
        print(f"\n[1] 练习场准备完毕: 已在临时文件夹里创建 3 个示例文件")
        print("    (完全不会碰你电脑上的真实文件)")

        # [2] 执行场景: 一条条命令按"输入 -> 执行 -> 解说"展示
        shell = MiniShell(office)
        print("\n[2] 执行命令场景 ($ 标记的是'输入的命令')")
        for step, (line, note) in enumerate(SCENARIO, start=1):
            print(f"\n  ({step}) $ {line}")
            print(f"      -> {shell.run(line)}")
            print(f"      解说: {note}")

        # [3] 最终状态与小结
        n_ops = len(SCENARIO)
        print("\n[3] 最终文件夹结构:")
        for p in sorted(office.rglob("*")):
            depth = len(p.relative_to(office).parts) - 1
            tag = "/" if p.is_dir() else ""
            print("      " + "  " * depth + f"- {p.name}{tag}")
        print(f"\n    {n_ops} 行命令就搞定了。要是用鼠标: 开窗口、拖拽、点名字改名…")
        print("    而且这 {n} 行存下来，明天可以照跑，接手的同事也能照跑。".format(n=n_ops))

    print("\n小结: Shell = 记住'当前位置'、代你完成文件操作的翻译官。")
    print("      现在就打开真正的终端，从 pwd、ls、cd 开始亲手敲吧!")


if __name__ == "__main__":
    main()
