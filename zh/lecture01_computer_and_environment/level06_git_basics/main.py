"""
Lecture 01 / Level 06 — Git — 版本管理的概念

在临时文件夹里创建真正的 Git 仓库，自动实战 init -> add -> commit ->
diff -> log -> branch 全流程。用真实的命令输出验证
"提交=审批盖章、分支=平行宇宙"的比喻。
未安装 git 时，先给出安装指引，再用概念模拟代替。
"""

import shutil
import subprocess
import tempfile
from pathlib import Path


def run_git(args, cwd, note=""):
    """执行 git 命令，按 '$ 命令 -> 输出' 的形式展示。"""
    print(f"\n  $ git {' '.join(args)}")
    if note:
        print(f"    解说: {note}")
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    output = (result.stdout + result.stderr).strip()
    for line in output.splitlines()[:12]:  # 太长就只看前面
        print(f"    | {line}")
    return output


def real_git_demo():
    """装了 git 的情况: 在临时文件夹用真实命令实战全流程。"""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "cafe_project"
        repo.mkdir()
        price_file = repo / "价目表.txt"

        # [1] 装档案柜: init + 登记实战身份 (只在这个仓库里有效)
        print("\n[1] 装档案柜 — git init")
        run_git(["init", "-b", "main"], repo, "开始管理本文件夹的履历 (生成 .git 档案柜)")
        run_git(["config", "user.name", "实习生"], repo, "刻在章上的名字 (实战用, 仅限本仓库)")
        run_git(["config", "user.email", "student@example.com"], repo, "刻在章上的邮箱")

        # [2] 第一次审批: 建文件 -> add(审批夹) -> commit(盖章)
        print("\n[2] 第一次审批 — add 与 commit")
        price_file.write_text("美式咖啡 4000韩元\n拿铁 4500韩元\n", encoding="utf-8")
        print(f"    (创建文件: {price_file.name})")
        run_git(["status", "--short"], repo, "?? = 档案柜还不认识的新文件")
        run_git(["add", "价目表.txt"], repo, "放上审批夹 (暂存)")
        run_git(["commit", "-m", "起草价目表"], repo, "盖章! 这一刻被永久保存")

        # [3] 修改与 diff: 确认哪里变了，再做第二次提交
        print("\n[3] 修改与 diff — 查看变化的部分")
        price_file.write_text("美式咖啡 4200韩元\n拿铁 4500韩元\n", encoding="utf-8")
        print("    (把美式咖啡价格从 4000 -> 4200 修改)")
        run_git(["diff"], repo, "- 是旧内容, + 是新内容")
        run_git(["add", "."], repo)
        run_git(["commit", "-m", "美式咖啡提价(4000->4200)"], repo, "诀窍是在信息里写'为什么'")

        # [4] 查历史: 章一个个摞起来的审批记录
        print("\n[4] 查历史 — git log")
        run_git(["log", "--oneline"], repo, "前面的短代码是提交编号 (哈希)")

        # [5] 分支: 在平行宇宙做实验后回到正传
        print("\n[5] 分支 — 平行宇宙实验")
        run_git(["switch", "-c", "experiment"], repo, "开一个 experiment 宇宙并移动过去")
        price_file.write_text("美式咖啡 9900韩元 (实验!)\n拿铁 4500韩元\n", encoding="utf-8")
        run_git(["add", "."], repo)
        run_git(["commit", "-m", "实验: 高端定价策略"], repo)
        print(f"    experiment 宇宙的价目表: {price_file.read_text(encoding='utf-8').splitlines()[0]}")
        run_git(["switch", "main"], repo, "回到正传(main) — 时间旅行!")
        print(f"    main 宇宙的价目表      : {price_file.read_text(encoding='utf-8').splitlines()[0]}")
        print("    -> 同一个文件，换个分支内容就变回去了。")
        print("       实验失败就只扔掉 experiment 宇宙，正传安然无恙。")


def fallback_simulation():
    """没装 git 的情况: 安装指引 + 提交概念模拟。"""
    print("\n[提示] 没有找到 git 命令。")
    print("  安装: macOS 用 'xcode-select --install' 或 'brew install git',")
    print("        Windows 建议安装 git-scm.com 的 Git for Windows。")
    print("  装好后再运行本脚本，就能用真实命令实战。")
    print("\n[概念模拟] 提交 = 文件夹状态的快照 + 审批章")
    history = [
        ("a1f9c02", "起草价目表", {"价目表.txt": "美式咖啡 4000韩元"}),
        ("b7e3d11", "美式咖啡提价(4000->4200)", {"价目表.txt": "美式咖啡 4200韩元"}),
    ]
    for i, (commit_hash, message, snapshot) in enumerate(history, start=1):
        print(f"\n  提交 {i}: [{commit_hash}] \"{message}\"")
        for fname, content in snapshot.items():
            print(f"    保管的快照: {fname} -> '{content}'")
    print("\n  能把文件夹回退到任何一个盖章时间点，这就是 Git 的力量。")


def main():
    print("=" * 60)
    print("Git 实战 — 审批盖章(提交)与平行宇宙(分支)")
    print("=" * 60)
    if shutil.which("git"):
        print("\n(发现 git — 将在临时文件夹里创建真正的仓库进行实战。")
        print(" 完全不会碰你的其他文件夹和仓库。)")
        real_git_demo()
    else:
        fallback_simulation()

    print("\n小结: 小步、勤提交，信息里写'为什么'。")
    print("      提交过的东西，无论犯什么错都能找回来。")


if __name__ == "__main__":
    main()
