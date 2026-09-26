"""
Lecture 01 / Level 07 — GitHub — 远程仓库与协作

不联网实战远程协作。在临时文件夹里建一个 bare 仓库 (总部档案馆)
当作"假 GitHub"，自动执行开发者 A、B 用 clone -> push -> pull
互相传递文件的协作场景。
未安装 git 时，先给出安装指引，再用概念模拟代替。
"""

import shutil
import subprocess
import tempfile
from pathlib import Path


def run(args, cwd, actor="", note=""):
    """执行 git 命令，打印出'是谁($ 前的名字)做了什么'。"""
    label = f"[{actor}] " if actor else ""
    print(f"\n  {label}$ git {' '.join(args)}")
    if note:
        print(f"      解说: {note}")
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    output = (result.stdout + result.stderr).strip()
    for line in output.splitlines()[:8]:
        print(f"      | {line}")
    return output


def setup_identity(repo, name):
    """登记实战用的提交身份 (仅限本仓库内有效)。"""
    run(["config", "user.name", name], repo)
    run(["config", "user.email", f"{name}@example.com"], repo)


def list_files(repo, actor):
    """展示分公司档案柜 (工作文件夹) 的文件列表。"""
    files = sorted(p.name for p in Path(repo).iterdir() if p.is_file())
    print(f"      [{actor}] 分公司文件夹内容: {files if files else '(空)'}")


def real_remote_demo():
    """用 bare 远程 + 两家分公司(A、B)实战 push/pull 协作流程。"""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        hq = base / "headquarters.git"   # 总部中央档案馆 (bare)
        dev_a = base / "dev_a"           # 开发者 A 的分公司
        dev_b = base / "dev_b"           # 开发者 B 的分公司

        # [1] 成立总部: bare 仓库 = 没有办公桌、只保管历史 (GitHub 服务器的真实形态)
        print("\n[1] 成立总部 — 创建 bare 远程仓库")
        run(["init", "--bare", "-b", "main", str(hq)], base,
            note="--bare: 没有工作文件夹的中央档案馆。这就是我们的'假 GitHub'")

        # [2] 开设分公司: A 和 B 各自 clone
        print("\n[2] 开设分公司 — 开发者 A、B 各自 clone")
        run(["clone", str(hq), str(dev_a)], base, "A", "把总部档案馆整个复制 (目前还是空的)")
        run(["clone", str(hq), str(dev_b)], base, "B", "B 也开设自己的分公司")
        setup_identity(dev_a, "dev-a")
        setup_identity(dev_b, "dev-b")

        # [3] A 的 push: 建文件 -> 提交 -> 送上总部
        print("\n[3] A 的工作与 push")
        (dev_a / "menu.txt").write_text("美式咖啡 4000韩元\n", encoding="utf-8")
        run(["add", "menu.txt"], dev_a, "A")
        run(["commit", "-m", "菜单初稿"], dev_a, "A", "分公司审批(提交)目前只存在于 A 的电脑里")
        run(["push", "origin", "main"], dev_a, "A", "push = 把审批文件呈送总部档案馆")

        # [4] B 的 pull: 取回总部的最新文件 — 协作的核心时刻
        print("\n[4] B 的 pull — A 的工作送达 B")
        list_files(dev_b, "B(pull 前)")
        run(["pull", "origin", "main"], dev_b, "B", "pull = 取回总部的新文件并合并")
        list_files(dev_b, "B(pull 后)")
        print("      -> A 创建的 menu.txt 出现在了 B 的分公司!")

        # [5] 你来我往: B 追加 -> push, A 用 pull 更新
        print("\n[5] 反方向 — B 追加、A 接收")
        menu_b = dev_b / "menu.txt"
        menu_b.write_text(menu_b.read_text(encoding="utf-8") + "拿铁 4500韩元\n", encoding="utf-8")
        run(["add", "."], dev_b, "B")
        run(["commit", "-m", "追加拿铁"], dev_b, "B")
        run(["push", "origin", "main"], dev_b, "B")
        run(["pull", "origin", "main"], dev_a, "A", "先 pull、后 push — 协作礼仪")
        print(f"      [A] 最终菜单: {(dev_a / 'menu.txt').read_text(encoding='utf-8').splitlines()}")
        print("      -> 两家分公司达到了同一个最新状态。")

        # [6] 如果是真 GitHub: PR 流程解说
        print("\n[6] 换成真正的 GitHub 协作，这个流程里会插进'评审':")
        for step in [
            "1) 开分支干活后 push",
            "2) 在 GitHub 网页上撰写 Pull Request (签报)",
            "3) 同事评审 — 留言、修改要求来回往复",
            "4) 批准后 merge 进 main (并入正传)",
            "5) 全体成员 git pull 更新",
        ]:
            print(f"      {step}")


def fallback_simulation():
    """没装 git 的情况: 只用文字模拟概念。"""
    print("\n[提示] 没有找到 git 命令。")
    print("  macOS: 'xcode-select --install' / Windows: 安装 git-scm.com 后重新运行。")
    print("\n[概念模拟] 远程协作的一天:")
    events = [
        ("总部", "开设中央档案馆 (bare 仓库)"),
        ("A", "clone — 整馆复制、开设分公司"),
        ("A", "提交后 push — 把审批文件呈送总部"),
        ("B", "pull — 取回总部的新文件、更新分公司"),
        ("B", "追加工作后 push, A 再 pull"),
    ]
    for actor, action in events:
        print(f"  [{actor:^4}] {action}")
    print("\n  地址是互联网 URL (GitHub) 还是本地文件夹，原理都一样。")


def main():
    print("=" * 60)
    print("远程仓库协作实战 — 总部(远程)与两家分公司(A、B)")
    print("=" * 60)
    if shutil.which("git"):
        real_remote_demo()
    else:
        fallback_simulation()

    print("\n小结: clone 开局, push 上传, pull 接收。")
    print("      先 pull、小步勤 push, 是减少冲突的协作礼仪。")


if __name__ == "__main__":
    main()
