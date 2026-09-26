"""
Lecture 01 / Level 05 — 代码编辑器与 Jupyter

不用安装 Jupyter 就能体验笔记本"单元格执行"原理的模拟器。
用一个字典模仿内核 (kernel) 的记忆，
重现单元格共享记忆的样子，以及执行顺序错乱时出现的
NameError 和错位结果 (out-of-order 事故)。
"""

import io
from contextlib import redirect_stdout


class NotebookKernel:
    """Jupyter 内核的缩小模型: 揣着记忆 (变量们) 执行单元格代码。"""

    def __init__(self):
        self.memory = {}        # 内核的记忆 — 所有单元格共享这一份!
        self.run_counter = 0    # 单元格左边的 [n] 执行序号

    def restart(self):
        """重启内核: 记忆全部消失。"""
        self.memory = {}
        self.run_counter = 0

    def run_cell(self, code: str):
        """执行一个单元格，返回 (序号, 输出, 错误)。"""
        self.run_counter += 1
        buffer = io.StringIO()
        error = None
        try:
            with redirect_stdout(buffer):        # 截获 print 输出
                exec(code, self.memory)          # 在共享记忆之上执行
        except Exception as exc:                 # 错误也像笔记本一样只展示
            error = f"{type(exc).__name__}: {exc}"
        return self.run_counter, buffer.getvalue().rstrip(), error

    def visible_vars(self):
        """内核记忆里的用户变量列表 (排除内部项目)。"""
        return sorted(k for k in self.memory if not k.startswith("__"))


# 笔记本文档: (单元格标题, 单元格代码) — 按画面从上到下的顺序排列
CELLS = [
    ("单元格 1 — 准备数据", "revenues = [1512, 1098, 1745, 702]\nprint('各门店销售额(千韩元):', revenues)"),
    ("单元格 2 — 合计", "total = sum(revenues)\nprint('总销售额:', total, '千韩元')"),
    ("单元格 3 — 平均", "average = total / len(revenues)\nprint('平均销售额:', round(average, 1), '千韩元')"),
    ("单元格 4 — 汇报", "print(f'汇报: 共 {total}千韩元, 门店平均 {average:.1f}千韩元')"),
]


def show_run(kernel, index, note=""):
    """执行一个单元格，并像笔记本画面一样打印出来。"""
    title, code = CELLS[index]
    n, out, err = kernel.run_cell(code)
    print(f"\n  [{n}] {title}{note}")
    for line in code.splitlines():
        print(f"      | {line}")
    if err:
        print(f"      !! 错误: {err}")
    elif out:
        print(f"      => {out}")
    print(f"      (内核记忆: {kernel.visible_vars() or '空'})")


def main():
    print("=" * 60)
    print("迷你笔记本模拟器 — 单元格与内核的原理")
    print("=" * 60)

    kernel = NotebookKernel()

    # [1] 介绍笔记本构成
    print(f"\n[1] 笔记本构成: {len(CELLS)} 个单元格 (销售分析场景)")
    print("    所有单元格共享名为'内核'的同一份 Python 记忆")

    # [2] Run All: 从上往下按顺序 — 记忆随单元格一路累积
    print("\n[2] Run All — 从上往下按顺序执行")
    for i in range(len(CELLS)):
        show_run(kernel, i)

    # [3] 重现乱序事故: 重启后从中间的单元格开始执行会怎样?
    print("\n[3] 重现乱序事故 — 重启内核后从单元格 3 开始执行?")
    kernel.restart()
    print("    (内核已重启: 记忆全部清零)")
    show_run(kernel, 2, "  <- 跳过了上面的单元格!")
    print("      解说: 记忆里没有 total，所以 NameError — 笔记本事故第一名")

    print("\n    这次按与画面不同的 1 -> 3 -> 2 -> 4 顺序执行看看:")
    kernel.restart()
    for i, note in [(0, ""), (2, "  <- 比合计(单元格 2)还早!"), (1, ""), (3, "")]:
        show_run(kernel, i, note)
    print("      解说: 要么报错，要么错误的坑被后来补上。")
    print("            一旦画面上的代码顺序和内核的记忆错位，")
    print("            从上往下读文档的同事就无法复现。")

    # [4] 总结教训
    print("\n[4] 总结教训")
    print("    - 单元格编号 [n] 不是画面位置，而是'被执行的顺序'")
    print("    - 分享之前务必用 Restart & Run All 确认可复现性")
    print("    - 实验用笔记本，定下来的逻辑搬进 .py 脚本才是正道")


if __name__ == "__main__":
    main()
