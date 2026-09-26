"""
Lecture 01 / Level 05 — コードエディタと Jupyter

Jupyter をインストールせずに、ノートブックの「セル実行」の原理を体験する
シミュレータです。カーネル(kernel)の記憶を辞書ひとつで真似て、
セルが記憶を共有する様子と、実行順序がずれたときに起こる
NameError・ずれた結果(out-of-order 事故)を再現します。
"""

import io
from contextlib import redirect_stdout


class NotebookKernel:
    """Jupyter カーネルの縮小模型: 記憶(変数たち)を持ってセルのコードを実行します。"""

    def __init__(self):
        self.memory = {}        # カーネルの記憶 — すべてのセルがこのひとつを共有!
        self.run_counter = 0    # セルの左の [n] 実行順の番号

    def restart(self):
        """カーネル再起動: 記憶がすべて消えます。"""
        self.memory = {}
        self.run_counter = 0

    def run_cell(self, code: str):
        """セルをひとつ実行して (番号, 出力, エラー) を返します。"""
        self.run_counter += 1
        buffer = io.StringIO()
        error = None
        try:
            with redirect_stdout(buffer):        # print の出力を横取り
                exec(code, self.memory)          # 共有の記憶の上で実行
        except Exception as exc:                 # エラーもノートブックのように表示だけ
            error = f"{type(exc).__name__}: {exc}"
        return self.run_counter, buffer.getvalue().rstrip(), error

    def visible_vars(self):
        """カーネルの記憶の中のユーザー変数一覧 (内部項目を除く)。"""
        return sorted(k for k in self.memory if not k.startswith("__"))


# ノートブック文書: (セルのタイトル, セルのコード) — 画面に上から下へ並んだ順序
CELLS = [
    ("セル 1 — データ準備", "revenues = [1512, 1098, 1745, 702]\nprint('店舗売上(千ウォン):', revenues)"),
    ("セル 2 — 合計", "total = sum(revenues)\nprint('総売上:', total, '千ウォン')"),
    ("セル 3 — 平均", "average = total / len(revenues)\nprint('平均売上:', round(average, 1), '千ウォン')"),
    ("セル 4 — 報告", "print(f'報告: 総 {total}千ウォン, 店舗平均 {average:.1f}千ウォン')"),
]


def show_run(kernel, index, note=""):
    """セルをひとつ実行し、ノートブックの画面のように出力します。"""
    title, code = CELLS[index]
    n, out, err = kernel.run_cell(code)
    print(f"\n  [{n}] {title}{note}")
    for line in code.splitlines():
        print(f"      | {line}")
    if err:
        print(f"      !! エラー: {err}")
    elif out:
        print(f"      => {out}")
    print(f"      (カーネルの記憶: {kernel.visible_vars() or '空です'})")


def main():
    print("=" * 60)
    print("ミニノートブックシミュレータ — セルとカーネルの原理")
    print("=" * 60)

    kernel = NotebookKernel()

    # [1] ノートブック構成の紹介
    print(f"\n[1] ノートブックの構成: セル {len(CELLS)}個 (売上分析シナリオ)")
    print("    すべてのセルは「カーネル」というひとつの Python の記憶を共有します")

    # [2] Run All: 上から順番に — 記憶がセルを経るごとに積み上がります
    print("\n[2] Run All — 上から順番に実行")
    for i in range(len(CELLS)):
        show_run(kernel, i)

    # [3] 順序事故の再現: 再起動後に途中のセルから実行すると?
    print("\n[3] 順序事故の再現 — カーネル再起動後にセル 3 から実行すると?")
    kernel.restart()
    print("    (カーネル再起動: 記憶がすべて初期化されました)")
    show_run(kernel, 2, "  <- 上のセルたちを飛ばした!")
    print("      解説: total が記憶に無いので NameError — ノートブック事故の第1位です")

    print("\n    今度は画面の順序と違って 1 -> 3 -> 2 -> 4 の順に実行してみると:")
    kernel.restart()
    for i, note in [(0, ""), (2, "  <- 合計(セル 2)より先に!"), (1, ""), (3, "")]:
        show_run(kernel, i, note)
    print("      解説: エラーが出たり、出ていた場所が後から埋まったりします。")
    print("            画面のコードの順序とカーネルの記憶がずれると、")
    print("            文書を上から読む同僚には再現できません。")

    # [4] 教訓の整理
    print("\n[4] 教訓の整理")
    print("    - セル番号 [n] は画面上の位置ではなく「実行された順序」です")
    print("    - 共有の前には必ず Restart & Run All で再現性を確認しましょう")
    print("    - 実験はノートブック、確定したロジックは .py スクリプトへ移すのが正式ルート")


if __name__ == "__main__":
    main()
