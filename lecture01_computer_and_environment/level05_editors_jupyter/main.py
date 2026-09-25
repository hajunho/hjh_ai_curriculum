"""
Lecture 01 / Level 05 — 코드 에디터와 Jupyter

Jupyter 를 설치하지 않고 노트북의 '셀 실행' 원리를 체험하는 시뮬레이터입니다.
커널(kernel)의 기억을 딕셔너리 하나로 흉내 내어,
셀들이 기억을 공유하는 모습과, 실행 순서가 어긋날 때 생기는
NameError·어긋난 결과(out-of-order 사고)를 재현합니다.
"""

import io
from contextlib import redirect_stdout


class NotebookKernel:
    """Jupyter 커널의 축소 모형: 기억(변수들)을 들고 셀 코드를 실행합니다."""

    def __init__(self):
        self.memory = {}        # 커널의 기억 — 모든 셀이 이 하나를 공유!
        self.run_counter = 0    # 셀 왼쪽의 [n] 실행 순번

    def restart(self):
        """커널 재시작: 기억이 전부 사라집니다."""
        self.memory = {}
        self.run_counter = 0

    def run_cell(self, code: str):
        """셀 하나를 실행하고 (순번, 출력, 에러) 를 돌려줍니다."""
        self.run_counter += 1
        buffer = io.StringIO()
        error = None
        try:
            with redirect_stdout(buffer):        # print 출력을 가로채기
                exec(code, self.memory)          # 공유 기억 위에서 실행
        except Exception as exc:                 # 에러도 노트북처럼 표시만
            error = f"{type(exc).__name__}: {exc}"
        return self.run_counter, buffer.getvalue().rstrip(), error

    def visible_vars(self):
        """커널 기억 속 사용자 변수 목록 (내부 항목 제외)."""
        return sorted(k for k in self.memory if not k.startswith("__"))


# 노트북 문서: (셀 제목, 셀 코드) — 화면에 위에서 아래로 놓인 순서
CELLS = [
    ("셀 1 — 데이터 준비", "revenues = [1512, 1098, 1745, 702]\nprint('지점 매출(천원):', revenues)"),
    ("셀 2 — 합계", "total = sum(revenues)\nprint('총 매출:', total, '천원')"),
    ("셀 3 — 평균", "average = total / len(revenues)\nprint('평균 매출:', round(average, 1), '천원')"),
    ("셀 4 — 보고", "print(f'보고: 총 {total}천원, 지점 평균 {average:.1f}천원')"),
]


def show_run(kernel, index, note=""):
    """셀 하나를 실행하고 노트북 화면처럼 출력합니다."""
    title, code = CELLS[index]
    n, out, err = kernel.run_cell(code)
    print(f"\n  [{n}] {title}{note}")
    for line in code.splitlines():
        print(f"      | {line}")
    if err:
        print(f"      !! 에러: {err}")
    elif out:
        print(f"      => {out}")
    print(f"      (커널 기억: {kernel.visible_vars() or '비어 있음'})")


def main():
    print("=" * 60)
    print("미니 노트북 시뮬레이터 — 셀과 커널의 원리")
    print("=" * 60)

    kernel = NotebookKernel()

    # [1] 노트북 구성 소개
    print(f"\n[1] 노트북 구성: 셀 {len(CELLS)}개 (매출 분석 시나리오)")
    print("    모든 셀은 '커널'이라는 하나의 파이썬 기억을 공유합니다")

    # [2] Run All: 위에서부터 순서대로 — 기억이 셀을 거치며 쌓입니다
    print("\n[2] Run All — 위에서부터 순서대로 실행")
    for i in range(len(CELLS)):
        show_run(kernel, i)

    # [3] 순서 사고 재현: 재시작 후 중간 셀부터 실행하면?
    print("\n[3] 순서 사고 재현 — 커널 재시작 후 셀 3부터 실행하면?")
    kernel.restart()
    print("    (커널 재시작: 기억이 전부 초기화되었습니다)")
    show_run(kernel, 2, "  <- 위 셀들을 건너뜀!")
    print("      해설: total 이 기억에 없어서 NameError — 노트북 사고 1위입니다")

    print("\n    이번엔 화면 순서와 다르게 1 -> 3 -> 2 -> 4 로 실행해 보면:")
    kernel.restart()
    for i, note in [(0, ""), (2, "  <- 합계(셀 2)보다 먼저!"), (1, ""), (3, "")]:
        show_run(kernel, i, note)
    print("      해설: 에러가 나거나, 났던 자리가 뒤늦게 메워집니다.")
    print("            화면의 코드 순서와 커널의 기억이 어긋나면")
    print("            문서를 위에서부터 읽는 동료는 재현할 수 없습니다.")

    # [4] 교훈 정리
    print("\n[4] 교훈 정리")
    print("    - 셀 번호 [n] 은 화면 위치가 아니라 '실행된 순서'입니다")
    print("    - 공유 전에는 반드시 Restart & Run All 로 재현성을 확인하세요")
    print("    - 실험은 노트북, 확정된 로직은 .py 스크립트로 옮기는 것이 정석")


if __name__ == "__main__":
    main()
