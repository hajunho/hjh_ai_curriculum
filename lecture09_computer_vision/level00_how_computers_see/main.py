"""
level00 — 컴퓨터가 이미지를 보는 방식

이미지가 "숫자 격자"일 뿐임을 세 가지 표현으로 확인합니다.
  1) 숫자 표 그대로 출력  2) 텍스트 아트(밝기->문자)  3) PNG 저장
추가로 해상도를 낮춰 가며 "타일 수"가 정보량임을 체험합니다.
"""

import os
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
CLASS_NAMES = ["사각형", "원", "삼각형"]
CLASS_NAMES_EN = ["square", "circle", "triangle"]  # 그림용(폰트 호환)
CHARS = " .:-=+*#@"  # 어두움 -> 밝음 순서의 문자 팔레트


def to_ascii(img: np.ndarray) -> str:
    """밝기(0~1)를 문자로 바꿔 '텍스트 아트'를 만든다.
    렌더링 = 숫자를 보기 좋은 기호로 바꾸는 규칙, 그 이상이 아니다."""
    lines = []
    for row in img:
        idx = (np.clip(row, 0.0, 1.0) * (len(CHARS) - 1)).astype(int)
        # 터미널 글자는 세로로 길어서, 픽셀당 문자 2개를 찍어야 비율이 맞는다
        lines.append("".join(CHARS[i] * 2 for i in idx))
    return "\n".join(lines)


def downscale_mean(img: np.ndarray, factor: int) -> np.ndarray:
    """factor x factor 블록을 평균 한 값으로 뭉쳐 해상도를 낮춘다."""
    h, w = img.shape
    return img.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def main() -> None:
    np.random.seed(0)  # 난수 seed 고정(재현성)

    # ------------------------------------------------------------------
    print("[1] 도형 이미지 데이터 만들기 — 다운로드 없이 numpy 로 즉석 생성")
    X, y = hjh_data.shape_images(n=300, size=16, seed=13)
    print(f"    X.shape = {X.shape}  (이미지 300장, 각각 16행 x 16열의 숫자 표)")
    print(f"    값 범위 = {X.min():.2f} ~ {X.max():.2f}  (0=검정, 1=흰색)")
    counts = {CLASS_NAMES[c]: int((y == c).sum()) for c in range(3)}
    print(f"    클래스 구성 = {counts}")

    # 원(label=1) 이미지 하나를 대표로 고른다
    sample = X[np.where(y == 1)[0][0]]

    # ------------------------------------------------------------------
    print("\n[2] 이미지 한 장을 '숫자 표' 그대로 보기 — 컴퓨터가 보는 원본")
    print("    (소수 첫째 자리만 표시. 1.0 이 몰린 곳이 도형입니다)")
    for row in sample:
        print("    " + " ".join(f"{v:.1f}"[1:] for v in row))  # '0.7'->'.7' 로 축약
    print("    -> 사람 눈에는 그냥 숫자 더미지만, 이것이 이미지의 전부입니다.")

    # ------------------------------------------------------------------
    print("\n[3] 같은 배열을 '텍스트 아트'로 보기 — 숫자->문자 규칙 하나만 추가")
    print(to_ascii(sample))
    print("    -> 배열은 그대로인데 도형(원)이 보입니다. 그림이란 숫자의 배열입니다.")

    # ------------------------------------------------------------------
    print("\n[4] 해상도 실험 — 모자이크 타일 수를 줄이면 어떻게 되나")
    for factor, name in [(1, "16x16 (원본)"), (2, "8x8"), (4, "4x4")]:
        small = sample if factor == 1 else downscale_mean(sample, factor)
        print(f"\n    --- {name}: 숫자 {small.size}개 ---")
        for line in to_ascii(small).split("\n"):
            print("    " + line)
    print("\n    -> 4x4 쯤 되면 원인지 사각형인지 구분이 어렵습니다.")
    print("       해상도는 '정보량'이자 '계산 비용'입니다. 과제에 필요한 만큼만 씁니다.")

    # ------------------------------------------------------------------
    print("\n[5] PNG 로 저장 — 지금까지 본 것과 같은 배열을 그림 파일로")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for c in range(3):  # 클래스별로 3장씩
        idx = np.where(y == c)[0][:3]
        for j, i in enumerate(idx):
            ax = axes[c][j]
            ax.imshow(X[i], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{CLASS_NAMES_EN[c]} (y={c})", fontsize=9)
            ax.axis("off")
    fig.suptitle("shape_images: 16x16 grayscale, 3 classes", fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "shapes_grid.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    저장 완료: {path}")
    print("    파일을 열어 보세요. [2]의 숫자 표와 완전히 같은 데이터입니다.")

    print("\n[정리] 이미지 = 밝기 숫자의 격자(스프레드시트).")
    print("       다음 레벨: 이미지가 숫자라면, 편집은 산수다 — 픽셀·채널·이미지 연산.")


if __name__ == "__main__":
    main()
