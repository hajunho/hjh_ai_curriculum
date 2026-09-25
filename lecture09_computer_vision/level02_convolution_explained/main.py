"""
level02 — 합성곱(Convolution) 이해

합성곱을 numpy 반복문으로 밑바닥부터 구현합니다.
  1) 6x6 미니 예제로 곱셈-덧셈 전 과정 추적
  2) 수직/수평 엣지 커널로 도형의 경계선 검출
  3) 커널 숫자만 바꾸면 흐림/선명화가 되는 것 확인 + PNG 비교판
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

# 검사 도장들: 커널 = "무엇을 찾을지"가 새겨진 숫자판
K_VERTICAL = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="float32")   # 세로 경계
K_HORIZONTAL = K_VERTICAL.T                                                     # 가로 경계
K_BLUR = np.full((3, 3), 1.0 / 9.0, dtype="float32")                            # 평균 = 흐림
K_SHARPEN = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype="float32")    # 선명화


def conv2d(img: np.ndarray, kernel: np.ndarray, padding: int = 0) -> np.ndarray:
    """합성곱의 전부: 커널을 밀며 '곱해서 더하기'. (교육용 순수 구현)"""
    if padding > 0:
        img = np.pad(img, padding)                    # 가장자리에 0 테두리
    h, w = img.shape
    k = kernel.shape[0]
    out = np.zeros((h - k + 1, w - k + 1), dtype="float32")
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            patch = img[y:y + k, x:x + k]             # 도장이 덮은 영역
            out[y, x] = float((patch * kernel).sum()) # 9쌍 곱의 합
    return out


def main() -> None:
    np.random.seed(2)  # seed 고정(재현성)

    # ------------------------------------------------------------------
    print("[1] 미니 예제 — 합성곱 한 번을 숫자 그대로 추적")
    mini = np.zeros((6, 6), dtype="float32")
    mini[:, 3:] = 1.0                                  # 왼쪽 어둡고 오른쪽 밝은 세로 경계
    print("    입력 6x6 (왼쪽=0, 오른쪽=1 인 세로 경계):")
    for row in mini:
        print("      " + " ".join(f"{v:.0f}" for v in row))
    print("    커널(수직 엣지):")
    for row in K_VERTICAL:
        print("      " + " ".join(f"{v:+.0f}" for v in row))
    patch = mini[0:3, 2:5]                             # 경계에 걸친 첫 도장 자리
    print("    위치 (0,2)에 도장을 찍으면 — 9쌍의 곱:")
    terms = []
    for i in range(3):
        for j in range(3):
            terms.append(f"{patch[i, j]:.0f}x{K_VERTICAL[i, j]:+.0f}")
    print("      " + "  ".join(terms))
    print(f"      합계 = {(patch * K_VERTICAL).sum():+.0f}  (경계라서 큰 값이 나옴)")
    flat = mini[0:3, 0:3]
    print(f"    평평한 위치 (0,0)의 합계 = {(flat * K_VERTICAL).sum():+.0f}  (변화 없음 -> 0)")

    # ------------------------------------------------------------------
    print("\n[2] conv2d 구현 확인 — 출력 크기와 패딩")
    out = conv2d(mini, K_VERTICAL)
    out_pad = conv2d(mini, K_VERTICAL, padding=1)
    print(f"    패딩 없음: {mini.shape} -> {out.shape}   (n-k+1 로 줄어듦)")
    print(f"    패딩 1  : {mini.shape} -> {out_pad.shape}   (크기 유지)")
    print("    출력(패딩 없음) — 경계 열에서만 큰 값:")
    for row in out:
        print("      " + " ".join(f"{v:+4.0f}" for v in row))

    # ------------------------------------------------------------------
    print("\n[3] 도형에 엣지 커널 적용 — 커널마다 다른 질문을 던진다")
    X, y = hjh_data.shape_images(n=60, size=16, seed=13)
    names = ["square", "circle", "triangle"]
    samples = [X[np.where(y == c)[0][0]] for c in range(3)]
    sq_v = np.abs(conv2d(samples[0], K_VERTICAL, padding=1))
    sq_h = np.abs(conv2d(samples[0], K_HORIZONTAL, padding=1))
    left_right = sq_v[:, :].max(axis=0)
    print(f"    사각형에 수직 커널: 응답 최대 {sq_v.max():.1f} (좌우 변 위치에서)")
    print(f"    사각형에 수평 커널: 응답 최대 {sq_h.max():.1f} (위아래 변 위치에서)")
    print("    -> 같은 이미지라도 커널이 다르면 '보이는 것'이 다릅니다.")
    _ = left_right  # (참고용 계산)

    # ------------------------------------------------------------------
    print("\n[4] 같은 연산, 다른 커널 — 흐림과 선명화")
    circle = samples[1]
    blurred = conv2d(circle, K_BLUR, padding=1)
    sharpened = np.clip(conv2d(circle, K_SHARPEN, padding=1), 0, 1)
    print(f"    원본 표준편차     = {circle.std():.3f}")
    print(f"    흐림 후 표준편차   = {blurred.std():.3f}  (값들이 평균 쪽으로 뭉개짐)")
    print(f"    선명화 후 표준편차 = {sharpened.std():.3f}  (차이가 더 벌어짐)")

    # ------------------------------------------------------------------
    print("\n[5] 비교판 PNG 저장 — 도형 3종 x (원본/수직/수평/엣지강도)")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(10, 7.5))
    for r, (name, img) in enumerate(zip(names, samples)):
        gv = conv2d(img, K_VERTICAL, padding=1)
        gh = conv2d(img, K_HORIZONTAL, padding=1)
        mag = np.sqrt(gv ** 2 + gh ** 2)               # 방향 무관 엣지 강도
        panels = [(img, f"{name} (input)", "gray"),
                  (gv, "vertical edges", "coolwarm"),
                  (gh, "horizontal edges", "coolwarm"),
                  (mag, "edge magnitude", "magma")]
        for c, (im, title, cmap) in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(im, cmap=cmap)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.suptitle("Hand-made 3x3 kernels: convolution as a pattern detector", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "convolution_edges.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    저장 완료: {path}")

    print("\n[정리] 합성곱 = 숫자 도장을 밀며 '곱해서 더하기'. 커널이 곧 질문이다.")
    print("       다음 레벨: 커널 숫자를 데이터가 정하게 하고 층으로 쌓는다 — CNN 조립.")


if __name__ == "__main__":
    main()
