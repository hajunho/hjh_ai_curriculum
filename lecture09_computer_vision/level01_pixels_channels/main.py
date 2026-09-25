"""
level01 — 픽셀·채널·이미지 연산

numpy 만으로 RGB 이미지를 직접 그린 뒤,
밝기(덧셈)·대비(곱셈)·반전·흑백 변환·크롭·리사이즈를 전부 구현합니다.
결과는 outputs/pixel_ops.png 비교판 한 장으로 정리합니다.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
BRIGHT_DELTA = 0.25   # 밝기 조절량 (덧셈)
CONTRAST_GAIN = 1.8   # 대비 배율 (곱셈)


def make_scene(size: int = 48) -> np.ndarray:
    """다운로드 없이 numpy 로 '하늘 + 빨간 사각형 + 노란 원' 장면을 그린다."""
    img = np.zeros((size, size, 3), dtype="float32")
    # 하늘: 위(밝은 파랑) -> 아래(어두운 파랑) 세로 그라데이션
    grad = np.linspace(0.9, 0.3, size)[:, None]           # (size,1) 세로 방향
    img[:, :, 2] = grad                                    # 파랑 채널
    img[:, :, 1] = grad * 0.6                              # 초록 살짝
    # 빨간 사각형 (건물): R 채널만 진하게
    img[26:44, 6:22] = [0.85, 0.15, 0.10]
    # 노란 원 (해): R+G 진하게, B 없음
    yy, xx = np.mgrid[0:size, 0:size]
    sun = ((yy - 10) ** 2 + (xx - 36) ** 2) <= 6 ** 2
    img[sun] = [1.0, 0.9, 0.1]
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    """흑백 변환: 사람 눈의 민감도를 반영한 가중 평균(관례 계수)."""
    return img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114


def resize_nearest(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """최근접 이웃 리사이즈: 새 픽셀마다 가장 가까운 원본 픽셀을 참조한다."""
    h, w = img.shape[:2]
    yy = np.clip(np.round(np.linspace(0, h - 1, new_h)), 0, h - 1).astype(int)
    xx = np.clip(np.round(np.linspace(0, w - 1, new_w)), 0, w - 1).astype(int)
    return img[yy][:, xx]      # 팬시 인덱싱 두 번이면 리사이즈 완성


def stats(name: str, a: np.ndarray) -> None:
    print(f"    {name:<18} min={a.min():>6.2f}  mean={a.mean():>5.2f}  max={a.max():>6.2f}")


def main() -> None:
    np.random.seed(1)  # 난수 seed 고정(이 레벨은 난수 미사용이지만 규칙 준수)

    # ------------------------------------------------------------------
    print("[1] numpy 로 RGB 이미지 직접 그리기 — 배열 조작 = 그림 그리기")
    img = make_scene(48)
    print(f"    shape = {img.shape}  (높이 48, 너비 48, 채널 3장: R/G/B)")
    print("    하늘은 그라데이션(linspace), 사각형은 슬라이싱, 원은 거리식 마스크로 그렸습니다.")

    # ------------------------------------------------------------------
    print("\n[2] 채널 분리 — '빨갛다'의 정체는 채널별 숫자 차이")
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    box = (slice(30, 40), slice(10, 20))  # 빨간 사각형 내부 영역
    print(f"    빨간 사각형 영역의 채널 평균: R={r[box].mean():.2f}  G={g[box].mean():.2f}  B={b[box].mean():.2f}")
    print("    -> R 채널에서만 밝다 = 우리 눈에 '빨강'으로 보인다")

    # ------------------------------------------------------------------
    print("\n[3] 픽셀 연산 4형제 — 밝기=덧셈, 대비=곱셈, 반전=빼기")
    brighter_raw = img + BRIGHT_DELTA                      # clip 전 (일부러)
    brighter = np.clip(brighter_raw, 0.0, 1.0)
    contrast = np.clip((img - 0.5) * CONTRAST_GAIN + 0.5, 0.0, 1.0)
    inverted = 1.0 - img
    gray = to_gray(img)
    stats("원본", img)
    stats(f"밝기 +{BRIGHT_DELTA} (clip 전)", brighter_raw)
    stats(f"밝기 +{BRIGHT_DELTA} (clip 후)", brighter)
    stats(f"대비 x{CONTRAST_GAIN}", contrast)
    stats("반전 1-x", inverted)
    stats("흑백 변환", gray)
    print("    -> clip 전 max 가 1.0 을 넘는 것에 주목. 연산 후 clip 은 습관입니다.")

    # ------------------------------------------------------------------
    print("\n[4] 기하 연산 — 크롭은 슬라이싱, 리사이즈는 인덱싱")
    crop = img[24:46, 4:24]                                # 빨간 사각형 주변만
    up = resize_nearest(img, 96, 96)                       # 2배 확대
    down = resize_nearest(img, 16, 16)                     # 1/3 축소
    print(f"    크롭:     {img.shape} -> {crop.shape}   (img[24:46, 4:24])")
    print(f"    확대:     {img.shape} -> {up.shape}  (정보는 늘지 않고 계단만 생김)")
    print(f"    축소:     {img.shape} -> {down.shape}  (숫자 {img[:, :, 0].size}개 -> {down[:, :, 0].size}개)")

    # ------------------------------------------------------------------
    print("\n[5] 비교판 PNG 저장")
    os.makedirs(OUT_DIR, exist_ok=True)
    panels = [
        ("original (RGB)", img, None), ("R channel", r, "gray"),
        ("G channel", g, "gray"), ("B channel", b, "gray"),
        (f"brightness +{BRIGHT_DELTA}", brighter, None), (f"contrast x{CONTRAST_GAIN}", contrast, None),
        ("inverted", inverted, None), ("grayscale", gray, "gray"),
        ("crop", crop, None), ("resize up 96x96", up, None),
        ("resize down 16x16", down, None), ("normalized (z-score)", (gray - gray.mean()) / gray.std(), "gray"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.5))
    for ax, (title, im, cmap) in zip(axes.ravel(), panels):
        if cmap == "gray" and title.startswith("normalized"):
            ax.imshow(im, cmap="gray")                     # z-score 는 범위 자동
        elif cmap == "gray":
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(im)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle("Pixel & channel operations (numpy only)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "pixel_ops.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    저장 완료: {path}")

    print("\n[정리] 이미지 편집 = 배열 산수. 밝기는 덧셈, 대비는 곱셈, 크롭은 슬라이싱.")
    print("       다음 레벨: 픽셀 '혼자'가 아니라 '이웃과 함께' 보는 연산 — 합성곱.")


if __name__ == "__main__":
    main()
