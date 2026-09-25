"""
level10 — 세그멘테이션·OCR·산업 검사

딥러닝 없이 고전 기법만으로 픽셀 단위 분석을 완성합니다.
  1) 임계값(threshold)으로 전경/배경 이진화
  2) 연결 요소(connected components) 라벨링을 BFS 로 직접 구현
  3) 영역별 속성(면적·상자·중심) 측정 -> 부품과 불량(얼룩) 자동 판정
결과는 outputs/segmentation.png 4단 비교판으로 저장합니다.
"""

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 이 레벨의 검사 장면은 hjh_data 와 같은 화풍으로 numpy 로 직접 그립니다.

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SIZE = 64
THRESH = 0.45          # 이 밝기 이상이면 '물체 픽셀'
DEFECT_AREA = 12       # 이 면적 미만의 연결 요소는 '불량 얼룩'으로 판정


def make_inspection_image(rng: np.random.Generator):
    """검사 장면: 정상 부품(도형 3개) + 불량 얼룩 4개 + 촬영 잡음."""
    img = np.zeros((SIZE, SIZE), dtype="float32")
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    # 정상 부품: 사각형/원/삼각형 (hjh_data 와 같은 화풍, 위치만 고정 배치)
    img[8:22, 8:22] = 0.95                                        # 사각형
    img[((yy - 16) ** 2 + (xx - 44) ** 2) <= 49] = 0.95           # 원 (r=7)
    tri = (yy >= 38) & (yy <= 54) & (np.abs(xx - 20) <= (yy - 38) / 2)
    img[tri] = 0.95                                               # 삼각형
    # 불량: 작은 얼룩(스패터) 4개 — 부품보다 훨씬 작다
    defects = []
    for _ in range(4):
        while True:
            dy, dx = (int(v) for v in rng.integers(4, SIZE - 4, size=2))
            if img[dy - 3:dy + 3, dx - 3:dx + 3].max() < 0.1:     # 부품과 안 겹치게
                break
        blob = ((yy - dy) ** 2 + (xx - dx) ** 2) <= int(rng.integers(2, 5))
        img[blob] = 0.85
        defects.append((dy, dx))
    img += rng.normal(0, 0.06, img.shape).astype("float32")       # 촬영 잡음
    return np.clip(img, 0.0, 1.0), defects


def connected_components(mask: np.ndarray):
    """4방향 연결 요소 라벨링 — BFS(양동이 채우기) 직접 구현.
    같은 덩어리의 픽셀에 같은 번호를 붙인다. 반환: 라벨 맵, 요소 개수."""
    labels = np.zeros(mask.shape, dtype=int)
    current = 0
    for sy in range(mask.shape[0]):
        for sx in range(mask.shape[1]):
            if mask[sy, sx] and labels[sy, sx] == 0:
                current += 1                          # 새 덩어리 발견
                stack = [(sy, sx)]
                labels[sy, sx] = current
                while stack:                          # 이웃으로 번지며 칠하기
                    y, x = stack.pop()
                    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]
                                and mask[ny, nx] and labels[ny, nx] == 0):
                            labels[ny, nx] = current
                            stack.append((ny, nx))
    return labels, current


def region_props(labels: np.ndarray, k: int) -> dict:
    """영역 k 의 속성: 면적, 바운딩 박스, 중심."""
    ys, xs = np.where(labels == k)
    return {"area": len(ys),
            "bbox": (int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1),
            "center": (float(ys.mean()), float(xs.mean()))}


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(10)
    np.random.seed(10)  # seed 고정(재현성)

    # ------------------------------------------------------------------
    print("[1] 검사 장면 준비 — 정상 부품 3개 + 몰래 뿌린 불량 얼룩 4개")
    img, true_defects = make_inspection_image(rng)
    print(f"    {SIZE}x{SIZE} 흑백 이미지. 사람 눈에는 쉬워도, 기계는 픽셀 단위 근거가 필요합니다.")
    print(f"    (정답) 얼룩 위치: {true_defects} — 검출기는 이 정보를 모릅니다.")

    # ------------------------------------------------------------------
    print("\n[2] 임계값 이진화 — '이 밝기보다 밝으면 물체'")
    mask = img >= THRESH
    print(f"    임계값 {THRESH}: 전경 픽셀 {int(mask.sum())}개 / 전체 {mask.size}개 "
          f"({mask.mean():.1%})")
    print("    배경 잡음(평균 0, 표준편차 0.06)은 임계값을 거의 못 넘습니다.")
    print("    -> 밝기 분포가 두 봉우리로 갈릴 때 잘 통하는 가장 값싼 세그멘테이션입니다.")

    # ------------------------------------------------------------------
    print("\n[3] 연결 요소 라벨링 — 붙어 있는 픽셀끼리 같은 번호")
    labels, n = connected_components(mask)
    print(f"    BFS 로 번지며 칠한 결과: 연결 요소 {n}개")

    # ------------------------------------------------------------------
    print("\n[4] 영역 측정과 판정 — 면적이 작으면 얼룩, 크면 부품")
    print(f"      {'id':>3} {'면적':>5} {'중심(y,x)':>14} {'판정':>8}")
    found_defects = []
    for k in range(1, n + 1):
        p = region_props(labels, k)
        verdict = "불량얼룩" if p["area"] < DEFECT_AREA else "부품"
        if verdict == "불량얼룩":
            found_defects.append(p)
        cy, cx = p["center"]
        print(f"      {k:>3} {p['area']:>5} {f'({cy:5.1f},{cx:5.1f})':>14} {verdict:>8}")
    print(f"    -> 면적 {DEFECT_AREA} 미만 규칙으로 얼룩 {len(found_defects)}개 검출 "
          f"(실제 {len(true_defects)}개).")
    matched = 0
    for p in found_defects:
        cy, cx = p["center"]
        if any(abs(cy - dy) < 3 and abs(cx - dx) < 3 for dy, dx in true_defects):
            matched += 1
    print(f"    정답 대조: {matched}/{len(true_defects)} 얼룩의 위치가 일치 — 픽셀 단위 검사 성공.")

    # ------------------------------------------------------------------
    print("\n[5] 비교판 PNG 저장 — 원본/이진화/영역 라벨/판정 오버레이")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("input (parts + defects)", fontsize=10)
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"threshold >= {THRESH}", fontsize=10)
    axes[2].imshow(np.where(labels > 0, labels, np.nan), cmap="tab10")
    axes[2].set_title(f"connected components ({n})", fontsize=10)
    axes[3].imshow(img, cmap="gray", vmin=0, vmax=1)
    overlay = np.zeros((SIZE, SIZE, 4), dtype=float)
    for k in range(1, n + 1):
        p = region_props(labels, k)
        color = (1, 0, 0, 0.55) if p["area"] < DEFECT_AREA else (0, 0.8, 0.2, 0.35)
        overlay[labels == k] = color
    axes[3].imshow(overlay)
    axes[3].set_title("verdict: green=part, red=defect", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Classic segmentation: threshold + connected components + area rule", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "segmentation.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    저장 완료: {path}")

    print(f"\n[정리] 세그멘테이션 = 픽셀마다 소속 정하기. 고전 3종 세트(임계값->연결요소->면적 규칙)")
    print(f"       만으로도 불량 검출이 됩니다 (소요 {time.time() - t0:.1f}초). OCR 도 같은 뼈대:")
    print("       글자 영역 분리(세그멘테이션) 후 각 영역을 분류(인식)합니다.")
    print("       다음 레벨: 이미지를 '패치 단어'로 읽는 비전 트랜스포머와 멀티모달.")


if __name__ == "__main__":
    main()
