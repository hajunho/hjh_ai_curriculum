"""
level09 — 객체 탐지의 원리

'무엇이 어디에'를 답하는 미니 탐지기를 밑바닥부터 만듭니다.
  1) IoU(교집합/합집합) 직접 계산
  2) 학습 장면에서 IoU 규칙으로 크롭 라벨링(도형/배경) -> 분류기 학습
  3) 새 장면을 슬라이딩 윈도우로 훑고  4) NMS 로 중복 상자 정리
  5) 정답 상자와 IoU 매칭으로 채점 + 결과 PNG
"""

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# 이 레벨의 장면은 hjh_data.shape_images 와 같은 화풍으로 numpy 로 직접 그립니다.

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
CLASS_EN = ["square", "circle", "triangle", "background"]
SIZE = 48             # 장면 크기
WIN = 16              # 슬라이딩 윈도우 크기 (분류기 입력)
STRIDE = 2            # 윈도우 이동 보폭
SCORE_TH = 0.90       # 이 확신 미만의 상자는 버림
NMS_IOU = 0.30        # 이보다 많이 겹치는 상자는 하나만 남김


def iou(a: tuple, b: tuple) -> float:
    """IoU = 겹친 넓이 / 합친 넓이. 상자는 (y0, x0, y1, x1)."""
    iy0, ix0 = max(a[0], b[0]), max(a[1], b[1])
    iy1, ix1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, iy1 - iy0) * max(0, ix1 - ix0)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def window_box(cy: int, cx: int) -> tuple:
    """도형 중심을 최대한 가운데 둔 16x16 정답 상자 (장면 밖으로 안 나가게 보정)."""
    y0 = min(max(cy - WIN // 2, 0), SIZE - WIN)
    x0 = min(max(cx - WIN // 2, 0), SIZE - WIN)
    return (y0, x0, y0 + WIN, x0 + WIN)


def make_scene(rng: np.random.Generator):
    """여러 도형이 흩어진 합성 장면과 정답(클래스, 상자) 목록을 만든다."""
    img = np.zeros((SIZE, SIZE), dtype="float32")
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    gt, centers = [], []
    for label in rng.permutation(3):                   # 도형 3개, 클래스 서로 다름
        for _ in range(200):                           # 겹치지 않는 자리 찾기
            cy, cx = (int(v) for v in rng.integers(8, SIZE - 8, size=2))
            if all(abs(cy - y) + abs(cx - x) >= 20 for y, x in centers):
                break
        centers.append((cy, cx))
        r = int(rng.integers(3, 5))
        if label == 0:                                 # 사각형 (hjh_data 와 같은 화풍)
            img[cy - r:cy + r, cx - r:cx + r] = 1.0
        elif label == 1:                               # 원
            img[((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r] = 1.0
        else:                                          # 삼각형
            mask = (yy >= cy - r) & (yy <= cy + r) & (np.abs(xx - cx) <= (yy - cy + r) / 2)
            img[mask] = 1.0
        gt.append((int(label), (cy - r, cx - r, cy + r, cx + r), window_box(cy, cx)))
    img += rng.normal(0, 0.08, img.shape).astype("float32")
    return np.clip(img, 0.0, 1.0), gt


def build_training_crops(rng: np.random.Generator, n_scenes: int = 40):
    """탐지 학습 데이터 만들기 — 크롭의 라벨을 IoU 규칙으로 정한다.
    정답 상자와 IoU>=0.55 면 그 도형, 아니면 전부 배경.
    '반쯤 걸친 창'도 배경으로 가르쳐야, 탐지가 도형 중심에만 반응한다."""
    pos, pos_y, bg = [], [], []
    for _ in range(n_scenes):
        scene, gt = make_scene(rng)
        for y0 in range(0, SIZE - WIN + 1, STRIDE):
            for x0 in range(0, SIZE - WIN + 1, STRIDE):
                box = (y0, x0, y0 + WIN, x0 + WIN)
                ious = [iou(box, wb) for _, _, wb in gt]
                best = int(np.argmax(ious))
                if ious[best] >= 0.55:
                    pos.append(scene[y0:y0 + WIN, x0:x0 + WIN])
                    pos_y.append(gt[best][0])
                else:                                   # 빗나간 창도, 반쯤 걸친 창도 배경
                    bg.append(scene[y0:y0 + WIN, x0:x0 + WIN])
    n_bg = int(len(pos) * 2.0)                          # 배경은 넘치므로 일부만
    bg_idx = rng.choice(len(bg), size=n_bg, replace=False)
    X = np.stack(pos + [bg[i] for i in bg_idx]).astype("float32")
    y = np.array(pos_y + [3] * n_bg, dtype="int64")
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def train_classifier(X: np.ndarray, y: np.ndarray) -> nn.Module:
    torch.manual_seed(9)
    xt, yt = torch.from_numpy(X).unsqueeze(1), torch.from_numpy(y)
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(256, 4))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(25):
        p = torch.randperm(len(xt))
        for i in range(0, len(xt), 64):
            idx = p[i:i + 64]
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def sliding_window_detect(model: nn.Module, scene: np.ndarray):
    """장면 위를 창문으로 훑으며 각 위치를 분류 -> 후보 상자 목록."""
    cands, n_windows = [], 0
    for y0 in range(0, SIZE - WIN + 1, STRIDE):
        for x0 in range(0, SIZE - WIN + 1, STRIDE):
            patch = torch.from_numpy(scene[y0:y0 + WIN, x0:x0 + WIN])[None, None]
            prob = torch.softmax(model(patch)[0], dim=0)
            cls = int(prob.argmax())
            n_windows += 1
            if cls != 3 and float(prob[cls]) >= SCORE_TH:   # 배경 아님 + 확신 높음
                cands.append((float(prob[cls]), cls, (y0, x0, y0 + WIN, x0 + WIN)))
    return cands, n_windows


def nms(cands: list) -> list:
    """비최대 억제: 확신 높은 상자부터 채택, 많이 겹치는 나머지는 탈락."""
    keep = []
    for score, cls, box in sorted(cands, reverse=True):
        if all(iou(box, kb) < NMS_IOU for _, _, kb in keep):
            keep.append((score, cls, box))
    return keep


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(9)
    np.random.seed(9)  # seed 고정(재현성)

    # ------------------------------------------------------------------
    print("[1] IoU — 상자 두 개가 '같은 물체'인지 재는 자")
    a, b = (10, 10, 20, 20), (14, 14, 24, 24)
    print(f"    상자 A={a}, 상자 B={b}")
    print(f"    겹침 6x6=36, 합집합 100+100-36=164 -> IoU = {iou(a, b):.3f}")
    print("    완전 일치면 1.0, 안 겹치면 0.0. 관례상 IoU >= 0.5 면 '맞힌 것'으로 봅니다.")

    # ------------------------------------------------------------------
    print("\n[2] 탐지용 학습 데이터 — 크롭의 라벨을 IoU 규칙으로 자동 결정")
    Xc, yc = build_training_crops(rng)
    counts = {CLASS_EN[c]: int((yc == c).sum()) for c in range(4)}
    print(f"    학습 장면 40개를 창문으로 훑어 크롭 {len(Xc)}개: {counts}")
    print("    도형 위에 잘 얹힌 창(IoU>=0.55)만 도형, 나머지는 전부 배경.")
    print("    '반쯤 걸친 창'도 배경으로 가르쳐야 탐지가 중심에서만 울립니다.")
    model = train_classifier(Xc, yc)

    # ------------------------------------------------------------------
    print("\n[3] 새 장면을 슬라이딩 윈도우로 훑기")
    scene, gt = make_scene(rng)
    print(f"    {SIZE}x{SIZE} 장면의 정답: "
          + ", ".join(f"{CLASS_EN[c]}@{wb}" for c, _, wb in gt))
    cands, n_win = sliding_window_detect(model, scene)
    print(f"    {WIN}x{WIN} 창을 보폭 {STRIDE}로 이동: 총 {n_win}개 위치를 각각 분류")
    print(f"    확신 {SCORE_TH} 이상의 후보 상자: {len(cands)}개 (같은 도형에 중복 다수)")

    # ------------------------------------------------------------------
    print("\n[4] NMS(비최대 억제) — 중복 상자 정리")
    dets = nms(cands)
    print(f"    후보 {len(cands)}개 -> 최종 {len(dets)}개")
    for score, cls, box in dets:
        print(f"      {CLASS_EN[cls]:>9} score={score:.2f} box={box}")

    # ------------------------------------------------------------------
    print("\n[5] 채점 — 정답 상자와 IoU >= 0.5 & 클래스 일치면 정탐(TP)")
    matched, tp = set(), 0
    for score, cls, box in dets:
        for j, (gc, _, gwb) in enumerate(gt):
            if j not in matched and gc == cls and iou(box, gwb) >= 0.5:
                matched.add(j)
                tp += 1
                break
    print(f"    정탐(TP) {tp} / 오탐(FP) {len(dets) - tp} / 미탐(FN) {len(gt) - tp}")
    print("    (실전 지표 mAP 는 이 정탐/오탐 집계를 확신 임계값 전체에 걸쳐 평균한 것입니다)")

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    gt_draw = [(1.0, c, tb) for c, tb, _ in gt]
    for ax, title, boxes, color in [
            (axes[0], "ground truth (green = tight boxes)", gt_draw, "lime"),
            (axes[1], f"detections after NMS ({len(dets)} boxes)", dets, "red")]:
        ax.imshow(scene, cmap="gray", vmin=0, vmax=1)
        for score, cls, (y0, x0, y1, x1) in boxes:
            ax.add_patch(patches.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
                                           fill=False, edgecolor=color, linewidth=1.5))
            ax.text(x0, y0 - 1.2, f"{CLASS_EN[cls]}" + ("" if score == 1.0 else f" {score:.2f}"),
                    color=color, fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("Mini detector: sliding window + classifier + NMS", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "object_detection.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    저장 완료: {path}")

    print(f"\n[정리] 탐지 = 분류기 + 위치 훑기 + NMS + IoU 채점 (소요 {time.time() - t0:.1f}초).")
    print("       현대 탐지기(YOLO 계열 등)는 이 '훑기'를 CNN 한 번의 계산으로 병렬화한 것 —")
    print("       원리는 오늘 만든 것과 같습니다. 다음 레벨: 픽셀 단위로 칠하는 세그멘테이션.")


if __name__ == "__main__":
    main()
