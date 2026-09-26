"""
level09 — How Object Detection Works

We build, from the ground up, a mini detector that answers 'what is where'.
  1) compute IoU (intersection/union) by hand
  2) label crops from training scenes via the IoU rule (shape/background) -> train a classifier
  3) sweep a new scene with a sliding window   4) clean up duplicate boxes with NMS
  5) grade by IoU matching against ground truth + a result PNG
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

# This level's scenes are drawn directly in numpy, in the same style as hjh_data.shape_images.

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
CLASS_EN = ["square", "circle", "triangle", "background"]
SIZE = 48             # scene size
WIN = 16              # sliding-window size (classifier input)
STRIDE = 2            # window step
SCORE_TH = 0.90       # boxes below this confidence are dropped
NMS_IOU = 0.30        # boxes overlapping more than this keep only one


def iou(a: tuple, b: tuple) -> float:
    """IoU = overlap area / union area. Boxes are (y0, x0, y1, x1)."""
    iy0, ix0 = max(a[0], b[0]), max(a[1], b[1])
    iy1, ix1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, iy1 - iy0) * max(0, ix1 - ix0)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def window_box(cy: int, cx: int) -> tuple:
    """A 16x16 ground-truth box centering the shape as well as possible (clamped inside the scene)."""
    y0 = min(max(cy - WIN // 2, 0), SIZE - WIN)
    x0 = min(max(cx - WIN // 2, 0), SIZE - WIN)
    return (y0, x0, y0 + WIN, x0 + WIN)


def make_scene(rng: np.random.Generator):
    """Create a synthetic scene with several scattered shapes plus the ground-truth (class, box) list."""
    img = np.zeros((SIZE, SIZE), dtype="float32")
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    gt, centers = [], []
    for label in rng.permutation(3):                   # 3 shapes, all different classes
        for _ in range(200):                           # find a non-overlapping spot
            cy, cx = (int(v) for v in rng.integers(8, SIZE - 8, size=2))
            if all(abs(cy - y) + abs(cx - x) >= 20 for y, x in centers):
                break
        centers.append((cy, cx))
        r = int(rng.integers(3, 5))
        if label == 0:                                 # square (same style as hjh_data)
            img[cy - r:cy + r, cx - r:cx + r] = 1.0
        elif label == 1:                               # circle
            img[((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r] = 1.0
        else:                                          # triangle
            mask = (yy >= cy - r) & (yy <= cy + r) & (np.abs(xx - cx) <= (yy - cy + r) / 2)
            img[mask] = 1.0
        gt.append((int(label), (cy - r, cx - r, cy + r, cx + r), window_box(cy, cx)))
    img += rng.normal(0, 0.08, img.shape).astype("float32")
    return np.clip(img, 0.0, 1.0), gt


def build_training_crops(rng: np.random.Generator, n_scenes: int = 40):
    """Build detection training data — crop labels decided by the IoU rule.
    IoU>=0.55 with a ground-truth box means that shape; everything else is background.
    'Half-covering windows' must be taught as background too, so detection fires only on centers."""
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
                else:                                   # missed windows AND half-covering ones: background
                    bg.append(scene[y0:y0 + WIN, x0:x0 + WIN])
    n_bg = int(len(pos) * 2.0)                          # background is overabundant, keep a subset
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
    """Sweep the scene with the window, classifying every position -> candidate box list."""
    cands, n_windows = [], 0
    for y0 in range(0, SIZE - WIN + 1, STRIDE):
        for x0 in range(0, SIZE - WIN + 1, STRIDE):
            patch = torch.from_numpy(scene[y0:y0 + WIN, x0:x0 + WIN])[None, None]
            prob = torch.softmax(model(patch)[0], dim=0)
            cls = int(prob.argmax())
            n_windows += 1
            if cls != 3 and float(prob[cls]) >= SCORE_TH:   # not background + high confidence
                cands.append((float(prob[cls]), cls, (y0, x0, y0 + WIN, x0 + WIN)))
    return cands, n_windows


def nms(cands: list) -> list:
    """Non-maximum suppression: accept boxes from highest confidence down; heavily overlapping ones lose."""
    keep = []
    for score, cls, box in sorted(cands, reverse=True):
        if all(iou(box, kb) < NMS_IOU for _, _, kb in keep):
            keep.append((score, cls, box))
    return keep


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(9)
    np.random.seed(9)  # fix the seed (reproducibility)

    # ------------------------------------------------------------------
    print("[1] IoU — the ruler measuring whether two boxes are 'the same object'")
    a, b = (10, 10, 20, 20), (14, 14, 24, 24)
    print(f"    box A={a}, box B={b}")
    print(f"    overlap 6x6=36, union 100+100-36=164 -> IoU = {iou(a, b):.3f}")
    print("    Perfect match = 1.0, no overlap = 0.0. Convention: IoU >= 0.5 counts as 'correct'.")

    # ------------------------------------------------------------------
    print("\n[2] Training data for detection — crop labels decided automatically by the IoU rule")
    Xc, yc = build_training_crops(rng)
    counts = {CLASS_EN[c]: int((yc == c).sum()) for c in range(4)}
    print(f"    Swept 40 training scenes with the window: {len(Xc)} crops: {counts}")
    print("    Only windows sitting properly on a shape (IoU>=0.55) are that shape; the rest are background.")
    print("    'Half-covering windows' taught as background too — so detection fires only at centers.")
    model = train_classifier(Xc, yc)

    # ------------------------------------------------------------------
    print("\n[3] Sweeping a new scene with the sliding window")
    scene, gt = make_scene(rng)
    print(f"    Ground truth of the {SIZE}x{SIZE} scene: "
          + ", ".join(f"{CLASS_EN[c]}@{wb}" for c, _, wb in gt))
    cands, n_win = sliding_window_detect(model, scene)
    print(f"    Moving a {WIN}x{WIN} window at stride {STRIDE}: {n_win} positions classified individually")
    print(f"    Candidate boxes above confidence {SCORE_TH}: {len(cands)} (many duplicates per shape)")

    # ------------------------------------------------------------------
    print("\n[4] NMS (non-maximum suppression) — cleaning up duplicate boxes")
    dets = nms(cands)
    print(f"    {len(cands)} candidates -> {len(dets)} final")
    for score, cls, box in dets:
        print(f"      {CLASS_EN[cls]:>9} score={score:.2f} box={box}")

    # ------------------------------------------------------------------
    print("\n[5] Grading — IoU >= 0.5 with a ground-truth box & matching class = true positive (TP)")
    matched, tp = set(), 0
    for score, cls, box in dets:
        for j, (gc, _, gwb) in enumerate(gt):
            if j not in matched and gc == cls and iou(box, gwb) >= 0.5:
                matched.add(j)
                tp += 1
                break
    print(f"    TP {tp} / FP {len(dets) - tp} / FN {len(gt) - tp}")
    print("    (the production metric mAP averages this TP/FP tally across all confidence thresholds)")

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
    print(f"    Saved: {path}")

    print(f"\n[Recap] Detection = classifier + position sweep + NMS + IoU grading (took {time.time() - t0:.1f}s).")
    print("        Modern detectors (YOLO lineage etc.) parallelize this 'sweep' into one CNN pass —")
    print("        the principle is the same as what we built today. Next level: painting pixel by pixel — segmentation.")


if __name__ == "__main__":
    main()
