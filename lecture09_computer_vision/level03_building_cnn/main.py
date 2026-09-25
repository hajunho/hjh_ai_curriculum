"""
level03 — CNN 구조 만들기

torch 로 미니 CNN 을 정의하고 (학습은 아직 안 합니다),
  1) Conv2d 의 무게 텐서 해부  2) 출력 크기 공식 검산
  3) 층별 shape 변화 추적      4) 파라미터 수 손 계산 vs torch 집계
  5) MLP 와의 파라미터 수 비교
"""

import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data


def conv_out(size: int, kernel: int, stride: int = 1, padding: int = 0) -> int:
    """출력 크기 공식: (in + 2p - k) / s + 1  — 이 강의에서 가장 자주 쓸 식."""
    return (size + 2 * padding - kernel) // stride + 1


class TinyCNN(nn.Module):
    """16x16 흑백 도형 3클래스용 미니 CNN.
    설계 리듬: 채널은 늘리고(1->8->16->32), 공간은 줄인다(16->16->8->4)."""

    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.head = nn.Linear(32 * 4 * 4, n_classes)

    def forward_traced(self, x: torch.Tensor) -> torch.Tensor:
        """층 하나 지날 때마다 shape 를 출력하며 통과시킨다."""
        steps = [
            ("입력 (B,C,H,W)", lambda t: t),
            ("conv1 1->8ch, s1, p1", lambda t: self.relu(self.conv1(t))),
            ("conv2 8->16ch, s2, p1", lambda t: self.relu(self.conv2(t))),
            ("conv3 16->32ch, s2, p1", lambda t: self.relu(self.conv3(t))),
            ("flatten", self.flatten),
            ("linear 512->3", self.head),
        ]
        for name, fn in steps:
            x = fn(x)
            print(f"      {name:<24} -> {tuple(x.shape)}")
        return x


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def main() -> None:
    torch.manual_seed(3)
    np.random.seed(3)  # seed 고정(재현성)

    # ------------------------------------------------------------------
    print("[1] Conv2d 층 해부 — '도장 여러 개를 든 검사팀'")
    conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
    w = conv.weight
    print(f"    weight.shape = {tuple(w.shape)}  (도장 8개, 각각 1채널 x 3x3)")
    print(f"    bias.shape   = {tuple(conv.bias.shape)}  (도장마다 편향 1개)")
    print(f"    이 층의 파라미터 = 8 x (1x3x3) + 8 = {8 * 9 + 8}개  "
          f"(torch 집계: {count_params(conv)}개)")

    # ------------------------------------------------------------------
    print("\n[2] 출력 크기 공식 검산 — out = (in + 2p - k)/s + 1")
    print("    입력 16, 커널 3 일 때:")
    print(f"      {'stride':>7} {'padding':>8} {'공식':>6} {'실측':>6}")
    x16 = torch.zeros(1, 1, 16, 16)
    for s, p in [(1, 0), (1, 1), (2, 1), (2, 0)]:
        c = nn.Conv2d(1, 1, 3, stride=s, padding=p)
        real = c(x16).shape[-1]
        pred = conv_out(16, 3, s, p)
        print(f"      {s:>7} {p:>8} {pred:>6} {real:>6}")
    print("    -> 공식과 실측이 항상 일치합니다. 층 설계는 이 식으로 끝까지 추적하세요.")

    # ------------------------------------------------------------------
    print("\n[3] 미니 CNN 에 도형 배치를 통과시키며 shape 추적")
    X, y = hjh_data.shape_images(n=32, size=16, seed=13)
    batch = torch.from_numpy(X).unsqueeze(1)           # (32,16,16) -> (32,1,16,16) B-C-H-W
    print(f"    numpy {X.shape} -> torch {tuple(batch.shape)}  (배치, 채널, 높이, 너비)")
    model = TinyCNN()
    with torch.no_grad():
        scores = model.forward_traced(batch)
    print(f"    출력 = 이미지 32장 각각의 3클래스 점수. 예: 첫 장 {scores[0].numpy().round(2)}")
    print("    (아직 학습 전 무작위 무게라 점수는 의미가 없습니다 — 구조만 확인)")

    # ------------------------------------------------------------------
    print("\n[4] 층별 파라미터 수 — 손 계산 vs torch 집계")
    rows = [
        ("conv1", "8 x (1x3x3) + 8", 8 * 1 * 9 + 8, model.conv1),
        ("conv2", "16 x (8x3x3) + 16", 16 * 8 * 9 + 16, model.conv2),
        ("conv3", "32 x (16x3x3) + 32", 32 * 16 * 9 + 32, model.conv3),
        ("head", "3 x 512 + 3", 3 * 512 + 3, model.head),
    ]
    print(f"      {'층':<7} {'공식':<20} {'손 계산':>8} {'torch':>8}")
    for name, formula, hand, mod in rows:
        print(f"      {name:<7} {formula:<20} {hand:>8,} {count_params(mod):>8,}")
    total = count_params(model)
    print(f"      {'합계':<28} {sum(r[2] for r in rows):>8,} {total:>8,}")
    print("    -> Conv 파라미터는 이미지 크기와 무관, Linear 는 Flatten 크기에 비례합니다.")

    # ------------------------------------------------------------------
    print("\n[5] 같은 입력을 MLP 로 처리한다면? — 가중치 공유의 위력")
    mlp = nn.Sequential(nn.Flatten(), nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))
    print(f"    MLP  (256->128->64->3)  파라미터: {count_params(mlp):>8,}개")
    print(f"    CNN  (TinyCNN)          파라미터: {total:>8,}개")
    print("    -> CNN 은 '같은 도장을 온 이미지에 재사용'하므로 훨씬 가볍습니다.")
    print("       입력이 16x16 이 아니라 160x160 이면 MLP 는 100배로 커지지만,")
    print("       CNN 의 Conv 부분은 파라미터가 그대로입니다 (Linear 만 커짐).")

    print("\n[정리] 구조 읽기 = shape 추적 + 파라미터 계산. 이 둘이면 어떤 CNN 도 견적이 나온다.")
    print("       다음 레벨: 공간을 요약하는 전용 부품, 풀링 — 그리고 특징의 계층.")


if __name__ == "__main__":
    main()
