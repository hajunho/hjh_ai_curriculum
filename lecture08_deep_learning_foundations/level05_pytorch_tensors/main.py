"""
PyTorch 의 기본 재료인 텐서(tensor)를 투어합니다.
생성/모양 바꾸기/브로드캐스팅/집계/행렬곱을 단계별로 실행하고,
numpy 배열과의 상호 변환(메모리 공유 포함)과 device 개념을 확인한 뒤
numpy <-> torch API 매핑표를 출력합니다.
"""

import numpy as np
import torch


def main():
    torch.manual_seed(42)                 # 재현성: 난수 시드 고정
    np.random.seed(42)

    print("[1] 텐서 = numpy 배열 + 자동미분 능력 + GPU 이사 능력")
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    z = torch.zeros(2, 3)
    r = torch.arange(6)
    g = torch.randn(2, 3)                 # 표준정규 난수 (시드 고정으로 항상 같음)
    print(f"    tensor  : {a.tolist()}  shape={tuple(a.shape)}, dtype={a.dtype}")
    print(f"    zeros   : shape={tuple(z.shape)}")
    print(f"    arange  : {r.tolist()}")
    print(f"    randn   : 첫 원소 {g[0, 0].item():+.4f} (시드 고정 -> 매번 동일)\n")

    print("[2] 모양 바꾸기 — 데이터는 그대로, 보는 창만 바꿉니다")
    x = torch.arange(12, dtype=torch.float32)
    m = x.reshape(3, 4)
    print(f"    arange(12).reshape(3,4) ->\n{m}")
    print(f"    m.T shape = {tuple(m.T.shape)}, m.flatten() 길이 = {m.flatten().numel()}\n")

    print("[3] 브로드캐스팅 — 모양이 달라도 규칙에 맞으면 자동 확장")
    col = torch.tensor([[10.0], [20.0], [30.0]])      # (3,1)
    row = torch.tensor([1.0, 2.0, 3.0, 4.0])          # (4,)
    print(f"    (3,1) + (4,) -> {tuple((col + row).shape)} 행렬:\n{col + row}\n")

    print("[4] 집계와 행렬곱")
    print(f"    m.sum() = {m.sum().item():.0f}, m.mean(dim=0) = {m.mean(dim=0).tolist()}")
    W = torch.randn(4, 2)
    out = m @ W                            # 신경망 한 층의 본질
    print(f"    (3,4) @ (4,2) = {tuple(out.shape)}  <- 신경망 한 층이 바로 이 연산입니다\n")

    print("[5] numpy <-> torch 변환 — 같은 메모리를 공유합니다(CPU 텐서)")
    arr = np.ones(3, dtype=np.float32)
    t = torch.from_numpy(arr)              # 복사가 아니라 '같은 종이를 같이 보는' 것
    arr[0] = 99.0                          # numpy 쪽을 고치면
    print(f"    numpy 를 99 로 수정 -> torch 텐서도 {t.tolist()} (공유 확인)")
    back = t.numpy()
    t[1] = -7.0
    print(f"    torch 를 -7 로 수정 -> numpy 배열도 {back.tolist()}")
    safe = torch.tensor(arr)               # 복사가 필요하면 torch.tensor() / clone()
    arr[2] = 0.0
    print(f"    torch.tensor(arr) 는 복사본 -> 원본 수정 무관: {safe.tolist()}\n")

    print("[6] device — 텐서가 '어느 계산기 위에' 있는가")
    print(f"    기본 device: {a.device} (CPU)")
    print(f"    cuda 사용 가능? {torch.cuda.is_available()}")
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    print(f"    mps(애플 GPU) 사용 가능? {mps_ok}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    moved = a.to(device)                   # GPU 가 있으면 이사, 없으면 제자리
    print(f"    a.to('{device}') -> device = {moved.device}")
    print("    # 실제 GPU 코드도 이 한 줄입니다: model.to('cuda'), batch.to('cuda')\n")

    print("[7] dtype — 정밀도(자릿수)와 메모리의 트레이드오프")
    f32 = torch.randn(4)
    f16 = f32.to(torch.float16)
    print(f"    float32: {f32[0].item():+.8f} ({f32.element_size()}바이트/원소)")
    print(f"    float16: {f16[0].item():+.8f} ({f16.element_size()}바이트/원소, 뒷자리 손실)\n")

    print("[8] numpy 대비 API 매핑표 — 아는 만큼 그대로 씁니다")
    rows = [
        ("np.array([1,2])",        "torch.tensor([1,2])"),
        ("np.zeros((2,3))",        "torch.zeros(2,3)"),
        ("np.arange(6)",           "torch.arange(6)"),
        ("np.random.randn(2,3)",   "torch.randn(2,3)"),
        ("a.reshape(3,4)",         "a.reshape(3,4) / a.view(3,4)"),
        ("a.T",                    "a.T / a.transpose(0,1)"),
        ("np.dot(a,b) / a @ b",    "torch.matmul(a,b) / a @ b"),
        ("a.sum(axis=0)",          "a.sum(dim=0)"),
        ("np.concatenate([a,b])",  "torch.cat([a,b])"),
        ("a.astype(np.float32)",   "a.to(torch.float32)"),
        ("np.maximum(a,0)",        "torch.relu(a) / a.clamp(min=0)"),
        ("(없음)",                 "a.to('cuda') — GPU 이사"),
        ("(없음)",                 "a.requires_grad_() — 자동미분"),
    ]
    print(f"    {'numpy':34s}| torch")
    print(f"    {'-'*34}|{'-'*34}")
    for np_api, th_api in rows:
        print(f"    {np_api:34s}| {th_api}")
    print("\n[9] 정리: numpy 를 알면 torch 의 90% 는 이미 아는 것입니다.")
    print("    나머지 10%(autograd, device)가 딥러닝을 가능하게 하는 부분이고, 다음 레벨 주제입니다.")


if __name__ == "__main__":
    main()
