"""
양자화(quantization) 실습.
fp32 가중치를 int8/int4 정수로 압축했다가 복원하는 대칭 양자화를 numpy 로
직접 구현하고, tiny_corpus 로 짧게 학습한 문자 단위 미니 언어모델의 가중치에
적용해 메모리 절감률과 손실(loss) 변화를 측정합니다.
블록 단위 양자화가 왜 GGUF 같은 배포 포맷의 핵심인지도 수치로 확인합니다.
"""

import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

torch.manual_seed(42)
np.random.seed(42)

CONTEXT = 8      # 몇 글자를 보고 다음 글자를 맞출지
EMB, HID = 24, 128


def quantize(w: np.ndarray, bits: int):
    """대칭 양자화: scale = max|w| / qmax, q = round(w / scale)."""
    qmax = 2 ** (bits - 1) - 1            # int8 -> 127, int4 -> 7
    scale = float(np.abs(w).max()) / qmax + 1e-12
    q = np.clip(np.round(w / scale), -qmax, qmax)
    return q.astype(np.int8), scale


def dequantize(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float32) * scale


def quantize_blockwise(w: np.ndarray, bits: int, block: int = 64):
    """1차원으로 펴서 block 개마다 따로 scale 을 두는 방식(GGUF 계열의 핵심 아이디어)."""
    flat = w.reshape(-1)
    out = np.empty_like(flat, dtype=np.float32)
    for i in range(0, flat.size, block):
        chunk = flat[i:i + block]
        q, s = quantize(chunk, bits)
        out[i:i + block] = dequantize(q, s)
    return out.reshape(w.shape)


class CharMLP(nn.Module):
    """CONTEXT 글자 임베딩을 이어붙여 다음 글자를 예측하는 초소형 언어모델."""

    def __init__(self, vocab: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, EMB)
        self.fc1 = nn.Linear(CONTEXT * EMB, HID)
        self.fc2 = nn.Linear(HID, vocab)

    def forward(self, x):
        h = self.emb(x).reshape(x.shape[0], -1)
        return self.fc2(torch.tanh(self.fc1(h)))


def make_batches(ids: torch.Tensor, n: int, batch: int, gen: torch.Generator):
    for _ in range(n):
        idx = torch.randint(0, len(ids) - CONTEXT - 1, (batch,), generator=gen)
        x = torch.stack([ids[i:i + CONTEXT] for i in idx])
        y = ids[idx + CONTEXT]
        yield x, y


def eval_loss(model: nn.Module, ids: torch.Tensor) -> float:
    gen = torch.Generator().manual_seed(7)  # 평가 배치도 고정
    model.eval()
    with torch.no_grad():
        losses = [F.cross_entropy(model(x), y).item()
                  for x, y in make_batches(ids, 20, 256, gen)]
    return sum(losses) / len(losses)


def apply_quant(model: nn.Module, fn) -> nn.Module:
    """모든 가중치 텐서에 양자화→복원 함수를 적용한 복사본 모델을 만듭니다."""
    import copy
    m = copy.deepcopy(model)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.from_numpy(fn(p.numpy().copy())))
    return m


if __name__ == "__main__":
    print("[1] 대칭 양자화 원리 — 작은 배열로 눈으로 확인")
    w = np.array([0.82, -1.30, 0.05, 0.44, -0.71], dtype=np.float32)
    q8, s8 = quantize(w, 8)
    restored = dequantize(q8, s8)
    print(f"    원본 fp32   : {np.round(w, 4)}")
    print(f"    int8 정수   : {q8}  (scale={s8:.6f})")
    print(f"    복원값      : {np.round(restored, 4)}")
    print(f"    최대 오차   : {np.abs(w - restored).max():.6f}")

    print("\n[2] 메모리 절감 — 파라미터 100만 개 기준")
    n_params = 1_000_000
    for name, bytes_per in [("fp32", 4), ("fp16", 2), ("int8", 1), ("int4", 0.5)]:
        mb = n_params * bytes_per / 1e6
        print(f"    {name:5s}: {mb:6.1f} MB  (fp32 대비 {4 / bytes_per:.0f}배 절감)")

    print("\n[3] 미니 언어모델 학습 (tiny_corpus, 문자 단위)")
    text = hjh_data.tiny_corpus()[:20000]
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    model = CharMLP(len(chars))
    n_p = sum(p.numel() for p in model.parameters())
    print(f"    어휘 {len(chars)}자, 파라미터 {n_p:,}개")
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(0)
    model.train()
    for step, (x, y) in enumerate(make_batches(ids, 400, 128, gen), 1):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"    step {step:3d}  loss {loss.item():.3f}")

    base = eval_loss(model, ids)
    print(f"\n[4] int8 양자화 후 성능 비교 (per-tensor)")
    m8 = apply_quant(model, lambda a: dequantize(*quantize(a, 8)))
    l8 = eval_loss(m8, ids)
    print(f"    fp32 loss = {base:.4f}")
    print(f"    int8 loss = {l8:.4f}  (증가 {l8 - base:+.4f}) → 거의 손해 없음")

    print("\n[5] int4 로 더 줄이면? + 이상치(outlier)와 블록 양자화")
    m4 = apply_quant(model, lambda a: dequantize(*quantize(a, 4)))
    l4 = eval_loss(m4, ids)
    print(f"    int4 loss = {l4:.4f}  (증가 {l4 - base:+.4f})")
    # 실제 대형 모델 가중치에는 유난히 큰 '이상치' 값이 섞여 있습니다.
    # 이상치 하나가 텐서 전체 scale 을 키워 나머지 정밀도를 망치는 것을 재현합니다.
    w_out = np.random.randn(1024).astype(np.float32) * 0.05
    w_out[100] = 3.0  # 이상치 하나 삽입
    err_tensor = np.abs(w_out - dequantize(*quantize(w_out, 4))).mean()
    err_block = np.abs(w_out - quantize_blockwise(w_out, 4, block=64)).mean()
    print(f"    이상치 포함 배열의 int4 평균 복원 오차:")
    print(f"      텐서 전체 scale 1개  : {err_tensor:.5f}")
    print(f"      64개 블록마다 scale  : {err_block:.5f}  ({err_tensor / err_block:.1f}배 정확)")
    print("    → 블록마다 scale 을 두면 큰 값 하나가 전체 정밀도를 망치지 않습니다.")
    print("    → llama.cpp 의 GGUF 포맷이 바로 이런 블록 양자화 가중치를 담는 파일입니다.")

    print("\n[6] 요약")
    print(f"    {'포맷':6s} {'크기(상대)':>10s} {'eval loss':>10s}")
    for name, rel, l in [("fp32", "1.00x", base), ("int8", "0.25x", l8),
                         ("int4", "0.125x", l4)]:
        print(f"    {name:6s} {rel:>10s} {l:>10.4f}")
    print("    핵심: 정밀도를 조금 포기하면 메모리는 4~8배 아낄 수 있고,")
    print("    블록 양자화 같은 기법이 그 '조금'을 더 작게 만들어 줍니다.")
