"""
PyTorch の基本素材であるテンソル(tensor)をツアーします。
生成/形の変更/ブロードキャスト/集計/行列積を段階ごとに実行し、
numpy 配列との相互変換(メモリ共有を含む)と device の概念を確認したあと
numpy <-> torch の API 対応表を出力します。
"""

import numpy as np
import torch


def main():
    torch.manual_seed(42)                 # 再現性: 乱数シード固定
    np.random.seed(42)

    print("[1] テンソル = numpy 配列 + 自動微分の能力 + GPU への引っ越し能力")
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    z = torch.zeros(2, 3)
    r = torch.arange(6)
    g = torch.randn(2, 3)                 # 標準正規乱数 (シード固定なので常に同じ)
    print(f"    tensor  : {a.tolist()}  shape={tuple(a.shape)}, dtype={a.dtype}")
    print(f"    zeros   : shape={tuple(z.shape)}")
    print(f"    arange  : {r.tolist()}")
    print(f"    randn   : 先頭要素 {g[0, 0].item():+.4f} (シード固定 -> 毎回同じ)\n")

    print("[2] 形の変更 — データはそのまま、見る窓だけを変えます")
    x = torch.arange(12, dtype=torch.float32)
    m = x.reshape(3, 4)
    print(f"    arange(12).reshape(3,4) ->\n{m}")
    print(f"    m.T shape = {tuple(m.T.shape)}, m.flatten() の長さ = {m.flatten().numel()}\n")

    print("[3] ブロードキャスト — 形が違っても規則に合えば自動で拡張")
    col = torch.tensor([[10.0], [20.0], [30.0]])      # (3,1)
    row = torch.tensor([1.0, 2.0, 3.0, 4.0])          # (4,)
    print(f"    (3,1) + (4,) -> {tuple((col + row).shape)} の行列:\n{col + row}\n")

    print("[4] 集計と行列積")
    print(f"    m.sum() = {m.sum().item():.0f}, m.mean(dim=0) = {m.mean(dim=0).tolist()}")
    W = torch.randn(4, 2)
    out = m @ W                            # ニューラルネットワーク 1 層の本質
    print(f"    (3,4) @ (4,2) = {tuple(out.shape)}  <- ニューラルネットワーク 1 層はまさにこの演算です\n")

    print("[5] numpy <-> torch 変換 — 同じメモリを共有します(CPU テンソル)")
    arr = np.ones(3, dtype=np.float32)
    t = torch.from_numpy(arr)              # コピーではなく「同じ紙を一緒に見ている」状態
    arr[0] = 99.0                          # numpy 側を書き換えると
    print(f"    numpy を 99 に修正 -> torch テンソルも {t.tolist()} (共有を確認)")
    back = t.numpy()
    t[1] = -7.0
    print(f"    torch を -7 に修正 -> numpy 配列も {back.tolist()}")
    safe = torch.tensor(arr)               # コピーが必要なら torch.tensor() / clone()
    arr[2] = 0.0
    print(f"    torch.tensor(arr) はコピー -> 元の修正とは無関係: {safe.tolist()}\n")

    print("[6] device — テンソルが「どの計算機の上に」あるのか")
    print(f"    既定の device: {a.device} (CPU)")
    print(f"    cuda は使える? {torch.cuda.is_available()}")
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    print(f"    mps(アップル GPU) は使える? {mps_ok}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    moved = a.to(device)                   # GPU があれば引っ越し、なければその場
    print(f"    a.to('{device}') -> device = {moved.device}")
    print("    # 実際の GPU コードもこの 1 行です: model.to('cuda'), batch.to('cuda')\n")

    print("[7] dtype — 精度(桁数)とメモリのトレードオフ")
    f32 = torch.randn(4)
    f16 = f32.to(torch.float16)
    print(f"    float32: {f32[0].item():+.8f} ({f32.element_size()}バイト/要素)")
    print(f"    float16: {f16[0].item():+.8f} ({f16.element_size()}バイト/要素、下の桁が失われる)\n")

    print("[8] numpy との API 対応表 — 知っている分だけそのまま使えます")
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
        ("(なし)",                 "a.to('cuda') — GPU への引っ越し"),
        ("(なし)",                 "a.requires_grad_() — 自動微分"),
    ]
    print(f"    {'numpy':34s}| torch")
    print(f"    {'-'*34}|{'-'*34}")
    for np_api, th_api in rows:
        print(f"    {np_api:34s}| {th_api}")
    print("\n[9] まとめ: numpy を知っていれば torch の 90% はすでに知っているのと同じです。")
    print("    残りの 10%(autograd, device)がディープラーニングを可能にする部分で、次のレベルの主題です。")


if __name__ == "__main__":
    main()
