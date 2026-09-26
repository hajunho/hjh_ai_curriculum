"""
PyTorch の autograd(自動微分)とオプティマイザを学びます。
(1) スカラーの例で backward() が計算した勾配を手計算と照らし合わせ、
(2) level04 で numpy の逆伝播 ~30 行で解いた XOR を
    loss.backward() の 1 行 + オプティマイザで解き直してコードの縮小を体感し、
(3) SGD と Adam の収束の速さを同じ条件で比較します。
"""

import torch
import torch.nn as nn


def make_xor():
    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = torch.tensor([[0.], [1.], [1.], [0.]])
    return X, y


def build_model():
    """level04 と同じ 2-8-1 構造。層を定義するだけで微分は autograd の仕事です。"""
    return nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1), nn.Sigmoid())


def train_xor(opt_name, lr, max_epochs=2000):
    """XOR の学習ループ。戻り値: (loss<0.05 に到達したエポック, 最終 loss)"""
    torch.manual_seed(0)                       # 2 つのオプティマイザに同じ初期重みを
    X, y = make_xor()
    model = build_model()
    loss_fn = nn.BCELoss()                     # level04 で手書きした BCE と同じ
    if opt_name == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    reached = None
    for epoch in range(1, max_epochs + 1):
        opt.zero_grad()                        # (1) 前の勾配の帳簿を白紙に
        p = model(X)                           # (2) 順伝播
        loss = loss_fn(p, y)                   # (3) 採点
        loss.backward()                        # (4) 逆伝播 — level04 の 30 行がこの 1 行
        opt.step()                             # (5) 重みの更新
        acc = ((p > 0.5) == y.bool()).float().mean().item()
        if loss.item() < 0.05 and reached is None:
            reached = epoch                    # 「事実上もう覚えた」時点
        if epoch in (1, 100, 500, 1000, max_epochs):
            print(f"      epoch {epoch:4d}: loss={loss.item():.4f}, 精度={acc:.0%}")
    return reached, loss.item()


def main():
    torch.manual_seed(0)                       # 再現性

    print("[1] autograd の味見 — y = x^2 + 3x を x=2 で微分")
    x = torch.tensor(2.0, requires_grad=True)  # 「計算履歴を記録せよ」スイッチ ON
    y = x ** 2 + 3 * x
    y.backward()                               # 帳簿を逆向きにたどって dy/dx を計算
    print(f"    autograd: dy/dx = {x.grad.item():.1f}")
    print(f"    手計算  : dy/dx = 2x + 3 = 2*2 + 3 = 7.0  -> 一致!\n")

    print("[2] zero_grad が必要な理由 — 勾配は「上書き」ではなく「累積」")
    x.grad.zero_()                             # 帳簿を初期化
    (x ** 2 + 3 * x).backward()
    (x ** 2 + 3 * x).backward()                # 初期化せずにもう一度
    print(f"    2 回 backward した後の grad = {x.grad.item():.1f} (7 ではなく 14 = 7+7 の累積)")
    print("    => 毎ステップ opt.zero_grad() で帳簿を空にする必要があります。\n")

    print("[3] XOR に再挑戦 — level04 (numpy の手作り) vs autograd")
    print("    level04: forward 4 行 + backward 15 行 + 検証 20 行を手で書いた")
    print("    今回  : モデル定義 1 行 + loss.backward() + opt.step() でおしまい\n")

    print("    [3-1] SGD (lr=0.5) — level04 と同じやり方の勾配降下")
    sgd_epoch, sgd_loss = train_xor("SGD", lr=0.5)

    print("    [3-2] Adam (lr=0.05) — 方向(モメンタム)と歩幅(適応学習率)を自動調節")
    adam_epoch, adam_loss = train_xor("Adam", lr=0.05)

    print("\n[4] オプティマイザの比較 (同じ初期重み、同じデータ)")
    print(f"    {'オプティマイザ':10s} {'loss<0.05 到達エポック':>14s} {'最終 loss':>12s}")
    print(f"    {'SGD':10s} {str(sgd_epoch):>16s} {sgd_loss:>12.5f}")
    print(f"    {'Adam':10s} {str(adam_epoch):>15s} {adam_loss:>12.5f}")
    assert sgd_epoch is not None and adam_epoch is not None, "XOR の収束に失敗"
    print("    Adam は勾配の移動平均(方向の慣性)と大きさの補正(パラメータごとの歩幅)を使って")
    print("    学習率に鈍感で、たいていより速く収束します。実務の既定値が Adam(系)である理由です。\n")

    print("[5] まとめ: 「微分を自動記録する帳簿(autograd)」+「下山の戦略(optimizer)」の組み合わせが")
    print("    すべての PyTorch 学習コードの 5 段階ループ(zero_grad -> forward -> loss -> backward -> step)です。")


if __name__ == "__main__":
    main()
