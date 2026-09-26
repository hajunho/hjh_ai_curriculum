"""
Learn PyTorch autograd (automatic differentiation) and optimizers.
(1) On a scalar example, check the gradient computed by backward() against a hand calculation,
(2) re-solve the XOR problem — which took ~30 lines of numpy backprop in level04 —
    with one loss.backward() line plus an optimizer, feeling the code shrink,
(3) and compare the convergence speed of SGD and Adam under identical conditions.
"""

import torch
import torch.nn as nn


def make_xor():
    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = torch.tensor([[0.], [1.], [1.], [0.]])
    return X, y


def build_model():
    """The same 2-8-1 structure as level04. You only define the layers; differentiation is autograd's job."""
    return nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1), nn.Sigmoid())


def train_xor(opt_name, lr, max_epochs=2000):
    """XOR training loop. Returns: (epoch reaching loss<0.05, final loss)"""
    torch.manual_seed(0)                       # same initial weights for both optimizers
    X, y = make_xor()
    model = build_model()
    loss_fn = nn.BCELoss()                     # identical to the BCE hand-written in level04
    if opt_name == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    reached = None
    for epoch in range(1, max_epochs + 1):
        opt.zero_grad()                        # (1) wipe the previous gradient ledger clean
        p = model(X)                           # (2) forward pass
        loss = loss_fn(p, y)                   # (3) grade it
        loss.backward()                        # (4) backward pass — level04's 30 lines are this one line
        opt.step()                             # (5) weight update
        acc = ((p > 0.5) == y.bool()).float().mean().item()
        if loss.item() < 0.05 and reached is None:
            reached = epoch                    # the point where it has 'effectively learned it all'
        if epoch in (1, 100, 500, 1000, max_epochs):
            print(f"      epoch {epoch:4d}: loss={loss.item():.4f}, accuracy={acc:.0%}")
    return reached, loss.item()


def main():
    torch.manual_seed(0)                       # reproducibility

    print("[1] A taste of autograd — differentiate y = x^2 + 3x at x=2")
    x = torch.tensor(2.0, requires_grad=True)  # the 'record my computations' switch, ON
    y = x ** 2 + 3 * x
    y.backward()                               # sweep the ledger backwards to compute dy/dx
    print(f"    autograd: dy/dx = {x.grad.item():.1f}")
    print(f"    by hand : dy/dx = 2x + 3 = 2*2 + 3 = 7.0  -> match!\n")

    print("[2] Why zero_grad is needed — gradients 'accumulate', they don't 'overwrite'")
    x.grad.zero_()                             # reset the ledger
    (x ** 2 + 3 * x).backward()
    (x ** 2 + 3 * x).backward()                # once more, without resetting
    print(f"    grad after two backwards = {x.grad.item():.1f} (not 7 but 14 = 7+7 accumulated)")
    print("    => You must empty the ledger every step with opt.zero_grad().\n")

    print("[3] XOR, take two — level04 (hand-made numpy) vs autograd")
    print("    level04: 4 lines of forward + 15 lines of backward + 20 lines of checks, written by hand")
    print("    now    : 1 line of model definition + loss.backward() + opt.step() — that's everything\n")

    print("    [3-1] SGD (lr=0.5) — gradient descent the same way as level04")
    sgd_epoch, sgd_loss = train_xor("SGD", lr=0.5)

    print("    [3-2] Adam (lr=0.05) — auto-adjusts direction (momentum) and stride (adaptive learning rate)")
    adam_epoch, adam_loss = train_xor("Adam", lr=0.05)

    print("\n[4] Optimizer comparison (same initial weights, same data)")
    print(f"    {'optimizer':10s} {'epoch loss<0.05':>16s} {'final loss':>12s}")
    print(f"    {'SGD':10s} {str(sgd_epoch):>16s} {sgd_loss:>12.5f}")
    print(f"    {'Adam':10s} {str(adam_epoch):>16s} {adam_loss:>12.5f}")
    assert sgd_epoch is not None and adam_epoch is not None, "XOR failed to converge"
    print("    Adam uses a moving average of gradients (directional inertia) and a magnitude correction")
    print("    (per-parameter stride), so it is less learning-rate sensitive and usually converges faster.")
    print("    That is why the workplace default is Adam (or a variant).\n")

    print("[5] Recap: the combination of an 'automatic derivative ledger (autograd)' and a 'descent strategy (optimizer)'")
    print("    is the 5-step loop of all PyTorch training code (zero_grad -> forward -> loss -> backward -> step).")


if __name__ == "__main__":
    main()
