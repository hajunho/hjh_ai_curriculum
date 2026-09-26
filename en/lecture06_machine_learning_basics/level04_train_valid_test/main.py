"""
level04 — Train/validation/test splits: proving the optimism bias of self-grading

Experiment A: how inflated do scores get when graded on the training data
              (gap by tree depth)
Experiment B: if you 'peek' at the test set to choose settings, how
              systematically inflated is the final reported score
              (statistics over 30 repeats)
Lesson: train=textbook, validation=practice exam, test=board exam (once, at the end).
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

K_CANDIDATES = list(range(1, 30, 2))     # 15 candidate settings (hyperparameters)


def make_data(seed: int, n: int = 1500):
    """Noisy binary-classification synthetic data (no download, reproducible by seed)."""
    return make_classification(
        n_samples=n, n_features=8, n_informative=4, n_redundant=2,
        flip_y=0.08,             # 8% of labels are pure noise -> the temptation to memorize
        class_sep=0.9, random_state=seed)


def experiment_a() -> None:
    """[2] Experiment A: training score vs test score — the gap widens with flexibility."""
    X, y = make_data(seed=0, n=1000)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    print("[2] Experiment A — the optimism bias of self-grading (grading on training data)")
    print("    Tree depth   train accuracy   test accuracy   gap")
    for depth in [1, 2, 4, 8, 16, None]:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_tr, y_tr)
        acc_tr = tree.score(X_tr, y_tr)
        acc_te = tree.score(X_te, y_te)
        label = "unlimited" if depth is None else f"{depth:>4}"
        print(f"    {label:>9}      {acc_tr:6.1%}          {acc_te:6.1%}      {acc_tr - acc_te:+6.1%}")
    print("    -> The deeper the tree, the closer training accuracy gets to 100% (memorizing")
    print("       past exams) while real-world accuracy falls. A training score proves nothing.\n")


def experiment_b(n_repeats: int = 30) -> None:
    """[3] Experiment B: test-peeking vs a proper 3-way split — statistics over 30 repeats.
    A 'new data' set (never used in model selection) stands in for true ability."""
    inflate_peek, inflate_proper = [], []
    for rep in range(n_repeats):
        X, y = make_data(seed=100 + rep)
        # 60:20:20 split + a separate 'new data' set for measuring true ability
        X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, random_state=rep)
        X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=rep)
        X_new, y_new = make_data(seed=9000 + rep, n=800)   # data met after deployment

        models = {k: KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr)
                  for k in K_CANDIDATES}

        # (a) Cheating: choose k by watching the test score, then report that score as-is
        k_peek = max(K_CANDIDATES, key=lambda k: models[k].score(X_te, y_te))
        reported_peek = models[k_peek].score(X_te, y_te)
        real_peek = models[k_peek].score(X_new, y_new)
        inflate_peek.append(reported_peek - real_peek)

        # (b) By the book: choose k with validation; touch the test exactly once at the end
        k_ok = max(K_CANDIDATES, key=lambda k: models[k].score(X_va, y_va))
        reported_ok = models[k_ok].score(X_te, y_te)
        real_ok = models[k_ok].score(X_new, y_new)
        inflate_proper.append(reported_ok - real_ok)

    peek = np.array(inflate_peek)
    proper = np.array(inflate_proper)
    print(f"[3] Experiment B — two ways to choose among {len(K_CANDIDATES)} candidate k's, repeated {n_repeats}x")
    print("    inflation = (reported score) - (true score on new data)")
    print(f"    (a) Chosen by peeking at the test: mean inflation {peek.mean():+.2%} (std {peek.std():.2%})")
    print(f"    (b) Chosen by validation (proper): mean inflation {proper.mean():+.2%} (std {proper.std():.2%})")
    print(f"    Times (a) was more inflated than (b): {int((peek > proper).sum())}/{n_repeats}")
    print("    -> Peeking inflation is not chance — it is structural.")
    print("       The more candidates, the more you pick 'the setting that got lucky on that exam paper'.\n")


if __name__ == "__main__":
    np.random.seed(0)

    # [1] Introducing the 3-way split -----------------------------------------
    X, y = make_data(seed=0, n=1000)
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=0)
    print("[1] Splitting 1000 rows 60:20:20")
    print(f"    Train (textbook) {len(X_tr)} / validation (practice exam) {len(X_va)} / test (board exam) {len(X_te)}\n")

    experiment_a()
    experiment_b()

    print("[4] Summary")
    print("    1. Grading on training data is always optimistic (experiment A's gap).")
    print("    2. Choosing settings by the test contaminates the final report too (experiment B).")
    print("    3. Whenever you see a performance number in a report, always ask:")
    print("       \"That number — measured on which data? And how many times did you look at that data?\"")
