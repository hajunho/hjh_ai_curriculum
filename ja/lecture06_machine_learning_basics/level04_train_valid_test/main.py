"""
level04 — 学習・検証・テストの分割: 自己採点の楽観バイアスを実験で証明

実験 A: 訓練データで採点すると成績がどれだけ水増しされるか (木の深さ別の gap)
実験 B: テストセットを「盗み見」しながら設定を選ぶと、最終報告の成績が
        どれだけ構造的に水増しされるか (30回反復の統計)
教訓: 訓練=教科書、検証=模擬試験、テスト=本番の入試 (最後に 1 回だけ)。
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

K_CANDIDATES = list(range(1, 30, 2))     # 設定(ハイパーパラメータ)の候補 15個


def make_data(seed: int, n: int = 1500):
    """ノイズ入りの二値分類合成データ (ダウンロード不要、seed で再現)。"""
    return make_classification(
        n_samples=n, n_features=8, n_informative=4, n_redundant=2,
        flip_y=0.08,             # 8% はラベル自体がノイズ -> 丸暗記の誘惑が生まれる
        class_sep=0.9, random_state=seed)


def experiment_a() -> None:
    """[2] 実験 A: 訓練成績 vs テスト成績 — 柔軟なモデルほど開く間隔。"""
    X, y = make_data(seed=0, n=1000)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    print("[2] 実験 A — 自己採点 (訓練データでの採点) の楽観バイアス")
    print("    木の深さ   訓練の正解率   テストの正解率   間隔(gap)")
    for depth in [1, 2, 4, 8, 16, None]:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_tr, y_tr)
        acc_tr = tree.score(X_tr, y_tr)
        acc_te = tree.score(X_te, y_te)
        label = "無制限" if depth is None else f"{depth:>4}"
        print(f"    {label:>7}      {acc_tr:6.1%}        {acc_te:6.1%}       {acc_tr - acc_te:+6.1%}")
    print("    -> 深くなるほど訓練成績は 100% に向かう(過去問の暗記)一方で、実戦成績は下落。")
    print("       訓練成績は実力の証明ではありません。\n")


def experiment_b(n_repeats: int = 30) -> None:
    """[3] 実験 B: テストの盗み見 vs 正しい 3 分割 — 30回反復の統計。
    「新データ」セット (モデル選択に一切使っていないデータ) を本当の実力とみなす。"""
    inflate_peek, inflate_proper = [], []
    for rep in range(n_repeats):
        X, y = make_data(seed=100 + rep)
        # 60:20:20 の分割 + 本当の実力測定用の「新データ」は別途生成
        X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, random_state=rep)
        X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=rep)
        X_new, y_new = make_data(seed=9000 + rep, n=800)   # デプロイ後に出会うデータ

        models = {k: KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr)
                  for k in K_CANDIDATES}

        # (a) 反則: テストの成績を見ながら k を選び、その成績をそのまま報告
        k_peek = max(K_CANDIDATES, key=lambda k: models[k].score(X_te, y_te))
        reported_peek = models[k_peek].score(X_te, y_te)
        real_peek = models[k_peek].score(X_new, y_new)
        inflate_peek.append(reported_peek - real_peek)

        # (b) 正攻法: 検証で k を選び、テストは最後に 1 回だけ
        k_ok = max(K_CANDIDATES, key=lambda k: models[k].score(X_va, y_va))
        reported_ok = models[k_ok].score(X_te, y_te)
        real_ok = models[k_ok].score(X_new, y_new)
        inflate_proper.append(reported_ok - real_ok)

    peek = np.array(inflate_peek)
    proper = np.array(inflate_proper)
    print(f"[3] 実験 B — 設定 k の候補 {len(K_CANDIDATES)}個を選ぶ 2 つの方式、{n_repeats}回反復")
    print("    水増し = (報告した成績) - (新データでの本当の成績)")
    print(f"    (a) テストを盗み見て選択: 平均の水増し {peek.mean():+.2%} (標準偏差 {peek.std():.2%})")
    print(f"    (b) 検証で選択(正攻法)  : 平均の水増し {proper.mean():+.2%} (標準偏差 {proper.std():.2%})")
    print(f"    (a) が (b) より水増しされた回数: {int((peek > proper).sum())}/{n_repeats}回")
    print("    -> 盗み見の水増しは偶然ではなく構造的です。")
    print("       候補が多いほど「その試験でたまたま良かった設定」を選んでしまいます。\n")


if __name__ == "__main__":
    np.random.seed(0)

    # [1] 3 分割の紹介 --------------------------------------------------------
    X, y = make_data(seed=0, n=1000)
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=0)
    print("[1] データ 1000件を 60:20:20 に 3 分割")
    print(f"    訓練(教科書) {len(X_tr)}件 / 検証(模擬試験) {len(X_va)}件 / テスト(本番入試) {len(X_te)}件\n")

    experiment_a()
    experiment_b()

    print("[4] まとめ")
    print("    1. 訓練データでの採点は常に楽観的である (実験 A の gap)。")
    print("    2. テストで設定を選ぶと、最終報告まで汚染される (実験 B)。")
    print("    3. 報告書の性能の数字を見たら、必ずこう尋ねること:")
    print("       「その数字、どのデータで測ったのですか? そのデータを何回見ましたか?」")
