"""
不均衡データ(不正 1.5%)への対処法の比較実験。
基本モデル / class_weight / オーバーサンプリング / しきい値調整 の 4 通りで
再現率-適合率のトレードオフを表で確認し、
「1 日のアラート処理予算」の制約の下で合理的なしきい値を選んでみます。
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             average_precision_score, precision_recall_curve)

# 履歴集計システムがまだなく、「決済の瞬間に分かるフィールド」しか使えないと仮定
# (集計特徴量 tx_count_1h を加えると問題がどれだけ易しくなるかは「やってみよう」の課題)
FEATURES = ["amount", "hour", "is_foreign"]


def new_model(**kw):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(random_state=42, max_iter=1000, **kw))


def show(name, y_true, y_pred):
    print(f"    {name:<28} 精度 {accuracy_score(y_true, y_pred):.3f} | "
          f"再現率 {recall_score(y_true, y_pred):.3f} | "
          f"適合率 {precision_score(y_true, y_pred, zero_division=0):.3f} | "
          f"アラート {int(y_pred.sum())}件")


def main() -> None:
    print("=" * 66)
    print(" 不均衡データ: 1.5% の不正を捕まえる 4 つの方法")
    print("=" * 66)

    df = pd.DataFrame(hjh_data.fraud_table(n=5000, seed=11))
    X, y = df[FEATURES], df["is_fraud"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=42, stratify=y)
    print(f"\n    データ: {len(df)}件, 不正 {y.sum()}件({y.mean():.2%}) / "
          f"テスト {len(y_te)}件のうち不正 {y_te.sum()}件")

    # [1] 基本設定: 怠け者のモデル -----------------------------------------
    print("\n[1] 基本設定 (しきい値 0.5、重み付けなし)")
    base = new_model()
    base.fit(X_tr, y_tr)
    show("基本のロジスティック回帰", y_te, base.predict(X_te))
    missed = int(y_te.sum()) - int((base.predict(X_te) & y_te).sum())
    print(f"    => 精度 99% 台で立派に見えますが、不正 {int(y_te.sum())}件中 {missed}件を見逃しています。")
    print("       (極端な話、「全部正常」と予測しても精度 98.6% — 精度はこの問題の指標ではありません)")

    # [2] class_weight: 罰点表の書き換え -------------------------------------
    print("\n[2] class_weight='balanced' — 少数クラスのミスに大きな罰点")
    weighted = new_model(class_weight="balanced")
    weighted.fit(X_tr, y_tr)
    show("重み付けモデル", y_te, weighted.predict(X_te))

    # [3] オーバーサンプリング: 学習データにだけ! ------------------------------------
    print("\n[3] 手動オーバーサンプリング — 分割「後」に学習データの不正の行だけ複製")
    rng = np.random.default_rng(42)
    pos_idx = y_tr[y_tr == 1].index
    ratio = int((y_tr == 0).sum() / (y_tr == 1).sum())  # ほぼ均衡になる倍率
    dup_idx = rng.choice(pos_idx, size=len(pos_idx) * (ratio - 1), replace=True)
    X_bal = pd.concat([X_tr, X_tr.loc[dup_idx]])
    y_bal = pd.concat([y_tr, y_tr.loc[dup_idx]])
    print(f"    学習データ {len(y_tr)}件 -> {len(y_bal)}件 (不正の比率 {y_bal.mean():.1%})")
    over = new_model()
    over.fit(X_bal, y_bal)
    show("オーバーサンプリングモデル", y_te, over.predict(X_te))
    print("    => [2] と似た効果。どちらも「罰点表の書き換え」の変形です。")
    print("       (テストの比率は絶対に操作禁止 — 試験は現実世界の比率で)")

    # [4] しきい値調整: 同じモデル、違う運用点 ------------------------------
    print("\n[4] しきい値調整 — 基本モデル 1 つで運用点だけ移動")
    proba = base.predict_proba(X_te)[:, 1]
    print(f"    {'しきい値':>6} | {'再現率':>6} | {'適合率':>6} | アラート件数")
    rows = []
    for th in [0.9, 0.7, 0.5, 0.3, 0.2, 0.1]:
        pred = (proba >= th).astype(int)
        r = recall_score(y_te, pred)
        p = precision_score(y_te, pred, zero_division=0)
        rows.append((th, r, p, int(pred.sum())))
        print(f"    {th:6.2f} | {r:6.1%} | {p:6.1%} | {int(pred.sum()):4d}件")
    print("    => 下に行くほど再現率は上がり、適合率は下がります。タダ飯はありません。")

    # [5] PR 曲線の要約とアラート予算 -----------------------------------------
    print("\n[5] PR 曲線の要約と「アラート予算」で運用点を選ぶ")
    pr_auc = average_precision_score(y_te, proba)
    prec_c, rec_c, th_c = precision_recall_curve(y_te, proba)
    print(f"    PR-AUC(平均適合率) = {pr_auc:.3f}  (しきい値全区間の総合点)")
    budget = 15  # 調査チームがこのテスト期間に処理できるアラート件数
    order = np.argsort(proba)[::-1]
    top = order[:budget]
    caught = int(y_te.iloc[top].sum())
    th_budget = proba[order[budget - 1]]
    print(f"    制約: 調査チームの処理可能量 = {budget}件")
    print(f"    => 疑いスコア上位 {budget}件だけアラート (対応するしきい値 約 {th_budget:.2f})")
    print(f"       捕まえた不正 {caught}件 / 全体 {int(y_te.sum())}件 "
          f"(再現率 {caught / y_te.sum():.1%}, 適合率 {caught / budget:.1%})")
    print("\n    教訓: どの運用点を選ぶかはモデルではなくビジネス(人員・コスト)が決めます。")
    print("          モデルの仕事は「良い曲線」を作るところまでです。")


if __name__ == "__main__":
    main()
