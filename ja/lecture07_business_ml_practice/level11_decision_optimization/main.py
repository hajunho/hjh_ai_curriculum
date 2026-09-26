"""
モデルを意思決定へ — 「誰にクーポンを配るか」利益最大化の実験。
解約確率に期待価値の計算(p*V*s - C)を重ねて
理論の損益分岐しきい値 p* = C/(V*s) と実験の最適しきい値をクロスチェックし、
利益曲線の PNG とキャンペーン ROI シナリオ報告を作ります。
"""

import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# ---- ビジネス単価 (level01 と同じ) ----
VALUE_V = 179_000   # 顧客残存価値 (ウォン)
COST_C = 12_000     # 介入コスト: クーポン + 相談 (ウォン)
SUCCESS_S = 0.30    # 介入成功率

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


def realized_profit(proba, y_true, threshold, s=SUCCESS_S):
    """しきい値以上の顧客に介入したときの、テストデータでの実現利益。
    実際に解約予定の顧客なら成功率 s の分だけ V を守り、そうでなければクーポン費用だけが出る。"""
    target = proba >= threshold
    y = np.asarray(y_true)
    tp = int((target & (y == 1)).sum())     # 解約予定の顧客への介入
    fp = int((target & (y == 0)).sum())     # 残留顧客への介入 (コストの無駄)
    return tp * (VALUE_V * s - COST_C) - fp * COST_C, int(target.sum())


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 68)
    print(" モデルを意思決定へ: 利益を最大化するクーポンしきい値を探す")
    print("=" * 68)

    # [1] 解約モデルと確率 --------------------------------------------------
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES], df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=42, stratify=y)
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    print(f"\n[1] 解約モデルの訓練完了。テスト {len(y_te)}名の解約確率を算出")
    print(f"    (参考: class_weight なしで訓練 — 期待価値の計算には較正された確率が必要)")

    # [2] 理論の損益分岐確率 p* ---------------------------------------------
    p_star = COST_C / (VALUE_V * SUCCESS_S)
    print("\n[2] 理論の損益分岐確率 (算数 1 行)")
    print(f"    介入の期待利益 = p*V*s - C > 0  <=>  p > C/(V*s)")
    print(f"    p* = {COST_C:,} / ({VALUE_V:,} x {SUCCESS_S:.0%}) = {p_star:.3f}")
    print(f"    => 解約確率 {p_star:.1%} 超の顧客にだけクーポンを配るのが理論最適。")
    print("       慣習的なしきい値 0.5 には何の根拠もありません。")

    # [3] 実験: しきい値をなぞる --------------------------------------------------
    print("\n[3] 実験: しきい値 0 -> 1 をなぞりながらテストの実現利益を計算")
    grid = np.arange(0.0, 1.001, 0.01)
    profits = np.array([realized_profit(proba, y_te, t)[0] for t in grid])
    best_i = int(np.argmax(profits))
    best_th = float(grid[best_i])
    print(f"    実験の最適しきい値 = {best_th:.2f} (実現利益 {profits[best_i]:+,.0f}ウォン)")
    print(f"    理論値 {p_star:.3f} との距離 = {abs(best_th - p_star):.3f}")
    print("    => 両者が近ければ確率の較正が使い物になるという意味。遠ければまず較正を点検!")

    # [4] 戦略比較表 --------------------------------------------------------
    print("\n[4] 戦略の比較 (テスト顧客基準の実現利益)")
    strategies = [
        ("A. 何もしない", 1.01),
        ("B. 全員クーポン", 0.0),
        ("C. 慣習のしきい値 0.5", 0.5),
        ("D. 理論 p* しきい値", p_star),
        ("E. 実験の最適しきい値", best_th),
    ]
    for name, th in strategies:
        profit, n_target = realized_profit(proba, y_te, th)
        print(f"    {name:<18} 介入 {n_target:>4}名 | 利益 {profit:>+12,.0f}ウォン")
    c_profit, _ = realized_profit(proba, y_te, 0.5)
    d_profit, _ = realized_profit(proba, y_te, p_star)
    if c_profit > 0:
        print(f"    => モデルはそのまま、しきい値だけ 0.5 -> p* に変えて利益 {d_profit/c_profit:.1f}倍。")
    print("       「モデル改善」より「意思決定の設計」が先である理由です。")

    # [5] 利益曲線 PNG ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grid, profits / 1e4, color="#4477aa", lw=2)
    ax.axvline(p_star, color="#228833", ls="--", lw=1.5,
               label=f"theory p* = {p_star:.3f}")
    ax.axvline(best_th, color="#cc6677", ls=":", lw=1.5,
               label=f"empirical best = {best_th:.2f}")
    ax.axvline(0.5, color="gray", ls="-.", lw=1, label="convention 0.5")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("coupon threshold (churn probability)")
    ax.set_ylabel("realized profit (10k KRW)")
    ax.set_title("Profit vs threshold: who should get the coupon?")
    ax.legend()
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "profit_curve.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f"\n[5] 利益曲線の保存: {png}")

    # [6] キャンペーン ROI シナリオ (10 万顧客規模) -------------------------------
    print("\n[6] キャンペーン ROI 報告 — 顧客 10 万人規模、成功率 3 シナリオ")
    n_scale = 100_000 / len(y_te)
    target = proba >= best_th
    n_target = int(target.sum() * n_scale)
    cost = n_target * COST_C
    print(f"    介入対象: 約 {n_target:,}名 / クーポン予算: {cost/1e4:,.0f}万ウォン")
    print(f"    {'シナリオ':<12} {'成功率':>5} | {'純利益':>12} | {'ROI':>7}")
    for label, s_val in [("保守的", 0.20), ("基本", 0.30), ("楽観", 0.40)]:
        profit, _ = realized_profit(proba, y_te, best_th, s=s_val)
        profit_scaled = profit * n_scale
        roi = profit_scaled / cost * 100
        print(f"    {label:<12} {s_val:>5.0%} | {profit_scaled/1e4:>+10,.0f}万ウォン | {roi:>6.0f}%")
    base_profit, _ = realized_profit(proba, y_te, best_th, s=0.30)
    base_roi = base_profit * n_scale / cost * 100
    print("\n    企画書の文例:")
    print(f'    「解約確率 {best_th:.0%} 超の顧客 約 {n_target:,}名にクーポンを提供すれば、')
    print(f'     基本シナリオ(成功率 30%)での期待 ROI は約 {base_roi:.0f}% です。')
    print(f'     成功率はキャンペーンの対照群で実測し、来四半期に再計算します。」')
    print("\n    教訓: モデルの確率に値札(V, C, s)を掛けた瞬間、")
    print("          機械学習は統計ではなく経営の道具になります。")


if __name__ == "__main__":
    main()
