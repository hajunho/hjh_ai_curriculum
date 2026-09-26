"""
混同行列をウォン建て金額に換算する計算機。
解約予測モデル (ロジスティック回帰) の TP/FP/FN/TN にビジネス単価を付けて
「このモデルはいくらの価値か」「再現率 1%pt はいくらか」を計算します。
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, recall_score, precision_score

# ---- ビジネス単価 (財務/マーケティング部門からもらう数字と仮定) ----
VALUE_V = 179_000   # 顧客残存価値: 守れば得られる売上 (ウォン/人)
COST_C = 12_000     # 介入コスト: クーポン + 相談の原価 (ウォン/人)
SUCCESS_S = 0.30    # 介入成功率: クーポンを受け取った解約予備軍が残留する確率

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


def profit_of(tp: int, fp: int) -> float:
    """混同行列 -> 期待利益(ウォン)。TP は一部を救い出し、FP はコストだけ使います。"""
    return tp * (VALUE_V * SUCCESS_S - COST_C) - fp * COST_C


def main() -> None:
    print("=" * 62)
    print(" 混同行列 -> ウォン翻訳機: 解約モデルはいくらの価値か")
    print("=" * 62)

    # [1] データ準備とモデル訓練 ------------------------------------
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES], df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    # class_weight="balanced": 解約(少数クラス)を見逃さないよう重み付け (level07 で詳しく)
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(random_state=42, class_weight="balanced"))
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    print(f"\n[1] ロジスティック回帰の訓練完了 (学習 {len(X_tr)}名 / テスト {len(X_te)}名, "
          f"テスト解約率 {y_te.mean():.1%})")

    # [2] 混同行列 ----------------------------------------------------
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    rec = recall_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    print("\n[2] テスト混同行列 (陽性 = 解約)")
    print(f"    TP(解約を的中)={tp:4d}  FN(解約を見逃し)={fn:4d}")
    print(f"    FP(空振り)    ={fp:4d}  TN(正常を通過) ={tn:4d}")
    print(f"    再現率 {rec:.1%} / 適合率 {prec:.1%}")

    # [3] マスごとに単価を付ける -----------------------------------------
    print("\n[3] ビジネス単価で換算")
    print(f"    単価: 顧客価値 V={VALUE_V:,}ウォン, 介入コスト C={COST_C:,}ウォン, 成功率 s={SUCCESS_S:.0%}")
    unit_tp = VALUE_V * SUCCESS_S - COST_C
    print(f"    TP 1件の価値 = V*s - C = {unit_tp:+,.0f}ウォン")
    print(f"    FP 1件の価値 = -C      = {-COST_C:+,}ウォン")
    print(f"    FN 1件 = 支出 0ウォン、ただし機会損失 V*s = {VALUE_V*SUCCESS_S:,.0f}ウォン")
    model_profit = profit_of(tp, fp)
    print(f"    => モデルの期待利益(テスト {len(X_te)}名基準): {model_profit:+,.0f}ウォン")

    # [4] 比較対象: 何もしない vs 全員クーポン ------------------------
    print("\n[4] 戦略の比較 (同じテスト顧客が基準)")
    n_pos = int(y_te.sum())
    do_nothing = profit_of(0, 0)
    give_all = profit_of(n_pos, len(y_te) - n_pos)  # 全員に介入: 解約者は全員 TP、残りは全員 FP
    print(f"    A. 何もしない                : {do_nothing:+13,.0f}ウォン")
    print(f"    B. 全員にクーポンを送付      : {give_all:+13,.0f}ウォン")
    print(f"    C. モデルが選んだ人にだけ    : {model_profit:+13,.0f}ウォン")
    best = max([("A", do_nothing), ("B", give_all), ("C", model_profit)], key=lambda t: t[1])
    print(f"    => 最善の戦略: {best[0]} — モデルの価値は「誰に使うかを選ぶ能力」です。")

    # [5] 再現率 1%pt の価値 ------------------------------------------
    print("\n[5] 「再現率 1%pt」はいくらなのか")
    n_customers = 100_000            # 実際のサービス規模に拡張
    churn_rate = float(y.mean())     # データから推定した解約率
    value_1pp = n_customers * churn_rate * 0.01 * unit_tp
    print(f"    公式: N * 解約率 * 0.01 * (V*s - C)")
    print(f"        = {n_customers:,}名 * {churn_rate:.1%} * 1%pt * {unit_tp:,.0f}ウォン")
    print(f"        = 約 {value_1pp:,.0f}ウォン (月次キャンペーンなら年間 約 {value_1pp*12:,.0f}ウォン)")
    print("\n    報告文の例:")
    print(f'    「再現率を {rec:.0%} から {rec+0.05:.0%} へ 5%pt 上げると、')
    print(f'     年間 約 {value_1pp*5*12/1e8:.1f}億ウォンの解約防止利益が追加されます。」')
    print("\n    教訓: ML 指標は単価を付けた瞬間、予算の言葉になります。")


if __name__ == "__main__":
    main()
