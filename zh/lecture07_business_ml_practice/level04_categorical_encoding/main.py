"""
类别编码三连比较实验。
用 sales_table 的门店·品类·星期(文字)预测销售额的 Ridge 回归,
分别以独热 / 顺序 / 目标编码训练, 比较'列数 vs 性能'。
目标编码为防泄漏, 只在训练数据上计算平均。
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import mean_absolute_error, r2_score

CAT_COLS = ["store", "category", "weekday"]
NUM_COLS = ["ad_cost"]
TARGET = "revenue"


def fit_eval(X_tr, X_te, y_tr, y_te, label: str, n_cols: int):
    """钉死划分和模型, 只换编码结果来测性能。"""
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    mae = mean_absolute_error(y_te, pred)
    r2 = r2_score(y_te, pred)
    print(f"    {label:<12} 列 {n_cols:2d}个 | MAE {mae:10,.0f}韩元 | R2 {r2:.3f}")
    return {"方法": label, "列数": n_cols, "MAE": round(mae), "R2": round(r2, 3)}


def main() -> None:
    print("=" * 62)
    print(" 类别编码比较: 独热 vs 顺序 vs 目标")
    print("=" * 62)

    # [1] 数据清洗与类别确认 ---------------------------------------
    df = pd.DataFrame(hjh_data.sales_table(n_days=365, seed=42))
    n0 = len(df)
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0].reset_index(drop=True)
    print(f"\n[1] 清洗: {n0}行 -> {len(df)}行 (删除缺失·负数销售额)")
    for c in CAT_COLS:
        print(f"    {c:<9}: 类别 {df[c].nunique()}个 例) {sorted(df[c].unique())[:3]} ...")

    y = df[TARGET]
    tr_idx, te_idx = train_test_split(df.index, test_size=0.3, random_state=42)
    results = []

    # [2] 独热编码 ----------------------------------------------------
    print("\n[2] 独热: 每个类别开一列复选框 (无排序歪曲, 列数增加)")
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(df.loc[tr_idx, CAT_COLS])  # fit 只用训练数据!
    ohe_cols = list(ohe.get_feature_names_out(CAT_COLS))
    X_tr = pd.DataFrame(ohe.transform(df.loc[tr_idx, CAT_COLS]),
                        index=tr_idx, columns=ohe_cols)
    X_te = pd.DataFrame(ohe.transform(df.loc[te_idx, CAT_COLS]),
                        index=te_idx, columns=ohe_cols)
    X_tr[NUM_COLS], X_te[NUM_COLS] = df.loc[tr_idx, NUM_COLS], df.loc[te_idx, NUM_COLS]
    results.append(fit_eval(X_tr, X_te, y[tr_idx], y[te_idx], "独热", X_tr.shape[1]))

    # [3] 顺序编码 --------------------------------------------------
    print("\n[3] 顺序: 类别 -> 编号。列少, 但生出假排序")
    orde = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    orde.fit(df.loc[tr_idx, CAT_COLS])
    X_tr = pd.DataFrame(orde.transform(df.loc[tr_idx, CAT_COLS]),
                        index=tr_idx, columns=CAT_COLS)
    X_te = pd.DataFrame(orde.transform(df.loc[te_idx, CAT_COLS]),
                        index=te_idx, columns=CAT_COLS)
    X_tr[NUM_COLS], X_te[NUM_COLS] = df.loc[tr_idx, NUM_COLS], df.loc[te_idx, NUM_COLS]
    results.append(fit_eval(X_tr, X_te, y[tr_idx], y[te_idx], "顺序", X_tr.shape[1]))
    print("      ('浦东店=3 是天河店=1 的 3 倍'这条假信息, 线性模型会当真)")

    # [4] 目标编码 ----------------------------------------------------
    print("\n[4] 目标: 类别 -> '训练数据'的分组平均销售额 (小心泄漏!)")
    global_mean = y[tr_idx].mean()
    X_tr = pd.DataFrame(index=tr_idx)
    X_te = pd.DataFrame(index=te_idx)
    for c in CAT_COLS:
        # 平均必须只在训练数据上计算, 未见类别用全局平均
        means = df.loc[tr_idx].groupby(c)[TARGET].mean()
        X_tr[c + "_te"] = df.loc[tr_idx, c].map(means)
        X_te[c + "_te"] = df.loc[te_idx, c].map(means).fillna(global_mean)
    X_tr[NUM_COLS], X_te[NUM_COLS] = df.loc[tr_idx, NUM_COLS], df.loc[te_idx, NUM_COLS]
    results.append(fit_eval(X_tr, X_te, y[tr_idx], y[te_idx], "目标", X_tr.shape[1]))

    # [5] 汇总 ------------------------------------------------------------
    print("\n[5] 汇总: 列数 vs 性能")
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))
    best = summary.loc[summary["R2"].idxmax(), "方法"]
    print(f"\n    本轮数据的赢家: {best}")
    print("    怎么读:")
    print("      - 顺序编码的 R2 下滑 = 给无序类别编号付出的代价 (线性模型)")
    print("      - 目标编码 = 3 列逼近独热 -> 类别几千个(高基数)时是有力选项")
    print("      - 但目标编码只要用全量数据算平均, 立刻变成泄漏事故。")
    print("\n    教训: 编码不是技术选型, 而是'给模型灌输哪种世界观'的选择。")


if __name__ == "__main__":
    main()
