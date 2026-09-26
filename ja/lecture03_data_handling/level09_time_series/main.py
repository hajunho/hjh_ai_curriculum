"""
時系列 (time series) データの扱い方の実習。
カフェチェーンの売上を時間軸の上に載せて
  - 間違った日付文字列の診断 (to_datetime errors="coerce")
  - 日別 -> 週別/月別のリサンプリング (resample)
  - 7日移動平均 (rolling) と前週比の成長率 (shift / pct_change)
を順に実行し、トレンドのグラフを outputs/ に保存します。
"""

import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")  # 画面なしでファイルにだけ図を保存します。
import matplotlib.pyplot as plt
import pandas as pd

# 共通データモジュール (hjh_data) を読み込むためのパス設定
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"


def load_clean_sales() -> pd.DataFrame:
    """売上データを作り、欠損とマイナスの汚染を取り除きます (level06 の復習)。"""
    df = pd.DataFrame(hjh_data.sales_table(n_days=365, seed=42))
    before = len(df)
    df = df.dropna(subset=["revenue"])          # 欠損した売上を除去
    df = df[df["revenue"] > 0].copy()           # マイナスの異常値を除去
    print(f"    クレンジング: {before:,}行 -> {len(df):,}行 (欠損・マイナス {before - len(df)}件を除去)")
    return df


def step1_diagnose_dates(df: pd.DataFrame) -> pd.DataFrame:
    """[1] 日付文字列の汚染を診断し、信頼できる日付を新しく作ります。"""
    print("\n[1] 日付の汚染の診断 — 文字列はまだ日付ではありません")
    parsed = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    n_bad = int(parsed.isna().sum())
    bad_examples = df.loc[parsed.isna(), "date"].unique()[:5]
    print(f"    errors='coerce' での変換結果: NaT (日付の欠損) が {n_bad:,}件 発生")
    print(f"    カレンダーに存在しない日付の例: {list(bad_examples)}")
    print("    -> このデータの date 列は「1か月=31日」という前提で振られていて信頼できません。")

    # 信頼できる day_index (開店後の経過日数) から本物の日付を再構成します。
    df = df.copy()
    df["real_date"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(df["day_index"], unit="D")
    print(f"    day_index から再構成した期間: {df['real_date'].min().date()} ~ {df['real_date'].max().date()}")
    return df


def step2_daily_series(df: pd.DataFrame) -> pd.Series:
    """[2] 全社の日別売上の時系列を作り、DatetimeIndex を確認します。"""
    print("\n[2] 日別の全社売上の時系列を作る")
    daily = df.groupby("real_date")["revenue"].sum().sort_index()
    print(f"    日数: {len(daily)}日 / インデックスの型: {type(daily.index).__name__}")
    print(f"    最初の3日:\n{(daily.head(3) / 1e6).round(1).to_string()}  (単位: 百万ウォン)")
    # dt アクセサ: 日付の列から「月」のような部品を取り出せます。
    month_of_first_rows = df["real_date"].dt.month.head(3).tolist()
    print(f"    dt アクセサの例 — 先頭3行の月: {month_of_first_rows}")
    return daily


def step3_resample(daily: pd.Series) -> pd.Series:
    """[3] resample: 日別 -> 週別 -> 月別へ時間の単位を変えます。"""
    print("\n[3] resample — 日別データを週間・月間の箱に入れて合算")
    weekly = daily.resample("W").sum()
    monthly = daily.resample("ME").sum()
    print(f"    週間の合計: {len(weekly)}週分 (最初の週は部分的な週なので値が小さいことがあります)")
    print(f"    月間の合計 (単位: 億ウォン):")
    for ts, val in monthly.items():
        print(f"      {ts.strftime('%Y-%m')}: {val / 1e8:6.2f}億")
    return weekly


def step4_rolling(daily: pd.Series) -> pd.Series:
    """[4] rolling(7): 7日移動平均で曜日による揺れを消します。"""
    print("\n[4] rolling(7) — 移動平均でノイズを除去")
    ma7 = daily.rolling(7).mean()
    print(f"    最初の6日は窓を満たせないので NaN: 先頭の NaN の数 = {int(ma7.isna().sum())}")
    sample_day = daily.index[9]
    print(f"    例) {sample_day.date()} の元データ {daily.iloc[9]/1e6:.1f}百万ウォン "
          f"vs 7日平均 {ma7.iloc[9]/1e6:.1f}百万ウォン")
    weekend_std = daily.std()
    smooth_std = ma7.dropna().std()
    print(f"    標準偏差の比較: 元データ {weekend_std/1e6:.1f} -> 移動平均 {smooth_std/1e6:.1f} (百万ウォン)"
          f" — 揺れが小さくなりました")
    return ma7


def step5_growth(weekly: pd.Series) -> None:
    """[5] shift / pct_change: 前週比の成長率を求めます。"""
    print("\n[5] 前週比の成長率 — shift と pct_change")
    # 両端の部分的な週 (7日を満たしていない週) は成長率を歪めるので切り落とします。
    full_weeks = weekly.iloc[1:-1]
    growth = full_weeks.pct_change() * 100
    # shift で同じ計算を自分で再現して検証します。
    manual = (full_weeks - full_weeks.shift(1)) / full_weeks.shift(1) * 100
    assert ((growth - manual).abs().dropna() < 1e-9).all(), "pct_change と shift の計算が違います"
    print("    検証: pct_change == (今週-先週)/先週  (shift で再現、一致)")
    top = growth.nlargest(3)
    bottom = growth.nsmallest(3)
    print("    成長率 上位3週:")
    for ts, val in top.items():
        print(f"      {ts.date()} 締めの週: {val:+.1f}%")
    print("    成長率 下位3週:")
    for ts, val in bottom.items():
        print(f"      {ts.date()} 締めの週: {val:+.1f}%")


def step6_plot(daily: pd.Series, ma7: pd.Series) -> None:
    """[6] 日別売上 + 7日移動平均のグラフを PNG で保存します。"""
    print("\n[6] グラフの保存")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    # 日本語フォントがない環境でも文字化けしないよう、グラフの中の文字は英語を使います。
    ax.plot(daily.index, daily.values / 1e6, color="#9ecae1", linewidth=0.8,
            label="daily revenue")
    ax.plot(ma7.index, ma7.values / 1e6, color="#08519c", linewidth=2.0,
            label="7-day moving average")
    ax.set_title("Cafe chain daily revenue (2025)")
    ax.set_ylabel("revenue (million KRW)")
    ax.legend()
    fig.tight_layout()
    out_path = OUT_DIR / "daily_trend.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"    保存完了: {out_path}")


def main() -> None:
    print("=" * 60)
    print("Level 09 — 時系列データの扱い方")
    print("=" * 60)
    df = load_clean_sales()
    df = step1_diagnose_dates(df)
    daily = step2_daily_series(df)
    weekly = step3_resample(daily)
    ma7 = step4_rolling(daily)
    step5_growth(weekly)
    step6_plot(daily, ma7)
    print("\n完了! ドリンクの繁忙期 (春から上昇するカーブ) が移動平均に表れているか、グラフで確認してみてください。")


if __name__ == "__main__":
    main()
