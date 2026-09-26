# Lecture 03 — データハンドリング (NumPy · Pandas)

> Excel でやっていたすべての表作業をコードに移し、Excel にはできないことまでやり遂げる講義です。

## この講義で学ぶこと

- データがなぜ「行と列の表」に整理されるのか、その表をコードで扱うと何が変わるのか
- NumPy 配列 (ndarray) で数十万個の数値を一度に計算する、ベクトル化の考え方
- Pandas の DataFrame で読み込み → クレンジング → フィルタ → 集計 → 結合 → 時系列 → ピボットまで、実務データ分析のパイプライン全体
- データが大きくなったときにメモリを節約し分割処理する大規模戦略、そしていつ DB/Spark へ移るべきか

すべての実習は、架空のカフェチェーン (渋谷・新宿・大阪・名古屋・福岡の5店舗) の売上データを使います。
データは `common/hjh_data.py` がコードで直接生成するのでインターネット接続は不要で、欠損値や負の値の汚染まで
実務さながらに仕込んであります。なお金額は韓国ウォン (₩, KRW) 建てです。

## 前提となる講義

- **lecture01 — コンピュータと開発環境**、**lecture02 — Python 基礎** (変数、リスト、辞書、繰り返し、関数、ファイルの読み書き)
- Excel で SUM/フィルタ/ピボットテーブルを使った経験があると、たとえ話がより腑に落ちます。

## レベル構成

| レベル | タイトル | 難易度 |
|---|---|---|
| [level00](level00_what_is_data/README.md) | データとは何か — 表の構造 | ⭐ |
| [level01](level01_excel_to_python/README.md) | Excel から Python へ | ⭐ |
| [level02](level02_numpy_basics/README.md) | NumPy 配列の基礎 | ⭐⭐ |
| [level03](level03_pandas_dataframe/README.md) | Pandas — Series と DataFrame | ⭐⭐ |
| [level04](level04_loading_data/README.md) | データの読み込み (CSV · Excel · JSON) | ⭐⭐ |
| [level05](level05_filter_sort_select/README.md) | フィルタリング・ソート・選択 | ⭐⭐ |
| [level06](level06_missing_outliers/README.md) | 欠損値と外れ値の処理 | ⭐⭐⭐ |
| [level07](level07_groupby_aggregation/README.md) | グループ化と集計 (groupby) | ⭐⭐⭐ |
| [level08](level08_merge_join/README.md) | 結合とジョイン (merge · concat) | ⭐⭐⭐ |
| [level09](level09_time_series/README.md) | 時系列データの扱い方 | ⭐⭐⭐⭐ |
| [level10](level10_pivot_reshape_window/README.md) | ピボット・リシェイプ・ウィンドウ演算 | ⭐⭐⭐⭐ |
| [level11](level11_large_data_strategies/README.md) | 大規模データ処理の戦略 | ⭐⭐⭐⭐⭐ |

## 最短ルート (時間がなければこの5つだけ)

1. **level03** — DataFrame の構造を知らなければ何もできません。
2. **level05** — 実務の質問の80%は「選び出して並べ替える」ことです。
3. **level06** — 現実のデータは必ず汚れています。
4. **level07** — 「店舗別の売上は?」のような質問の答えが groupby です。
5. **level08** — 表1枚で終わる分析はありません。結合こそ実戦です。

最短ルートを終えた後、時系列の業務があるなら level09 を、レポート業務が多いなら level10 を追加してください。

## この講義が実務で活きる場面

- **月次実績レポート**: 30店舗の売上 CSV を開き、店舗別・カテゴリ別の合計と前月比の成長率を5分で出します。(level04·07·09)
- **データ検収**: 取引ログに空の値やマイナスの金額が混ざって届いたとき、何件が汚染されているかを診断し、処理方針を決めます。(level06)
- **目標管理**: 売上表と店舗情報表、目標表を結合して、店舗別の目標達成率ランキングを作ります。(level08·10)
- **自動化**: 毎週繰り返していた Excel の手作業をスクリプト1本に置き換え、同じ結果が毎回再現されるようにします。(level01·11)

## 実行方法

各レベルのフォルダで、次のように実行します。図や CSV などの成果物は各レベルの `outputs/` フォルダに生成されます。

```bash
cd lecture03_data_handling/level03_pandas_dataframe
python3 main.py
```
