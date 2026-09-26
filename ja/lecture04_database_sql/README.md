# Lecture 04 — データベースと SQL

> 「データは会社の帳簿です。SQL はその帳簿に話しかける言語です。」

Excel ファイルをメールで回して仕事をしていた人が、会社のデータベース (Database) に
直接質問を投げられる人へと成長するための講義です。実習はすべて Python に内蔵された
SQLite (エスキューライト) を使うので、**何もインストールする必要はありません。**
各レベルのフォルダで `main.py` を実行すると、実習用のショッピングサイトのデータベース
`hjh_shop.db` が自動的に作られ、SQL 文と実行結果が並んで表示されます。

この講義の主役は SQL です。Python は SQL を実行してくれる「実行係」に過ぎません。
コードよりも **SQL 文そのもの** を読んで書き写すことに集中してみてください。

## この講義で学ぶこと

- データベースが Excel の共有より優れている理由 (同時更新・整合性・権限)
- テーブル・主キー・外部キーでデータがどうつながるか
- SELECT / WHERE / ORDER BY / GROUP BY / JOIN / サブクエリ — 実務の質問を SQL に変換する方法
- INSERT / UPDATE / DELETE とトランザクション — データを安全に変更する方法
- インデックスと実行計画 — 遅いクエリを速くする原理
- ウィンドウ関数 — 順位・累計・移動合計のような分析クエリ
- Python + pandas で SQL の結果をレポートまで自動化するミニパイプライン

## 前提となる講義

- **lecture02 (Python 基礎)** — main.py を読んで実行できるレベルで十分です。
- **lecture03 (NumPy · Pandas)** — level11 で pandas を少しだけ使いますが、知らなくても進められます。

## レベル目次

| レベル | タイトル | 難易度 |
|---|---|---|
| [level00](level00_why_databases/README.md) | データベースはなぜ必要なのか | ⭐ |
| [level01](level01_tables_keys/README.md) | テーブル・行・列・主キー | ⭐ |
| [level02](level02_select_basics/README.md) | SELECT の基礎 | ⭐ |
| [level03](level03_where_filtering/README.md) | WHERE — 条件検索 | ⭐⭐ |
| [level04](level04_order_limit_distinct/README.md) | ソート・重複排除・上位 N 件 | ⭐⭐ |
| [level05](level05_aggregate_groupby/README.md) | 集計関数と GROUP BY | ⭐⭐ |
| [level06](level06_joins/README.md) | JOIN — 複数テーブルをつなぐ | ⭐⭐⭐ |
| [level07](level07_subqueries/README.md) | サブクエリ | ⭐⭐⭐ |
| [level08](level08_dml_transactions/README.md) | データの変更とトランザクション | ⭐⭐⭐ |
| [level09](level09_indexes_performance/README.md) | インデックスとクエリ性能 | ⭐⭐⭐⭐ |
| [level10](level10_window_functions/README.md) | ウィンドウ関数と分析クエリ | ⭐⭐⭐⭐ |
| [level11](level11_python_db_pipeline/README.md) | Python 連携とデータパイプライン | ⭐⭐⭐⭐ |

## 最短ルート (時間がない人はこの 5 つだけ)

1. **level02 — SELECT の基礎**: すべての SQL の出発点。
2. **level03 — WHERE**: 「条件に合うものだけ」を選び出す。実務の質問の 8 割はこれです。
3. **level05 — GROUP BY**: 「都市別」「カテゴリ別」— レポートの言語。
4. **level06 — JOIN**: 複数の帳簿をつないで初めて本当の分析になります。
5. **level11 — Python 連携**: SQL の結果を自動レポートに仕上げる総まとめ。

## 実習の進め方

```bash
cd lecture04_database_sql/level02_select_basics
python3 main.py
```

各 main.py は実行するたびに、レベルフォルダの中に `hjh_shop.db` を作り直します。
うっかりデータを壊しても、もう一度実行すれば元どおりになるので、思い切り実験してください。
出力では必ず **SQL 文が先**、その下に結果の表が表示されます。
SQL 文を声に出して読んでみるのが、いちばん良い復習になります。

## この講義が実務で活きる場面

- **マーケティング**: 「前四半期の VIP 顧客のうち、再購入がない人のリストをください」→ WHERE + サブクエリ
- **営業管理**: 「支店別・月別の売上ランキングと累計達成率」→ GROUP BY + ウィンドウ関数
- **在庫/オペレーション**: 「注文が入ったら在庫を減らす。ただし足りなければ全体を取り消す」→ トランザクション
- **データ分析**: 社内 DB から SQL で抽出したデータを pandas で整形し、週次レポートを自動配信
- **開発者との協業**: 「この画面、なぜ遅いんですか?」の代わりに「このクエリにインデックスがないようです」と言えるようになる

Excel しか使えなかった頃は「データを抽出してもらえますか」とお願いする側だったのが、
この講義を終えると **自分で質問して自分で答えを取り出す側** になります。
