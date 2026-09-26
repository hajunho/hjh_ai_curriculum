# Lecture 05 — 統計とデータ可視化

数字を「要約」し、要約が隠しているものを「グラフ」であぶり出し、最後には
「この差は本物か、それとも偶然か」を判定する講義です。Excel で AVERAGE を
押したことがあるなら、すでに半分は始まっています。残りの半分 — その平均を
いつ信じてはいけないのか — をここで学びます。

## 何を学ぶか

- 代表値 (平均・中央値・分散) がデータを要約する仕組みと、その落とし穴
- 分布・ヒストグラム・Matplotlib でデータをグラフにする方法
- 人をだますグラフと誠実なグラフを見分ける目
- 相関と因果、確率、正規分布と中心極限定理
- 標本から全体を推定する方法: 信頼区間、仮説検定、p 値
- 実務の花形: A/B テストの設計と解釈、ベイズ的思考による不確実性の報告

## 前提となる講義

- **lecture02 — Python プログラミング基礎** (必須)
- **lecture03 — データハンドリング (NumPy · Pandas)** (必須: 配列とデータフレームを使います)
- **lecture04** まで終えていれば、さらにスムーズです。

## レベル目次

| レベル | タイトル | 難易度 |
|---|---|---|
| [level00](level00_summarizing_reality/README.md) | 数字で現実を要約するということ | ⭐ |
| [level01](level01_mean_median_variance/README.md) | 平均・中央値・分散 | ⭐ |
| [level02](level02_distributions_histograms/README.md) | 分布とヒストグラム | ⭐⭐ |
| [level03](level03_matplotlib_basics/README.md) | Matplotlib の基本グラフ | ⭐⭐ |
| [level04](level04_good_vs_bad_charts/README.md) | 良いグラフ vs 悪いグラフ | ⭐⭐ |
| [level05](level05_correlation_causation/README.md) | 相関関係と因果関係 | ⭐⭐⭐ |
| [level06](level06_probability_basics/README.md) | 確率の基礎 | ⭐⭐⭐ |
| [level07](level07_normal_clt/README.md) | 正規分布と中心極限定理 | ⭐⭐⭐ |
| [level08](level08_sampling_confidence/README.md) | 標本と信頼区間 | ⭐⭐⭐ |
| [level09](level09_hypothesis_pvalue/README.md) | 仮説検定と p 値 | ⭐⭐⭐⭐ |
| [level10](level10_ab_testing/README.md) | A/B テストの設計と解釈 | ⭐⭐⭐⭐ |
| [level11](level11_bayesian_thinking/README.md) | ベイズ的思考と不確実性の伝え方 | ⭐⭐⭐⭐⭐ |

## 最短コース (時間がなければこの 5 つだけ)

1. **level01** — 平均・中央値・分散: あらゆる数字の報告の基礎
2. **level04** — 良いグラフ vs 悪いグラフ: だまされない、だまさない
3. **level05** — 相関と因果: 会議室で最も頻繁に間違えられるポイント
4. **level09** — 仮説検定と p 値: 「この差は本物ですか?」に答える方法
5. **level10** — A/B テスト: データで意思決定する組織の標準ツール

## この講義が実務で活きる場面

- **月次報告**: 「平均売上 12% 増」という一文の裏に極端な値が 1 件
  隠れていないか、中央値と分布で確かめます (level01〜02)。
- **経営層向け資料**: 軸を切り詰めた棒グラフで成果を水増しするミスを
  避け、他人が作った歪んだグラフを見抜きます (level03〜04)。
- **マーケティング効果分析**: 「広告費を増やしたら売上が伸びた」が因果なのか、
  繁忙期という交絡因子のせいなのかを検討します (level05)。
- **新商品・UI 刷新の判断**: A/B テストを設計し、途中でこっそり結果を
  のぞき見すること (peeking) がなぜ危険なのかを理解します (level09〜10)。
- **不確実性の報告**: 「できる/できない」の二択ではなく「B 案が優れている確率 92%」の
  ように、経営層が判断できる言葉で伝えます (level08、11)。

## 実行方法

各レベルのフォルダで、リポジトリの仮想環境を使って実行します。

```bash
cd lecture05_statistics_visualization/level00_summarizing_reality
python3 main.py
```

グラフを描くレベルでは、実行後にそのフォルダの `outputs/` に PNG ファイルが
生成され、パスが画面に表示されます。データはすべて `common/hjh_data.py` または
コード内で生成するため、インターネット接続は不要です。
