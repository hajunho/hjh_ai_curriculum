# hjh AI Curriculum — 非エンジニアのビジネスパーソンのための AI 完走コース(日本語版)

> **Excel しか使ったことのない人が、機械学習・ディープラーニング・LLM エンジニアリングまで。**
> IT を専攻していないビジネスパーソンが「AI を理解し、自分の手で扱える人」に
> なれるよう設計した、12講義・144レベルの実習中心カリキュラムです。

- **作成**: ハ・ジュノ (hajunho) · ZeliDesk / エデュテック学科
- **本フォルダについて**: これは日本語版 (Japanese edition) です。韓国語のオリジナルは
  リポジトリのルートにあります。データ生成器 (`ja/common/hjh_data.py`) も日本語に
  ローカライズ済みで、この `ja/` フォルダだけで完結して学習できます。
- **ライセンス**: MIT — すべての講義ノートとコードは、このカリキュラムのために新しく
  書き下ろした純粋な創作物です。外部の教材やデータセットを複製しておらず、
  実習データもすべてコードで直接生成します。
- **言語**: 講義は日本語、コードの識別子は英語

---

## このカリキュラムが他と違う点

1. **たとえ話が先、数式は後** — すべての概念を、仕事や日常のたとえ話で先に説明します。
2. **インストール地獄なし** — 実習データはコードが自分で作り出すのでダウンロード不要。
   GPU のない普通のノート PC で全課程が動きます。
3. **ビジネスの問題で学ぶ** — 売上予測、顧客の解約(チャーン)、不正取引の検知、
   社内文書チャットボットなど、会社で実際に出会う問題で実習します。
4. **ゼロからの実装込み** — パーセプトロン、誤差逆伝播、トークナイザー、ミニ GPT まで
   自分の手で作り、「ブラックボックス」の中を開けて見ます。

## 全体構成

12の講義 (lecture01〜12)、各講義は12のレベル (level00〜11) で構成されます。
level00 は完全入門(たとえ話と概念)、level11 は実務レベルの発展編です。

| 講義 | テーマ | 一行紹介 |
|---|---|---|
| [lecture01](lecture01_computer_and_environment/) | コンピュータと開発環境 | ターミナル、Python のインストール、Git、クラウドまで |
| [lecture02](lecture02_python_basics/) | Python プログラミング基礎 | 変数からオブジェクト指向・テストまで |
| [lecture03](lecture03_data_handling/) | データハンドリング | Excel からの卒業 — NumPy · Pandas |
| [lecture04](lecture04_database_sql/) | データベースと SQL | SELECT からウィンドウ関数・Python 連携まで |
| [lecture05](lecture05_statistics_visualization/) | 統計と可視化 | 平均の落とし穴から A/B テストまで |
| [lecture06](lecture06_machine_learning_basics/) | 機械学習入門 | 回帰・分類・アンサンブル・モデル解釈 |
| [lecture07](lecture07_business_ml_practice/) | ビジネス ML 実践 | 売上予測・顧客チャーン・不正取引検知 |
| [lecture08](lecture08_deep_learning_foundations/) | ディープラーニング基礎 | パーセプトロンのフルスクラッチから PyTorch まで |
| [lecture09](lecture09_computer_vision/) | コンピュータビジョン | CNN・転移学習・産業応用 |
| [lecture10](lecture10_nlp_text/) | 自然言語処理 | 日本語の前処理からトランスフォーマー実装まで |
| [lecture11](lecture11_llm_and_rag/) | LLM 活用と RAG | プロンプト、埋め込み、社内文書チャットボット |
| [lecture12](lecture12_llm_engineering_mlops/) | LLM エンジニアリングと MLOps | トークナイザー・ミニ GPT・アラインメント・量子化・サービング |

詳しい目次は [CURRICULUM.md](CURRICULUM.md)、環境の準備は [SETUP.md](SETUP.md)、
用語につまずいたら [GLOSSARY.md](GLOSSARY.md) を見てください。

## 学習の進め方

1. `SETUP.md` のとおりに Python 環境を準備します (30分)。
2. 各レベルのフォルダの `README.md` を読み → `main.py` を実行し → 課題を解きます。
3. 1レベルは 30〜90分の分量です。週5レベルなら **約7か月** で完走できます。
4. 順番どおりが王道ですが、各講義の README に「最短ルート」を案内してあります。

```bash
git clone https://github.com/hajunho/hjh_ai_curriculum.git
cd hjh_ai_curriculum/ja
python3 -m venv ../.venv && source ../.venv/bin/activate
pip install -r ../requirements.txt
python3 lecture01_computer_and_environment/level00_what_computers_do/main.py
```

## 著作権とライセンス

このリポジトリのすべての文書・コード・データ生成器は、ハ・ジュノ (hajunho) がこの
カリキュラムのために新しく書いたものです。特定の商用講座や書籍の内容を写しておらず、
外部データセットのファイルも含みません。MIT ライセンスのもと、誰でも自由に
使用・改変・再配布できます。講義などでお使いの際は、出典を残していただけると幸いです。
