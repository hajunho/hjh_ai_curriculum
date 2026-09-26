# 環境準備ガイド (SETUP)

このカリキュラムは **GPU のない普通のノート PC**(Mac / Windows)で全課程が動きます。

## 1. Python のインストール確認

ターミナル(Mac: ターミナル.app、Windows: PowerShell)で:

```bash
python3 --version   # 3.10 以上なら OK (Windows は python --version)
```

入っていなければ https://www.python.org/downloads/ からインストールしてください。
詳しい手順は lecture01/level03 で図解のように順を追って説明します。

## 2. リポジトリの取得と仮想環境

日本語版は `ja/` フォルダの中にあります。

```bash
git clone https://github.com/hajunho/hjh_ai_curriculum.git
cd hjh_ai_curriculum/ja

python3 -m venv ../.venv
source ../.venv/bin/activate     # Windows: ..\.venv\Scripts\activate
pip install -r ../requirements.txt
```

プロンプトの先頭に `(.venv)` が付けば成功です。仮想環境が何なのかは
lecture01/level04 で学びます。今は「このプロジェクト専用の道具箱」とだけ
考えておいてください。

## 3. 動作確認

```bash
python3 common/hjh_data.py
```

「すべてのジェネレーターが正常に動作しました。」と表示されれば準備完了です。

## 4. 講義の進め方

すべてのレベルで同じやり方です。

```bash
cd lecture06_machine_learning_basics/level03_linear_regression
cat README.md      # 講義ノートを読む (GitHub 上で読んでも OK)
python3 main.py    # 実習を実行する
```

## 5. よくある質問

- **インターネットは必要ですか?** インストール後は不要です。実習データはコードが直接生成します。
- **GPU は必要ですか?** いいえ。ディープラーニングの講義も小さく設計してあり、CPU で数分以内に終わります。
- **LLM の API キーは必要ですか?** lecture11 の一部レベルはキーがあるとより楽しめますが、
  キーなしでも動くオフラインモードをすべてのレベルに入れてあります。
- **エラーが出ます。** まず各レベルの README 下部にある「よくあるつまずき」を確認してください。
