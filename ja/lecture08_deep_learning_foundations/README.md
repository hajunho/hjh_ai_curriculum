# Lecture 08 — ディープラーニング基礎

> 「コンピュータに自分で特徴を見つけさせる」というディープラーニングの核心の発想を、パーセプトロンから誤差逆伝播のフルスクラッチ実装、PyTorch の学習ループ、過学習防止と学習の安定化、GPU・分散学習の概念まで順を追って学びます。
> 数式は最小限に、たとえ話と自分で実行するコードで理解します。

## この講義で学ぶこと

- 線形モデルには解けない問題 (XOR) とは何か、ニューラルネットワークはそれをどう解くのか
- パーセプトロン・活性化関数・損失関数・勾配降下法・誤差逆伝播 — ディープラーニングの部品を numpy だけで自分の手で組み立てる
- PyTorch のテンソル・autograd・オプティマイザ・nn.Module — フルスクラッチ実装がフレームワークで数行に縮む感覚
- Dataset/DataLoader ベースの標準的な学習ループと学習曲線の読み方
- ドロップアウト・weight decay・早期終了で過学習を防ぐ方法
- 学習率スケジュール (warmup・cosine)・勾配クリッピング・バッチ正規化で学習を安定化させる方法
- GPU がなぜ速いのか、混合精度 (fp16/bf16) と分散学習 (DDP/FSDP) とは何か — 8B モデルの学習メモリを自分で見積もる

## 前提となる講義

- **lecture03 — データハンドリング** (NumPy の配列演算が読める必要があります)
- **lecture06 — 機械学習入門** (特に level03 線形回帰、level05 ロジスティック回帰、level09 過学習)

level00〜04 は numpy のみを使い、level05 から PyTorch (CPU) を使います。すべての実習は GPU なしの CPU で 90 秒以内に終わる小型の規模です。

## レベル一覧

| レベル | タイトル | 難易度 |
|---|---|---|
| [level00](level00_why_neural_networks/README.md) | ニューラルネットワークはなぜ登場したのか | ⭐ |
| [level01](level01_perceptron_from_scratch/README.md) | パーセプトロンを自作する | ⭐⭐ |
| [level02](level02_activation_functions/README.md) | 活性化関数 | ⭐⭐ |
| [level03](level03_loss_gradient_descent/README.md) | 損失関数と勾配降下法 | ⭐⭐⭐ |
| [level04](level04_backprop_from_scratch/README.md) | 誤差逆伝播のフルスクラッチ実装 | ⭐⭐⭐⭐ |
| [level05](level05_pytorch_tensors/README.md) | PyTorch の第一歩 — テンソル | ⭐⭐⭐ |
| [level06](level06_autograd_optimizers/README.md) | autograd とオプティマイザ | ⭐⭐⭐ |
| [level07](level07_mlp_classifier/README.md) | MLP 分類モデルを作る | ⭐⭐⭐ |
| [level08](level08_training_loops/README.md) | 学習ループ・バッチ・エポック | ⭐⭐⭐ |
| [level09](level09_regularization_dropout/README.md) | 過学習を防ぐテクニック | ⭐⭐⭐⭐ |
| [level10](level10_lr_schedules_stability/README.md) | 学習率と学習の安定化 | ⭐⭐⭐⭐ |
| [level11](level11_gpu_amp_distributed/README.md) | GPU・混合精度・分散学習の概念 | ⭐⭐⭐⭐⭐ |

## 最短コース (時間がなければこの 5 つだけ)

1. **level00** — ニューラルネットワークがなぜ必要かを知らないと、残りが暗記になってしまいます。
2. **level04** — 誤差逆伝播を一度フルスクラッチで実装すると、ディープラーニングが「魔法」から「機械」になります。
3. **level06** — autograd が level04 の手作業をどう置き換えるのかを対比して見ます。
4. **level08** — 実務コードの骨格である標準学習ループ。以降のすべての講義 (ビジョン・NLP・LLM) で再利用されます。
5. **level09** — 実務で最も頻繁に出会う事故 (過学習) の予防法です。

## この講義が実務で使われる場面

- **モデル性能の会議で**: 「loss 曲線がこの形なら過学習です」を自分で読み、言えるようになります (level08〜09)。
- **外注・協業の検収で**: 開発会社から届いた学習ログから、学習率の問題・発散の兆候を見抜きます (level03, level10)。
- **インフラ予算の稟議で**: 「このモデルの学習に GPU メモリはどれだけ必要か」を根拠のある数字で見積もります (level11)。
- **次の講義への橋**: lecture09 (コンピュータビジョン)、lecture10 (NLP)、lecture12 (LLM) のすべてのモデルが、ここで作る学習ループの上で動きます。
