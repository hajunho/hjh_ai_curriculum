"""
LLM 制作の全体マップをシミュレーションするデモです。
事前学習 -> アニーリング -> SFT -> 選好アラインメント(DPO) -> 量子化 -> デプロイの
6 段階の工場ラインを順番に通り抜けながら、各段階の入力物/産出物と
おおよそのコスト (GPU 時間、ドル) を近似式で計算して出力します。
実際の学習は行いません。「規模の感覚」をつかむことが目標です。
"""

import random

# ---------------------------------------------------------------------------
# 仮想のターゲットモデル仕様 — 7B(70億パラメータ)級のオープンモデルを想定します。
# ---------------------------------------------------------------------------
PARAMS = 7e9                 # パラメータ数
PRETRAIN_TOKENS = 2e12       # 事前学習トークン数 (2兆個)
ANNEAL_TOKENS = 5e10         # アニーリング(高品質データで仕上げ)のトークン数
SFT_TOKENS = 100_000 * 600   # 指示-応答 10万件 x 平均600トークン
DPO_TOKENS = 50_000 * 800 * 2  # 選好ペア5万件 x 800トークン x (方針+基準モデルで2倍)
GPU_FLOPS = 990e12 * 0.40    # H200級 GPU の実効演算力 (peak 990TFLOPs、MFU 40%想定)
GPU_PRICE = 3.5              # クラウド GPU の1時間レンタル料 (ドル)


def train_cost(n_params: float, n_tokens: float):
    """学習演算量の近似式: FLOPs ~= 6 x パラメータ数 x トークン数"""
    flops = 6.0 * n_params * n_tokens
    gpu_hours = flops / GPU_FLOPS / 3600.0
    return flops, gpu_hours, gpu_hours * GPU_PRICE


def won(dollars: float) -> str:
    """ドルを読みやすい文字列に (為替レートは 1ドル=1,400ウォン想定)"""
    return f"${dollars:,.0f} (約 {dollars * 1400 / 1e8:.1f}億ウォン)" if dollars > 1e5 \
        else f"${dollars:,.0f} (約 {dollars * 1400 / 1e4:,.0f}万ウォン)"


def simulate_loss(start: float, end: float, steps: int, rng: random.Random):
    """段階ごとに loss が下がっていく様子を真似します (指数的減少 + 小さなノイズ)。"""
    losses = []
    for i in range(steps):
        t = i / (steps - 1)
        base = end + (start - end) * (0.03 ** t)   # 指数的に減少
        losses.append(base + rng.uniform(-0.02, 0.02))
    return losses


def run_pipeline():
    rng = random.Random(42)  # シード固定 — 実行するたびに同じ結果

    # 各段階: (名前, 工場のたとえ, 入力物, 産出物, (学習トークン or None), loss 区間)
    stages = [
        ("事前学習 (Pretraining)", "原材料を溶かして鋼を作る製鉄所",
         "ウェブ文書などの生テキスト 2兆トークン", "base チェックポイント (続きを書くだけ)",
         PRETRAIN_TOKENS, (10.5, 2.1)),
        ("アニーリング (Annealing)", "鋼をゆっくり冷やして強度を高める熱処理",
         "base + 教科書級の高品質データ 500億トークン", "base-annealed チェックポイント",
         ANNEAL_TOKENS, (2.1, 1.9)),
        ("指示チューニング (SFT)", "製品マニュアル通りに組み立てる組立ライン",
         "base-annealed + 指示-応答の例 10万件", "sft チェックポイント (指示に従う)",
         SFT_TOKENS, (1.9, 1.2)),
        ("選好アラインメント (DPO)", "顧客の評価で仕上げの品質を整える QC ライン",
         "sft + chosen/rejected の選好ペア 5万件", "chat チェックポイント (回答品質が向上)",
         DPO_TOKENS, (1.2, 1.1)),
        ("量子化 (Quantization)", "完成品を小さくまとめて梱包する包装ライン",
         "chat チェックポイント fp16 14GB", "int4 モデルファイル 約4GB",
         None, None),
        ("デプロイ (Serving)", "物流センターから注文即出荷する配送網",
         "int4 モデル + 推論サーバー", "API エンドポイント (トークン単位の課金)",
         None, None),
    ]

    print("=" * 66)
    print("  LLM 工場ラインのシミュレーション — 7B モデルができあがるまで")
    print("=" * 66)
    print(f"  ターゲットモデル: {PARAMS / 1e9:.0f}B パラメータ"
          f" / GPU: H200級 (実効 {GPU_FLOPS / 1e12:.0f} TFLOPs、1時間あたり${GPU_PRICE})")

    total_dollars = 0.0
    for idx, (name, analogy, inp, out, tokens, loss_range) in enumerate(stages):
        print(f"\n[{idx + 1}] {name}")
        print(f"    たとえ : {analogy}")
        print(f"    入力   : {inp}")
        print(f"    出力   : {out}")

        if tokens is not None:
            # 学習段階 — コストを近似式で見積もります。
            flops, hours, dollars = train_cost(PARAMS, tokens)
            total_dollars += dollars
            print(f"    演算量 : {flops:.2e} FLOPs (6 x N x D 近似)")
            if hours > 100:
                print(f"    GPU    : {hours:,.0f} GPU時間"
                      f" = H200 512枚で約 {hours / 512 / 24:.1f}日")
            else:
                print(f"    GPU    : {hours:,.1f} GPU時間 — 事前学習に比べれば一瞬")
            print(f"    コスト : {won(dollars)}")
            lo = simulate_loss(*loss_range, steps=5, rng=rng)
            curve = " -> ".join(f"{v:.2f}" for v in lo)
            print(f"    loss   : {curve}")
        elif "量子化" in name:
            # 量子化の段階 — 学習ではなく変換なので GPU 数時間で終わります。
            dollars = 2 * GPU_PRICE
            total_dollars += dollars
            print(f"    GPU    : 約 2 GPU時間 (学習ではなく変換作業)")
            print(f"    コスト : ${dollars:.0f} — メモリ 14GB -> 4GB (約71%削減)")
        else:
            # デプロイ段階 — 固定費ではなく使用量に比例するコストです。
            latency = rng.uniform(0.25, 0.45)
            print(f"    GPU    : 常時1枚以上 (使用量に比例) / 最初のトークンまでの遅延"
                  f" 約 {latency:.2f}秒")
            print(f"    コスト : 作るコストではなく「運用費」 — リクエスト量に比例")

    print("\n" + "=" * 66)
    print(f"  [7] 総制作コスト(デプロイ前まで): {won(total_dollars)}")
    share = train_cost(PARAMS, PRETRAIN_TOKENS)[2] / total_dollars * 100
    print(f"      このうち事前学習が {share:.1f}% — だから企業は base モデルを")
    print(f"      自前で作らず、公開モデルに SFT/DPO だけ載せる戦略をとります。")
    print("      この講義では、このライン全体をミニチュアで自分で回してみます。")
    print("=" * 66)


if __name__ == "__main__":
    run_pipeline()
