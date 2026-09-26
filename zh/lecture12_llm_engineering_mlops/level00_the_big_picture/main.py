"""
模拟 LLM 制造全景地图的演示程序。
按顺序走过预训练 -> 退火 -> SFT -> 偏好对齐(DPO) -> 量化 -> 部署
这条 6 道工序的工厂流水线，逐一输出每道工序的输入/产出物，
并用近似公式估算大致成本 (GPU 小时、美元)。
程序不做真实训练，目标是建立"规模直觉"。
"""

import random

# ---------------------------------------------------------------------------
# 假想的目标模型规格 — 假设是 7B (70 亿参数) 级的开源模型。
# ---------------------------------------------------------------------------
PARAMS = 7e9                 # 参数量
PRETRAIN_TOKENS = 2e12       # 预训练 token 数 (2 万亿个)
ANNEAL_TOKENS = 5e10         # 退火 (高质量数据收尾) token 数
SFT_TOKENS = 100_000 * 600   # 指令-回答 10 万条 x 平均 600 token
DPO_TOKENS = 50_000 * 800 * 2  # 偏好对 5 万对 x 800 token x (策略+参考模型 2 倍)
GPU_FLOPS = 990e12 * 0.40    # H200 级 GPU 实效算力 (峰值 990TFLOPs，按 MFU 40% 计)
GPU_PRICE = 3.5              # 云端 GPU 每小时租金 (美元)


def train_cost(n_params: float, n_tokens: float):
    """训练计算量近似公式: FLOPs ~= 6 x 参数量 x token 数"""
    flops = 6.0 * n_params * n_tokens
    gpu_hours = flops / GPU_FLOPS / 3600.0
    return flops, gpu_hours, gpu_hours * GPU_PRICE


def won(dollars: float) -> str:
    """把美元换算成易读的字符串 (按 1 美元 = 1,400 韩元折算成韩元 KRW)"""
    return f"${dollars:,.0f} (约 {dollars * 1400 / 1e8:.1f}亿韩元)" if dollars > 1e5 \
        else f"${dollars:,.0f} (约 {dollars * 1400 / 1e4:,.0f}万韩元)"


def simulate_loss(start: float, end: float, steps: int, rng: random.Random):
    """模拟各阶段 loss 逐步下降的样子 (指数衰减 + 少量噪声)。"""
    losses = []
    for i in range(steps):
        t = i / (steps - 1)
        base = end + (start - end) * (0.03 ** t)   # 指数式下降
        losses.append(base + rng.uniform(-0.02, 0.02))
    return losses


def run_pipeline():
    rng = random.Random(42)  # 固定种子 — 每次运行结果都相同

    # 每个阶段: (名称, 工厂比喻, 输入物, 产出物, (训练 token 或 None), loss 区间)
    stages = [
        ("预训练 (Pretraining)", "把原料熔炼成钢材的炼钢厂",
         "网页文档等原始文本 2 万亿 token", "base 检查点 (只会续写)",
         PRETRAIN_TOKENS, (10.5, 2.1)),
        ("退火 (Annealing)", "让钢材缓慢冷却以提高强度的热处理工序",
         "base + 教科书级高质量数据 500 亿 token", "base-annealed 检查点",
         ANNEAL_TOKENS, (2.1, 1.9)),
        ("指令微调 (SFT)", "按产品手册组装的装配线",
         "base-annealed + 指令-回答示例 10 万条", "sft 检查点 (能听懂指令)",
         SFT_TOKENS, (1.9, 1.2)),
        ("偏好对齐 (DPO)", "用客户评价打磨出厂品质的 QC 质检线",
         "sft + chosen/rejected 偏好对 5 万对", "chat 检查点 (回答质量提升)",
         DPO_TOKENS, (1.2, 1.1)),
        ("量化 (Quantization)", "把成品压缩体积再装箱的包装线",
         "chat 检查点 fp16 14GB", "int4 模型文件约 4GB",
         None, None),
        ("部署 (Serving)", "接单即发货的物流中心",
         "int4 模型 + 推理服务器", "API 端点 (按 token 计费)",
         None, None),
    ]

    print("=" * 66)
    print("  LLM 工厂流水线模拟 — 一个 7B 模型的诞生记")
    print("=" * 66)
    print(f"  目标模型: {PARAMS / 1e9:.0f}B 参数"
          f" / GPU: H200 级 (实效 {GPU_FLOPS / 1e12:.0f} TFLOPs，每小时 ${GPU_PRICE})")

    total_dollars = 0.0
    for idx, (name, analogy, inp, out, tokens, loss_range) in enumerate(stages):
        print(f"\n[{idx + 1}] {name}")
        print(f"    比喻   : {analogy}")
        print(f"    输入   : {inp}")
        print(f"    产出   : {out}")

        if tokens is not None:
            # 训练阶段 — 用近似公式做成本估算。
            flops, hours, dollars = train_cost(PARAMS, tokens)
            total_dollars += dollars
            print(f"    计算量 : {flops:.2e} FLOPs (6 x N x D 近似)")
            if hours > 100:
                print(f"    GPU    : {hours:,.0f} GPU小时"
                      f" = 512 张 H200 约 {hours / 512 / 24:.1f} 天")
            else:
                print(f"    GPU    : {hours:,.1f} GPU小时 — 与预训练相比只是一瞬间")
            print(f"    成本   : {won(dollars)}")
            lo = simulate_loss(*loss_range, steps=5, rng=rng)
            curve = " -> ".join(f"{v:.2f}" for v in lo)
            print(f"    loss   : {curve}")
        elif "量化" in name:
            # 量化阶段 — 不是训练而是转换，几个 GPU 小时就能完成。
            dollars = 2 * GPU_PRICE
            total_dollars += dollars
            print(f"    GPU    : 约 2 GPU小时 (是转换作业，不是训练)")
            print(f"    成本   : ${dollars:.0f} — 内存 14GB -> 4GB (约省 71%)")
        else:
            # 部署阶段 — 不是固定成本，而是随用量增长的费用。
            latency = rng.uniform(0.25, 0.45)
            print(f"    GPU    : 常驻 1 张以上 (随用量伸缩) / 首 token 延迟"
                  f" 约 {latency:.2f}秒")
            print(f"    成本   : 不是制造成本，而是'运营费' — 与请求量成正比")

    print("\n" + "=" * 66)
    print(f"  [7] 制造总成本(部署之前): {won(total_dollars)}")
    share = train_cost(PARAMS, PRETRAIN_TOKENS)[2] / total_dollars * 100
    print(f"      其中预训练占 {share:.1f}% — 所以企业一般不自己造 base 模型，")
    print(f"      而是拿公开模型只做 SFT/DPO，这已是行业标准策略。")
    print("      在这门课里，我们会把这整条流水线做成微缩版亲手跑一遍。")
    print("=" * 66)


if __name__ == "__main__":
    run_pipeline()
