"""
A demo that simulates the full map of LLM production.
We walk through the 6-stage factory line — pre-training -> annealing -> SFT ->
preference alignment (DPO) -> quantization -> serving — printing each stage's
inputs/outputs and an approximate cost (GPU hours, dollars) computed with a
back-of-the-envelope formula. No actual training happens; the goal is to
develop a sense of scale.
"""

import random

# ---------------------------------------------------------------------------
# Specs of our hypothetical target model — a 7B-class open model.
# ---------------------------------------------------------------------------
PARAMS = 7e9                 # number of parameters
PRETRAIN_TOKENS = 2e12       # pre-training tokens (2 trillion)
ANNEAL_TOKENS = 5e10         # annealing tokens (high-quality data finish)
SFT_TOKENS = 100_000 * 600   # 100k instruction-response pairs x avg 600 tokens
DPO_TOKENS = 50_000 * 800 * 2  # 50k preference pairs x 800 tokens x 2 (policy + reference model)
GPU_FLOPS = 990e12 * 0.40    # effective compute of an H200-class GPU (peak 990 TFLOPs, 40% MFU)
GPU_PRICE = 3.5              # cloud GPU rental per hour (dollars)


def train_cost(n_params: float, n_tokens: float):
    """Approximate training compute: FLOPs ~= 6 x parameters x tokens"""
    flops = 6.0 * n_params * n_tokens
    gpu_hours = flops / GPU_FLOPS / 3600.0
    return flops, gpu_hours, gpu_hours * GPU_PRICE


def won(dollars: float) -> str:
    """Format dollars as a readable string (Korean won shown at a rate of KRW 1,400)."""
    return f"${dollars:,.0f} (about ₩{dollars * 1400 / 1e6:,.0f} million)" if dollars > 1e5 \
        else f"${dollars:,.0f} (about ₩{dollars * 1400:,.0f})"


def simulate_loss(start: float, end: float, steps: int, rng: random.Random):
    """Mimic how the loss falls during each stage (exponential decay + small noise)."""
    losses = []
    for i in range(steps):
        t = i / (steps - 1)
        base = end + (start - end) * (0.03 ** t)   # exponential decay
        losses.append(base + rng.uniform(-0.02, 0.02))
    return losses


def run_pipeline():
    rng = random.Random(42)  # fixed seed — same result on every run

    # Each stage: (name, factory analogy, input, output, (training tokens or None), loss range)
    stages = [
        ("Pre-training", "the steel mill that melts raw ore into steel",
         "2 trillion tokens of raw text (web documents etc.)", "base checkpoint (can only continue text)",
         PRETRAIN_TOKENS, (10.5, 2.1)),
        ("Annealing", "the heat-treatment step that cools steel slowly to harden it",
         "base + 50 billion tokens of textbook-grade data", "base-annealed checkpoint",
         ANNEAL_TOKENS, (2.1, 1.9)),
        ("Instruction tuning (SFT)", "the assembly line that builds to the product manual",
         "base-annealed + 100k instruction-response examples", "sft checkpoint (follows instructions)",
         SFT_TOKENS, (1.9, 1.2)),
        ("Preference alignment (DPO)", "the QC line that polishes finish quality using customer ratings",
         "sft + 50k chosen/rejected preference pairs", "chat checkpoint (better answer quality)",
         DPO_TOKENS, (1.2, 1.1)),
        ("Quantization", "the packing line that shrinks the finished product for shipping",
         "chat checkpoint, fp16, 14GB", "int4 model file, about 4GB",
         None, None),
        ("Serving", "the distribution center that ships the moment an order arrives",
         "int4 model + inference server", "API endpoint (billed per token)",
         None, None),
    ]

    print("=" * 66)
    print("  LLM factory-line simulation — the making of a 7B model")
    print("=" * 66)
    print(f"  Target model: {PARAMS / 1e9:.0f}B parameters"
          f" / GPU: H200-class (effective {GPU_FLOPS / 1e12:.0f} TFLOPs, ${GPU_PRICE}/hour)")

    total_dollars = 0.0
    for idx, (name, analogy, inp, out, tokens, loss_range) in enumerate(stages):
        print(f"\n[{idx + 1}] {name}")
        print(f"    Analogy: {analogy}")
        print(f"    Input  : {inp}")
        print(f"    Output : {out}")

        if tokens is not None:
            # Training stage — estimate the cost with the approximation formula.
            flops, hours, dollars = train_cost(PARAMS, tokens)
            total_dollars += dollars
            print(f"    Compute: {flops:.2e} FLOPs (6 x N x D approximation)")
            if hours > 100:
                print(f"    GPU    : {hours:,.0f} GPU-hours"
                      f" = about {hours / 512 / 24:.1f} days on 512 H200s")
            else:
                print(f"    GPU    : {hours:,.1f} GPU-hours — an instant compared to pre-training")
            print(f"    Cost   : {won(dollars)}")
            lo = simulate_loss(*loss_range, steps=5, rng=rng)
            curve = " -> ".join(f"{v:.2f}" for v in lo)
            print(f"    loss   : {curve}")
        elif "Quantization" in name:
            # Quantization stage — a conversion, not training; a few GPU hours suffice.
            dollars = 2 * GPU_PRICE
            total_dollars += dollars
            print(f"    GPU    : about 2 GPU-hours (a conversion job, not training)")
            print(f"    Cost   : ${dollars:.0f} — memory 14GB -> 4GB (about a 71% reduction)")
        else:
            # Serving stage — cost scales with usage rather than being fixed.
            latency = rng.uniform(0.25, 0.45)
            print(f"    GPU    : at least 1 running at all times (scales with usage) / time to first token"
                  f" about {latency:.2f}s")
            print(f"    Cost   : not a build cost but an 'operating cost' — proportional to request volume")

    print("\n" + "=" * 66)
    print(f"  [7] Total build cost (up to deployment): {won(total_dollars)}")
    share = train_cost(PARAMS, PRETRAIN_TOKENS)[2] / total_dollars * 100
    print(f"      Pre-training accounts for {share:.1f}% of it — which is why companies")
    print(f"      don't build base models themselves, but layer SFT/DPO on open models.")
    print("      In this lecture we run this entire line ourselves, in miniature.")
    print("=" * 66)


if __name__ == "__main__":
    run_pipeline()
