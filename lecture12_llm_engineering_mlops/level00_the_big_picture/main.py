"""
LLM 제작의 전체 지도를 시뮬레이션하는 데모입니다.
사전학습 -> 어닐링 -> SFT -> 선호정렬(DPO) -> 양자화 -> 배포의
6단계 공장 라인을 순서대로 지나가며, 각 단계의 입력물/산출물과
대략적인 비용(GPU 시간, 달러)을 근사 공식으로 계산해 출력합니다.
실제 학습은 하지 않으며, "규모 감각"을 잡는 것이 목표입니다.
"""

import random

# ---------------------------------------------------------------------------
# 가상의 목표 모델 사양 — 7B(70억 파라미터)급 오픈모델을 가정합니다.
# ---------------------------------------------------------------------------
PARAMS = 7e9                 # 파라미터 수
PRETRAIN_TOKENS = 2e12       # 사전학습 토큰 수 (2조 개)
ANNEAL_TOKENS = 5e10         # 어닐링(고품질 데이터 마무리) 토큰 수
SFT_TOKENS = 100_000 * 600   # 지시-응답 10만 건 x 평균 600토큰
DPO_TOKENS = 50_000 * 800 * 2  # 선호쌍 5만 건 x 800토큰 x (정책+기준 모델 2배)
GPU_FLOPS = 990e12 * 0.40    # H200급 GPU 실효 연산력 (peak 990TFLOPs, MFU 40% 가정)
GPU_PRICE = 3.5              # 클라우드 GPU 1시간 대여료 (달러)


def train_cost(n_params: float, n_tokens: float):
    """학습 연산량 근사 공식: FLOPs ~= 6 x 파라미터 수 x 토큰 수"""
    flops = 6.0 * n_params * n_tokens
    gpu_hours = flops / GPU_FLOPS / 3600.0
    return flops, gpu_hours, gpu_hours * GPU_PRICE


def won(dollars: float) -> str:
    """달러를 읽기 쉬운 문자열로 (환율 1,400원 가정)"""
    return f"${dollars:,.0f} (약 {dollars * 1400 / 1e8:.1f}억 원)" if dollars > 1e5 \
        else f"${dollars:,.0f} (약 {dollars * 1400 / 1e4:,.0f}만 원)"


def simulate_loss(start: float, end: float, steps: int, rng: random.Random):
    """단계별 loss 가 내려가는 모습을 흉내 냅니다 (지수 감소 + 작은 잡음)."""
    losses = []
    for i in range(steps):
        t = i / (steps - 1)
        base = end + (start - end) * (0.03 ** t)   # 지수적으로 감소
        losses.append(base + rng.uniform(-0.02, 0.02))
    return losses


def run_pipeline():
    rng = random.Random(42)  # 시드 고정 — 실행할 때마다 같은 결과

    # 각 단계: (이름, 공장 비유, 입력물, 산출물, (학습 토큰 or None), loss 구간)
    stages = [
        ("사전학습 (Pretraining)", "원재료를 녹여 강철을 만드는 제철소",
         "웹 문서 등 원시 텍스트 2조 토큰", "base 체크포인트 (이어쓰기만 가능)",
         PRETRAIN_TOKENS, (10.5, 2.1)),
        ("어닐링 (Annealing)", "강철을 천천히 식혀 강도를 높이는 열처리",
         "base + 교과서급 고품질 데이터 500억 토큰", "base-annealed 체크포인트",
         ANNEAL_TOKENS, (2.1, 1.9)),
        ("지시학습 (SFT)", "제품 매뉴얼대로 조립하는 조립 라인",
         "base-annealed + 지시-응답 예제 10만 건", "sft 체크포인트 (지시를 따름)",
         SFT_TOKENS, (1.9, 1.2)),
        ("선호정렬 (DPO)", "고객 평가로 마감 품질을 다듬는 QC 라인",
         "sft + chosen/rejected 선호쌍 5만 건", "chat 체크포인트 (답변 품질 향상)",
         DPO_TOKENS, (1.2, 1.1)),
        ("양자화 (Quantization)", "완제품을 부피 줄여 포장하는 포장 라인",
         "chat 체크포인트 fp16 14GB", "int4 모델 파일 약 4GB",
         None, None),
        ("배포 (Serving)", "물류센터에서 주문 즉시 출고하는 배송망",
         "int4 모델 + 추론 서버", "API 엔드포인트 (토큰 단위 과금)",
         None, None),
    ]

    print("=" * 66)
    print("  LLM 공장 라인 시뮬레이션 — 7B 모델이 만들어지기까지")
    print("=" * 66)
    print(f"  목표 모델: {PARAMS / 1e9:.0f}B 파라미터"
          f" / GPU: H200급 (실효 {GPU_FLOPS / 1e12:.0f} TFLOPs, 시간당 ${GPU_PRICE})")

    total_dollars = 0.0
    for idx, (name, analogy, inp, out, tokens, loss_range) in enumerate(stages):
        print(f"\n[{idx + 1}] {name}")
        print(f"    비유   : {analogy}")
        print(f"    입력   : {inp}")
        print(f"    출력   : {out}")

        if tokens is not None:
            # 학습 단계 — 비용을 근사 공식으로 견적냅니다.
            flops, hours, dollars = train_cost(PARAMS, tokens)
            total_dollars += dollars
            print(f"    연산량 : {flops:.2e} FLOPs (6 x N x D 근사)")
            if hours > 100:
                print(f"    GPU    : {hours:,.0f} GPU시간"
                      f" = H200 512장으로 약 {hours / 512 / 24:.1f}일")
            else:
                print(f"    GPU    : {hours:,.1f} GPU시간 — 사전학습에 비하면 순식간")
            print(f"    비용   : {won(dollars)}")
            lo = simulate_loss(*loss_range, steps=5, rng=rng)
            curve = " -> ".join(f"{v:.2f}" for v in lo)
            print(f"    loss   : {curve}")
        elif "양자화" in name:
            # 양자화 단계 — 학습이 아니라 변환이라 GPU 몇 시간이면 끝납니다.
            dollars = 2 * GPU_PRICE
            total_dollars += dollars
            print(f"    GPU    : 약 2 GPU시간 (학습이 아닌 변환 작업)")
            print(f"    비용   : ${dollars:.0f} — 메모리 14GB -> 4GB (약 71% 절감)")
        else:
            # 배포 단계 — 고정비가 아니라 사용량 비례 비용입니다.
            latency = rng.uniform(0.25, 0.45)
            print(f"    GPU    : 상시 1장 이상 (사용량 비례) / 첫 토큰 지연"
                  f" 약 {latency:.2f}초")
            print(f"    비용   : 만드는 비용이 아니라 '운영비' — 요청량에 비례")

    print("\n" + "=" * 66)
    print(f"  [7] 총 제작 비용(배포 전까지): {won(total_dollars)}")
    share = train_cost(PARAMS, PRETRAIN_TOKENS)[2] / total_dollars * 100
    print(f"      이 중 사전학습이 {share:.1f}% — 그래서 기업들은 base 모델을")
    print(f"      직접 만들지 않고, 공개 모델에 SFT/DPO만 얹는 전략을 씁니다.")
    print("      이번 강의에서는 이 라인 전체를 미니어처로 직접 돌려 봅니다.")
    print("=" * 66)


if __name__ == "__main__":
    run_pipeline()
