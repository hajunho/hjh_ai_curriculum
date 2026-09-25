"""
GPU·혼합정밀도·분산학습의 '왜'를 숫자로 이해하는 개념 실습입니다.
(1) 행렬곱 크기별 CPU 시간 측정 -> 연산량이 세제곱으로 커짐을 확인,
(2) fp32/fp16/bf16 메모리 계산기 -> 정밀도와 메모리의 거래,
(3) 8B 파라미터 모델의 학습 메모리 견적 -> GPU 예산 회의용 숫자 만들기.
모든 계산은 CPU 에서 몇 초 안에 끝납니다.
"""

import time

import numpy as np
import torch

GB = 1024 ** 3


def bench_matmul(sizes, repeats=3):
    """(n,n)@(n,n) 행렬곱 시간 측정. 연산량은 약 2*n^3 FLOP."""
    results = []
    for n in sizes:
        rng = np.random.default_rng(0)                    # 재현성
        A = rng.standard_normal((n, n), dtype=np.float32)
        B = rng.standard_normal((n, n), dtype=np.float32)
        A @ B                                             # 워밍업 (캐시/스레드 준비)
        t0 = time.perf_counter()
        for _ in range(repeats):
            A @ B
        dt = (time.perf_counter() - t0) / repeats
        gflops = 2 * n ** 3 / dt / 1e9
        results.append((n, dt, gflops))
    return results


def bytes_per_element(dtype_name):
    return {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}[dtype_name]


def model_memory_gb(n_params, dtype_name):
    """파라미터 저장 메모리(GB) = 파라미터 수 x 원소당 바이트."""
    return n_params * bytes_per_element(dtype_name) / GB


def training_memory_gb(n_params, mixed=False, act_gb=20.0):
    """학습 메모리 견적(GB). 관례적 근사:
    - 순수 fp32     : 가중치4 + 그래디언트4 + Adam 상태8(m,v 각 fp32)      = 16바이트/파라미터
    - 혼합정밀(AMP) : 가중치2(fp16) + 그래디언트2 + fp32 마스터4 + Adam 8  = 16바이트/파라미터
      (계산은 빨라지지만 Adam 학습 '저장' 메모리는 크게 안 줄어드는 것이 포인트)
    여기에 활성값(중간 계산 결과) 대략치를 더합니다."""
    per_param = 16
    detail = ("가중치 fp16 2 + grad 2 + fp32 마스터 4 + Adam 8" if mixed
              else "가중치 4 + grad 4 + Adam(m,v) 8")
    states = n_params * per_param / GB
    return states, states + act_gb, detail


def main():
    torch.manual_seed(0)

    print("[1] 왜 GPU 인가 — 딥러닝의 계산은 90% 이상이 행렬곱입니다")
    print("    CPU = 박사 몇 명(복잡한 일 잘함), GPU = 산수 요원 수천 명(단순 곱셈을 동시에).")
    print("    행렬곱은 '단순 곱셈 대량 동시 처리'라 GPU 의 압승 종목입니다.\n")

    print("[2] CPU 행렬곱 벤치마크 — (n,n)@(n,n), 연산량 = 2n^3")
    sizes = [128, 256, 512, 1024]
    rows = bench_matmul(sizes)
    print(f"    {'n':>6s} {'시간(ms)':>10s} {'연산량(GFLOP)':>14s} {'속도(GFLOP/s)':>14s}")
    for n, dt, gflops in rows:
        print(f"    {n:6d} {dt*1e3:10.2f} {2*n**3/1e9:14.3f} {gflops:14.1f}")
    n0, t0_, _ = rows[0]
    n3, t3_, _ = rows[-1]
    print(f"    n 이 {n3//n0}배 -> 연산량은 {(n3/n0)**3:.0f}배. 크기가 조금 커져도 계산은 폭증합니다.")
    print("    최신 GPU 는 이 종목에서 수백 TFLOP/s — CPU 의 수백~수천 배입니다.\n")

    print("[3] 정밀도(dtype) 별 메모리 계산기 — '자릿수 적은 계산기'의 거래")
    print(f"    {'파라미터 수':>14s} {'fp32':>10s} {'fp16/bf16':>10s} {'int8':>10s}")
    for label, n_params in [("1억 (0.1B)", int(1e8)), ("10억 (1B)", int(1e9)),
                            ("80억 (8B)", int(8e9)), ("700억 (70B)", int(7e10))]:
        f32 = model_memory_gb(n_params, "fp32")
        f16 = model_memory_gb(n_params, "fp16")
        i8 = model_memory_gb(n_params, "int8")
        print(f"    {label:>13s} {f32:9.1f}G {f16:9.1f}G {i8:9.1f}G")
    print("    fp16: 소수점 정밀도를 절반 내주고 메모리/속도 2배를 얻습니다.")
    print("    bf16: fp16 과 같은 2바이트지만 지수부가 fp32 와 같아 큰 수에 강함 -> LLM 학습 표준.")
    print("    혼합정밀(AMP): 곱셈은 fp16/bf16, 합산·가중치 원본은 fp32 로 -> 속도와 안정성 겸비.\n")

    print("[4] 8B 모델 '학습' 메모리 견적 — 추론과는 차원이 다릅니다")
    n_params = int(8e9)
    act = 20.0                                            # 활성값 대략치(배치·문맥길이에 따라 변동)
    for mixed in (False, True):
        act_now = act / 2 if mixed else act               # AMP: 활성값이 절반 정밀도라 약 1/2
        states, total, detail = training_memory_gb(n_params, mixed, act_now)
        mode = "혼합정밀(AMP)" if mixed else "순수 fp32"
        print(f"    [{mode}] 파라미터당 16바이트 ({detail})")
        print(f"      상태 저장 {states:.0f} GB + 활성값 ~{act_now:.0f} GB = 약 {total:.0f} GB")
    print("    포인트: AMP 도 Adam 상태 탓에 '저장' 메모리는 16바이트/파라미터로 같습니다.")
    print("    이득은 (1) 계산 속도 ~2배 (2) 활성값 절반 (3) 통신량 절반에서 나옵니다.")
    total_gb = training_memory_gb(n_params, True, act / 2)[1]
    print(f"    => 80GB GPU 1장에 들어가나? {'예' if total_gb <= 80 else '아니오'} "
          f"(필요 {total_gb:.0f} GB) — 나눠 들어야 합니다.\n")

    print("[5] 분산학습 = 큰 짐 나눠 들기")
    print("    DDP  (데이터 병렬): 같은 모델을 GPU 마다 복사, 데이터만 나눔.")
    print("          -> 짐(모델)이 1장에 들어갈 때 속도를 올리는 방법. 기울기만 서로 평균.")
    print("    FSDP (ZeRO, 모델 샤딩): 가중치·그래디언트·Adam 상태 자체를 쪼개 나눠 듦.")
    print("          -> 짐이 1장에 안 들어갈 때의 방법. 필요할 때만 조각을 모았다 흩습니다.")
    n_gpus_options = [1, 2, 4, 8, 16]
    states = n_params * 16 / GB
    print(f"\n    8B 모델 FSDP 분산 시 GPU 1장당 부담 (상태 {states:.0f} GB 분할 + 활성값 {act:.0f} GB 는 각자)")
    print(f"    {'GPU 수':>8s} {'장당 상태(GB)':>14s} {'장당 합계(GB)':>14s} {'80GB 1장 OK?':>13s}")
    for g in n_gpus_options:
        share = states / g + act
        print(f"    {g:8d} {states/g:14.1f} {share:14.1f} {'OK' if share <= 80 else 'X':>10s}")

    print("\n[6] 실제 코드는 이렇게 씁니다 (개념 확인용 주석 — 이 실습은 CPU 라 실행 안 함)")
    print("    # AMP : with torch.autocast('cuda', dtype=torch.bfloat16): loss = model(x)")
    print("    #       fp16 사용 시 GradScaler 로 loss 를 키웠다 줄이는 loss scaling 병행")
    print("    # DDP : model = torch.nn.parallel.DistributedDataParallel(model)")
    print("    # FSDP: model = torch.distributed.fsdp.FullyShardedDataParallel(model)")
    print("\n[7] 정리: GPU 예산 질문의 답은 '파라미터 수 x 바이트 수 + 활성값'이라는 곱셈입니다.")
    print("    이 견적 하나로 '왜 8B 학습에 GPU 여러 장이 필요한가'를 회의에서 설명할 수 있습니다.")


if __name__ == "__main__":
    main()
