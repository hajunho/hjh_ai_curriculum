# Lecture 08 — 딥러닝 기초

> "컴퓨터가 스스로 특징을 찾아내게 한다"는 딥러닝의 핵심 발상을, 퍼셉트론부터 역전파 밑바닥 구현, PyTorch 학습 루프, 과적합 방지와 학습 안정화, GPU·분산학습 개념까지 순서대로 배웁니다.
> 수식은 최소로, 비유와 직접 실행하는 코드로 이해합니다.

## 이 강의에서 배우는 것

- 선형 모델이 못 푸는 문제(XOR)가 무엇이고, 신경망이 그것을 어떻게 푸는지
- 퍼셉트론·활성화 함수·손실 함수·경사하강법·역전파 — 딥러닝의 부품을 numpy만으로 직접 조립
- PyTorch 텐서·autograd·옵티마이저·nn.Module — 밑바닥 구현이 프레임워크에서 몇 줄로 줄어드는지 체감
- Dataset/DataLoader 기반 표준 학습 루프와 학습곡선 읽는 법
- 드롭아웃·weight decay·조기종료로 과적합을 막는 방법
- 학습률 스케줄(warmup·cosine)·그래디언트 클리핑·배치정규화로 학습을 안정화하는 방법
- GPU가 왜 빠른지, 혼합정밀도(fp16/bf16)와 분산학습(DDP/FSDP)이 무엇인지 — 8B 모델 학습 메모리를 직접 견적

## 선행 강의

- **lecture03 — 데이터 다루기** (NumPy 배열 연산을 읽을 수 있어야 합니다)
- **lecture06 — 머신러닝 입문** (특히 level03 선형회귀, level05 로지스틱 회귀, level09 과적합)

level00~04는 numpy만 사용하고, level05부터 PyTorch(CPU)를 사용합니다. 모든 실습은 노트북 없이 CPU에서 90초 안에 끝나는 소형 규모입니다.

## 레벨 목차

| 레벨 | 제목 | 난이도 |
|---|---|---|
| [level00](level00_why_neural_networks/README.md) | 신경망이 왜 등장했나 | ⭐ |
| [level01](level01_perceptron_from_scratch/README.md) | 퍼셉트론 직접 만들기 | ⭐⭐ |
| [level02](level02_activation_functions/README.md) | 활성화 함수 | ⭐⭐ |
| [level03](level03_loss_gradient_descent/README.md) | 손실 함수와 경사하강법 | ⭐⭐⭐ |
| [level04](level04_backprop_from_scratch/README.md) | 역전파 밑바닥 구현 | ⭐⭐⭐⭐ |
| [level05](level05_pytorch_tensors/README.md) | PyTorch 첫걸음 — 텐서 | ⭐⭐⭐ |
| [level06](level06_autograd_optimizers/README.md) | autograd 와 옵티마이저 | ⭐⭐⭐ |
| [level07](level07_mlp_classifier/README.md) | MLP 분류 모델 만들기 | ⭐⭐⭐ |
| [level08](level08_training_loops/README.md) | 학습 루프·배치·에폭 | ⭐⭐⭐ |
| [level09](level09_regularization_dropout/README.md) | 과적합 방지 기법 | ⭐⭐⭐⭐ |
| [level10](level10_lr_schedules_stability/README.md) | 학습률과 학습 안정화 | ⭐⭐⭐⭐ |
| [level11](level11_gpu_amp_distributed/README.md) | GPU·혼합정밀도·분산학습 개념 | ⭐⭐⭐⭐⭐ |

## 빠른 경로 (시간이 없다면 이 5개만)

1. **level00** — 신경망이 왜 필요한지 모르면 나머지가 암기가 됩니다.
2. **level04** — 역전파를 한 번 밑바닥으로 구현해 보면 딥러닝이 "마법"에서 "기계"가 됩니다.
3. **level06** — autograd 가 level04 의 수작업을 어떻게 대체하는지 대비해서 봅니다.
4. **level08** — 실무 코드의 뼈대인 표준 학습 루프. 이후 모든 강의(비전·NLP·LLM)에서 재사용됩니다.
5. **level09** — 실무에서 가장 자주 만나는 사고(과적합)의 예방법입니다.

## 이 강의가 실무에서 쓰이는 장면

- **모델 성능 회의에서**: "loss 곡선이 이렇게 생겼으면 과적합입니다"를 직접 읽고 말할 수 있습니다 (level08~09).
- **외주·협업 검수에서**: 개발사가 보낸 학습 로그에서 학습률 문제·발산 징후를 알아챕니다 (level03, level10).
- **인프라 예산 품의에서**: "이 모델을 학습하려면 GPU 메모리가 얼마나 필요한가"를 근거 숫자로 견적합니다 (level11).
- **다음 강의로 가는 다리**: lecture09(컴퓨터 비전), lecture10(NLP), lecture12(LLM)의 모든 모델이 여기서 만든 학습 루프 위에서 돌아갑니다.
