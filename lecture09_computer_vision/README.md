# Lecture 09 — 컴퓨터 비전

> 컴퓨터가 이미지를 "숫자 격자"로 읽는 순간부터, CNN 으로 도형을 분류하고,
> 객체 탐지·세그멘테이션·비전 트랜스포머까지 — 눈이 하는 일을 코드로 재현하는 여정입니다.

## 이 강의에서 배우는 것

- 이미지가 컴퓨터 안에서 어떻게 숫자로 표현되는지 (픽셀·채널·해상도)
- 합성곱(Convolution)이 왜 이미지에 특화된 연산인지 — 직접 구현해서 확인
- CNN(합성곱 신경망)의 구조를 층별로 설계하고 파라미터 수를 계산하는 법
- 도형 이미지 분류를 학습→평가→오류 분석까지 완주하는 전체 과정
- 데이터 증강·전이학습으로 "데이터가 부족할 때" 성능을 끌어올리는 기술
- 객체 탐지(바운딩 박스·IoU), 세그멘테이션, OCR 이 산업 현장에서 쓰이는 원리
- 비전 트랜스포머(ViT)와 CLIP 식 이미지-텍스트 정렬 — 최신 비전 AI 의 문법

모든 실습은 인터넷 다운로드 없이, 저장소 안에서 numpy 로 직접 그린 도형 이미지
(`common/hjh_data.py` 의 `shape_images`)만으로 진행합니다. 유명 데이터셋 없이도
CNN 의 원리는 전부 증명할 수 있습니다.

## 선행 강의

- **lecture03 — 데이터 다루기** (NumPy 배열 인덱싱·슬라이싱을 읽을 수 있어야 합니다)
- **lecture06 — 머신러닝 입문** (학습/평가 분리, 정확도·오차의 개념)
- **lecture08 — 딥러닝 기초** (PyTorch 텐서, 학습 루프, MLP — level03 부터 필요)

level00~02 는 numpy 만 사용하므로 lecture08 을 건너뛰고 먼저 맛봐도 됩니다.

## 레벨 목차

| 레벨 | 제목 | 난이도 |
|---|---|---|
| [level00](level00_how_computers_see/README.md) | 컴퓨터가 이미지를 보는 방식 | ⭐ |
| [level01](level01_pixels_channels/README.md) | 픽셀·채널·이미지 연산 | ⭐⭐ |
| [level02](level02_convolution_explained/README.md) | 합성곱(Convolution) 이해 | ⭐⭐⭐ |
| [level03](level03_building_cnn/README.md) | CNN 구조 만들기 | ⭐⭐⭐ |
| [level04](level04_pooling_features/README.md) | 풀링과 특징의 계층 | ⭐⭐⭐ |
| [level05](level05_shape_classification/README.md) | 도형 이미지 분류 실습 | ⭐⭐⭐ |
| [level06](level06_data_augmentation/README.md) | 데이터 증강 | ⭐⭐⭐ |
| [level07](level07_transfer_learning/README.md) | 전이학습 | ⭐⭐⭐⭐ |
| [level08](level08_image_pipeline/README.md) | 이미지 분류 실전 파이프라인 | ⭐⭐⭐⭐ |
| [level09](level09_object_detection/README.md) | 객체 탐지의 원리 | ⭐⭐⭐⭐ |
| [level10](level10_segmentation_ocr_industry/README.md) | 세그멘테이션·OCR·산업 검사 | ⭐⭐⭐⭐ |
| [level11](level11_vit_multimodal/README.md) | 비전 트랜스포머와 멀티모달 | ⭐⭐⭐⭐⭐ |

## 빠른 경로 (시간이 없다면 이 5개만)

1. **level00** — 이미지 = 숫자 격자. 이 한 문장을 몸으로 이해하지 못하면 이후 전부가 마법처럼 보입니다
2. **level02** — 합성곱을 numpy 로 직접 구현: CNN 의 심장을 열어 봅니다
3. **level05** — 도형 분류 완주: 학습→평가→오분류 분석의 전체 사이클
4. **level07** — 전이학습: 실무에서 이미지 과제의 9할이 이 방식으로 풀립니다
5. **level09** — 객체 탐지: "무엇이 어디에"까지 답하는 원리

## 이 강의가 실무에서 쓰이는 장면

- **제조 품질 검사**: 생산 라인 카메라로 불량(흠집·얼룩·누락)을 자동 검출 (level05, 08, 10)
- **문서 자동화**: 계약서·영수증 스캔본에서 글자 영역을 찾아 텍스트로 변환하는 OCR 파이프라인 (level09, 10)
- **리테일·물류**: 매대 사진에서 상품 위치·결품 확인, 창고 내 물품 카운팅 (level09)
- **의료·안전**: 판독 보조, CCTV 이상 상황 감지 — "탐지 결과를 어디까지 믿을 것인가"를 판단하는 기준 (level08, 09)
- **외주·솔루션 검수**: "정확도 99%"라는 비전 솔루션 제안서를 받았을 때, 데이터 분할·오류 분석·증강 여부를 따져 묻는 능력 (level06, 08)
- **최신 기술 이해**: "우리 서비스에 멀티모달 AI 를 붙이자"는 논의에서 ViT·CLIP 이 무엇인지 알고 대화하기 (level11)

## 진행 팁

- 모든 레벨은 `main.py`를 직접 실행하면서 읽도록 설계되어 있습니다. 특히 이 강의는
  PNG 그림이 핵심 산출물이니, 각 레벨의 `outputs/` 폴더에 생성된 이미지를 꼭 열어 보세요.
- 실행: 저장소 루트의 가상환경으로 `python3 main.py` (환경 설정은 루트 `SETUP.md` 참고)
- torch 학습이 들어가는 레벨(level03~09, 11)도 전부 CPU 에서 수십 초 안에 끝나도록 소형으로 설계했습니다.
- 데이터는 전부 코드가 그 자리에서 그려 내므로 인터넷 연결이 필요 없습니다.
