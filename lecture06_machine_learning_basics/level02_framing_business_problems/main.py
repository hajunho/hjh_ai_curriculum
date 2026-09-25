"""
level02 — 비즈니스 질문 -> ML 문제 명세서 체크리스트 엔진

"이탈을 줄이고 싶다" 같은 소망을 5요소 명세서(대상/단위/시점/지표/액션)로
적으면, 빈칸·누설 위험·ML 적합성을 자동 점검해 주는 도구를 만듭니다.
모델을 만들지 않는 유일한 레벨이지만, 실무에서는 가장 자주 쓰는 기술입니다.
"""

from dataclasses import dataclass, field


@dataclass
class Feature:
    """예측 재료(특징) 하나. available_at: 이 값을 알 수 있는 시점."""
    name: str
    available_at: str  # "예측 시점 이전" 또는 "결과 확정 이후"


@dataclass
class MLSpec:
    """ML 문제 명세서 — 회의실 화이트보드에 쓰는 5요소 + 기준선."""
    business_goal: str = ""    # 비즈니스 목표 (소망)
    target: str = ""           # 1. 예측 대상: 측정 가능한 정의
    unit: str = ""             # 2. 예측 단위: 한 행이 무엇인가
    timing: str = ""           # 3. 예측 시점
    metric: str = ""           # 4. 평가지표
    action: str = ""           # 5. 예측 후 액션
    baseline: str = ""         # 모델 없이 지금 방식의 성적
    features: list[Feature] = field(default_factory=list)


def check_completeness(spec: MLSpec) -> list[str]:
    """5요소 + 기준선이 채워졌는지 점검."""
    problems = []
    checks = [
        (spec.target, "예측 대상(target)이 비었습니다. '이탈'이 아니라 '다음 30일 내 해지(1/0)'처럼 측정 가능하게."),
        (spec.unit, "예측 단위(unit)가 비었습니다. 한 행이 고객인지 계정인지 주문인지 정하세요."),
        (spec.timing, "예측 시점(timing)이 비었습니다. 언제 예측 버튼을 누르는지 정하세요."),
        (spec.metric, "평가지표(metric)가 비었습니다. 비즈니스 손익과 연결된 지표를 고르세요."),
        (spec.action, "액션(action)이 비었습니다. 액션 없는 예측은 장식품입니다."),
        (spec.baseline, "기준선(baseline)이 비었습니다. 모델 없이 지금 방식의 성적을 먼저 재세요."),
    ]
    for value, msg in checks:
        if not value.strip():
            problems.append(msg)
    return problems


def check_leakage(spec: MLSpec) -> list[str]:
    """누설 점검: '예측 버튼을 누르는 순간 이 값을 알 수 있는가?'"""
    return [
        f"누설 의심: '{f.name}' 은(는) {f.available_at}에 생기는 값입니다. "
        f"예측 시점({spec.timing})에는 존재하지 않으므로 재료에서 빼야 합니다."
        for f in spec.features if f.available_at != "예측 시점 이전"
    ]


def validate(title: str, spec: MLSpec) -> None:
    """명세서 하나를 점검하고 결과를 출력."""
    print(f"  ■ 명세서: {title}")
    print(f"    목표: {spec.business_goal}")
    issues = check_completeness(spec) + check_leakage(spec)
    if not issues:
        print(f"    [통과] 5요소 완비 — 대상: {spec.target} / 단위: {spec.unit}")
        print(f"           시점: {spec.timing} / 지표: {spec.metric}")
        print(f"           액션: {spec.action}")
    else:
        for i, msg in enumerate(issues, 1):
            print(f"    [지적 {i}] {msg}")
    print(f"    점수: {6 + len(spec.features) - len(issues)} / {6 + len(spec.features)}\n")


def judge_ml_fitness() -> None:
    """[4] 될 문제 / 안 될 문제 자동 분류.
    기준: 반복 발생, 충분한 정답 데이터, 패턴 존재 가능성, 오차 허용."""
    candidates = [
        ("매장별 내일 샌드위치 수요 예측", dict(repeats=True, labels=3000, pattern=True, error_ok=True)),
        ("부가세 10% 자동 계산", dict(repeats=True, labels=100000, pattern=True, error_ok=False)),
        ("우리 회사 M&A 성공 여부 예측", dict(repeats=False, labels=12, pattern=True, error_ok=True)),
        ("카드 이상거래 실시간 탐지", dict(repeats=True, labels=50000, pattern=True, error_ok=True)),
    ]
    for name, c in candidates:
        reasons = []
        if not c["repeats"]:
            reasons.append("반복 발생하는 일이 아님 (사례가 안 쌓임)")
        if c["labels"] < 500:
            reasons.append(f"정답 데이터가 {c['labels']}건뿐 (학습 불가 수준)")
        if not c["error_ok"]:
            reasons.append("오차 허용 0 -> 이미 명확한 규칙이 있다면 규칙으로 (level00)")
        verdict = "ML 로 풀 만함" if not reasons else "ML 부적합"
        print(f"    {name:28s} -> {verdict}")
        for r in reasons:
            print(f"        이유: {r}")


if __name__ == "__main__":
    # [1] 나쁜 명세서: 소망만 있고 전부 빈칸 --------------------------------
    print("[1] 나쁜 명세서 — '이탈을 줄이고 싶다'를 그대로 제출한 경우")
    validate("소망 그대로", MLSpec(business_goal="이탈을 줄이고 싶다"))

    # [2] 좋은 명세서: 5요소를 채운 번역 완료본 -----------------------------
    print("[2] 좋은 명세서 — 같은 소망을 ML 문제로 번역한 경우")
    good = MLSpec(
        business_goal="구독 이탈을 줄이고 싶다",
        target="이번 달 말 기준, 다음 30일 안에 구독을 해지함 (1/0)",
        unit="활성 구독 고객 1명 (매월 1일 스냅샷)",
        timing="매월 1일 오전, 전월 말까지 확정된 데이터만 사용",
        metric="위험 상위 10% 명단의 정밀도/재현율 (정확도 아님, level06 참고)",
        action="위험 상위 10% 고객에게 CS팀이 1주 내 리텐션 상담 + 혜택 제안",
        baseline="현재는 '3개월 미접속자 전원 문자' — 반응률 2%",
        features=[
            Feature("최근 30일 이용일수", "예측 시점 이전"),
            Feature("최근 30일 고객센터 문의 수", "예측 시점 이전"),
            Feature("가입 개월 수", "예측 시점 이전"),
        ],
    )
    validate("이탈 예측 v1", good)

    # [3] 누설 함정: 결과 확정 이후에 생기는 열이 재료에 섞인 경우 -----------
    print("[3] 누설 점검 — 미래를 훔쳐보는 재료가 섞인 명세서")
    leaky = MLSpec(
        business_goal="구독 이탈을 줄이고 싶다",
        target=good.target, unit=good.unit, timing=good.timing,
        metric=good.metric, action=good.action, baseline=good.baseline,
        features=[
            Feature("최근 30일 이용일수", "예측 시점 이전"),
            Feature("해지 위약금 청구액", "결과 확정 이후"),   # <- 해지해야 생기는 값!
            Feature("해지 사유 설문 응답", "결과 확정 이후"),
        ],
    )
    validate("이탈 예측 v2 (함정)", leaky)
    print("    -> 누설 열은 시험 성적만 완벽하게 만들고 실전에서는 쓸 수 없습니다.\n")

    # [4] 될 문제 / 안 될 문제 분류 -----------------------------------------
    print("[4] 후보 문제 4개의 ML 적합성 판정")
    judge_ml_fitness()
    print()
    print("[5] 요약: 번역 순서 = 소망 -> (대상/단위/시점/지표/액션) -> 누설 점검 -> 기준선")
    print("    이 체크리스트를 통과한 뒤에야 모델링(level03~)을 시작할 가치가 있습니다.")
