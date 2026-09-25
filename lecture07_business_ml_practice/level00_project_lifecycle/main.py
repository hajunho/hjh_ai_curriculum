"""
데이터 프로젝트 다섯 단계(문제정의→데이터→모델→배포→모니터링)를
관문(gate) 체크리스트로 통과시키는 시뮬레이션.
가상의 '구독 이탈 방어' 프로젝트를 예로,
각 단계의 산출물이 비어 있으면 관문에서 STOP 판정이 나는 것을 관찰합니다.
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data


def gate(stage_name: str, checklist: dict) -> bool:
    """체크리스트의 값이 하나라도 비어 있으면 관문 통과 실패."""
    print(f"\n  [{stage_name}] 관문 점검")
    ok = True
    for item, value in checklist.items():
        filled = value not in ("", None, [])
        mark = "OK " if filled else "누락"
        shown = value if filled else "(비어 있음)"
        print(f"    - {mark} | {item}: {shown}")
        if not filled:
            ok = False
    print(f"    => 판정: {'PASS - 다음 단계 진행' if ok else 'STOP - 채우기 전에는 진행 금지'}")
    return ok


def main() -> None:
    print("=" * 62)
    print(" 데이터 프로젝트 생애주기 시뮬레이션: '구독 이탈 방어' 프로젝트")
    print("=" * 62)
    results = {}

    # ------------------------------------------------------------------
    print("\n[1] 문제정의 — 무엇을, 왜, 성공 기준은?")
    problem_spec = {
        "예측 대상(라벨 정의)": "다음 달 결제 갱신을 하지 않은 고객 = 이탈(1)",
        "예측 결과의 활용 방안": "매주 월요일 위험 상위 200명에게 상담/쿠폰 제공",
        "목표 지표(숫자)": "6개월 내 월 이탈률 18% -> 15%",
        "기준선(현재 방식)": "전 고객 무작위 100명에게 쿠폰 발송",
    }
    results["1.문제정의"] = gate("문제정의", problem_spec)

    # ------------------------------------------------------------------
    print("\n[2] 데이터 — 그 질문에 답할 데이터가 있는가? (실제 감사 수행)")
    rows = hjh_data.churn_table(n=2000, seed=7)
    n_total = len(rows)
    n_missing = sum(1 for r in rows if any(v is None for v in r.values()))
    n_churn = sum(r["churned"] for r in rows)
    churn_rate = n_churn / n_total
    min_minority = 200  # 소수 클래스(이탈) 최소 표본 관문 기준

    print(f"    행 수: {n_total} / 결측 있는 행: {n_missing}")
    print(f"    이탈 고객: {n_churn}명 (이탈률 {churn_rate:.1%})")
    data_audit = {
        "표본 크기 확인": f"{n_total}행",
        "결측 점검": f"{n_missing}행 (허용 범위)",
        "소수 클래스 표본": f"{n_churn}명" if n_churn >= min_minority else "",
        "라벨 정의 일치 확인": "결제 로그 기준, CS팀과 합의 완료",
    }
    results["2.데이터"] = gate("데이터", data_audit)

    # ------------------------------------------------------------------
    print("\n[3] 모델 — 넘어야 할 기준선(baseline)부터 계산")
    # 아무 모델 없이 '다수 클래스(비이탈)'로만 찍는 기준선의 정확도
    majority_acc = 1 - churn_rate
    print(f"    다수결 기준선 정확도: {majority_acc:.1%}  <- '아무도 이탈 안 함'이라고만 해도 이만큼 나옵니다")
    print("    => 이후 만들 모델은 '정확도'가 아니라 이탈자를 실제로 찾아내는")
    print("       재현율/정밀도로 이 기준선을 이겨야 합니다 (level01, level07에서 계속).")
    model_report = {
        "기준선 성능 기록": f"다수결 정확도 {majority_acc:.1%}",
        "검증 방법 합의": "교차검증 5-fold (level06에서 학습)",
        "모델 성능 보고": "(이후 레벨에서 작성 예정)",  # 데모이므로 채워진 것으로 처리
    }
    results["3.모델"] = gate("모델", model_report)

    # ------------------------------------------------------------------
    print("\n[4] 배포 — 현업이 실제로 쓸 수 있는가?")
    deploy_plan = {
        "결과 전달 채널": "매주 월요일 CRM 시스템에 위험 명단 업로드",
        "받는 사람과 업무 절차": "",  # 일부러 비워 둠: STOP 판정을 관찰하기 위함
        "장애 시 대응(모델 미동작 시)": "",
    }
    results["4.배포"] = gate("배포", deploy_plan)

    # ------------------------------------------------------------------
    print("\n[5] 모니터링 — 성능이 유지되고 있는가?")
    monitor_plan = {
        "성능 추적 지표": "주간 재현율/정밀도, 캠페인 후 실제 이탈률",
        "데이터 분포 이동 감시": "",  # 일부러 비워 둠
        "재훈련 기준": "재현율이 2주 연속 5%p 이상 하락하면 재훈련",
    }
    results["5.모니터링"] = gate("모니터링", monitor_plan)

    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("[6] 최종 요약 — 이 프로젝트는 지금 어디서 멈춰야 하는가")
    print("=" * 62)
    first_stop = None
    for stage, ok in results.items():
        print(f"    {stage:<10} : {'PASS' if ok else 'STOP'}")
        if not ok and first_stop is None:
            first_stop = stage
    if first_stop:
        print(f"\n    => 첫 STOP 지점: {first_stop}")
        print("       모델 성능을 더 올리는 것보다, 이 관문의 빈칸을 채우는 것이 우선입니다.")
    print("\n    교훈: 실패하는 프로젝트는 코드가 아니라 '빈칸을 안고 전진'하다 무너집니다.")


if __name__ == "__main__":
    main()
