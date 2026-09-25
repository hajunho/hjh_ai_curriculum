"""
level00 — 규칙 vs 학습

같은 분류 문제(카드 이상거래 탐지)를 두 방식으로 풀어 비교합니다.
  A. 수제 규칙: 사람이 감으로 정한 IF 문 ("50만 원 이상 + 새벽이면 사기")
  B. 학습: 임계값 후보를 전부 시험해 데이터가 최적값을 고르는 탐색 (직접 구현)
핵심 메시지: 전통 프로그래밍은 "규칙+데이터->답", 머신러닝은 "데이터+답->규칙".
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

NIGHT_HOURS = {0, 1, 2, 3, 4, 23}          # 새벽/심야 시간대


def score(y_true: list[int], y_pred: list[int]) -> dict:
    """규칙의 성적표. 사기가 1.4%뿐이라 단순 정확도는 착시를 일으키므로
    '사기 적중률(재현율)'과 '정상 적중률'의 평균(균형 점수)을 기준으로 삼는다."""
    n_fraud = sum(y_true)
    hit_fraud = sum(p == 1 for t, p in zip(y_true, y_pred) if t == 1) / n_fraud
    hit_normal = sum(p == 0 for t, p in zip(y_true, y_pred) if t == 0) / (len(y_true) - n_fraud)
    n_alarm = sum(y_pred)
    precision = (sum(t == 1 for t, p in zip(y_true, y_pred) if p == 1) / n_alarm) if n_alarm else 0.0
    return {"balanced": (hit_fraud + hit_normal) / 2, "recall": hit_fraud,
            "precision": precision, "alarms": n_alarm}


def hand_rule(row: dict) -> int:
    """[수제 규칙] 회의실에서 나올 법한 감(感) 기반 규칙.
    '50만 원 넘는 큰돈이 새벽에 빠져나가면 사기겠지.'"""
    return 1 if (row["amount"] >= 500_000 and row["hour"] in NIGHT_HOURS) else 0


def learn_threshold(rows: list[dict], feature: str) -> tuple[float, float]:
    """[학습] '값 >= T 이면 사기' 규칙의 임계값 T 를 전부 시험한다.
    사람은 규칙의 '모양'만 정하고, '숫자' T 는 데이터가 고른다."""
    y_true = [r["is_fraud"] for r in rows]
    best_t, best_s = None, -1.0
    for t in sorted({r[feature] for r in rows}):
        y_pred = [1 if r[feature] >= t else 0 for r in rows]
        s = score(y_true, y_pred)["balanced"]
        if s > best_s:
            best_t, best_s = t, s
    return best_t, best_s


def learn_combo(rows: list[dict]) -> tuple[float, float]:
    """[학습 확장] '금액 >= T 그리고 새벽 시간대'의 T 를 탐색.
    사람 규칙과 모양은 같지만 숫자는 데이터가 고른다."""
    y_true = [r["is_fraud"] for r in rows]
    best_t, best_s = None, -1.0
    for t in sorted({round(r["amount"], -4) for r in rows}):    # 만원 단위 후보
        y_pred = [1 if (r["amount"] >= t and r["hour"] in NIGHT_HOURS) else 0
                  for r in rows]
        s = score(y_true, y_pred)["balanced"]
        if s > best_s:
            best_t, best_s = t, s
    return best_t, best_s


def report(name: str, s: dict) -> None:
    print(f"    {name}")
    print(f"      균형 점수 {s['balanced']:.1%} / 사기 적중률 {s['recall']:.1%} / "
          f"경보 정밀도 {s['precision']:.1%} (경보 {s['alarms']}건)")


if __name__ == "__main__":
    # [1] 데이터 준비 -----------------------------------------------------
    rows = hjh_data.fraud_table(n=5000, seed=11)   # seed 고정 -> 항상 같은 데이터
    y_true = [r["is_fraud"] for r in rows]
    print("[1] 데이터: 카드 거래", len(rows), "건 (스팸필터와 같은 구조의 분류 문제)")
    hjh_data.head(rows, 3)
    print(f"    사기 비율: {sum(y_true) / len(y_true):.1%}  (사기=1, 정상=0)\n")

    # [2] 방식 A — 수제 규칙 ---------------------------------------------
    pred_hand = [hand_rule(r) for r in rows]
    s_hand = score(y_true, pred_hand)
    print("[2] 방식 A — 수제 규칙 (사람이 숫자를 감으로 결정)")
    print("    규칙: 금액 >= 500,000원 그리고 새벽 시간대이면 사기")
    report("성적:", s_hand)
    print("      -> 잡은 건 전부 진짜지만, 사기의 70%를 놓칩니다. 기준이 너무 높았던 것.\n")

    # [3] 방식 B — 학습: 최적 임계값을 데이터가 고른다 --------------------
    print("[3] 방식 B — 학습 (임계값 T 후보를 전부 시험해 데이터가 선택)")
    best_feat, best_t, best_s = None, None, -1.0
    for feat in ["amount", "hour", "is_foreign"]:
        t, s = learn_threshold(rows, feat)
        print(f"    특징 {feat:12s}: 최적 T={t:>10,} -> 균형 점수 {s:.1%}")
        if s > best_s:
            best_feat, best_t, best_s = feat, t, s
    pred_learn = [1 if r[best_feat] >= best_t else 0 for r in rows]
    s_learn = score(y_true, pred_learn)
    print(f"    => 데이터가 고른 규칙: \"{best_feat} >= {best_t:,} 이면 사기\"")
    report("성적:", s_learn)
    print("      -> 사람은 몰랐던 숫자를 데이터가 찾았지만, 헛경보(정밀도↓)가 많습니다.\n")

    # [4] 방식 B 확장 — 특징 2개 결합 -------------------------------------
    t2, s2 = learn_combo(rows)
    pred_two = [1 if (r["amount"] >= t2 and r["hour"] in NIGHT_HOURS) else 0 for r in rows]
    s_combo = score(y_true, pred_two)
    print("[4] 방식 B 확장 — 사람 규칙과 같은 모양, 숫자만 데이터가 결정")
    print(f"    데이터가 고른 규칙: \"금액 >= {t2:,}원 그리고 새벽이면 사기\"")
    report("성적:", s_combo)
    print("      -> 사람의 감(50만 원)과 데이터의 답(약 2만 원)이 이렇게 다릅니다.\n")

    # [5] 종합 비교 -------------------------------------------------------
    print("[5] 종합 비교 (균형 점수 = 사기 적중률과 정상 적중률의 평균)")
    print("    방식                          균형점수   사기적중률   경보정밀도")
    print(f"    A. 수제 규칙                   {s_hand['balanced']:6.1%}     {s_hand['recall']:6.1%}      {s_hand['precision']:6.1%}")
    print(f"    B. 학습(특징 1개)              {s_learn['balanced']:6.1%}     {s_learn['recall']:6.1%}      {s_learn['precision']:6.1%}")
    print(f"    B. 학습(특징 2개 결합)         {s_combo['balanced']:6.1%}     {s_combo['recall']:6.1%}      {s_combo['precision']:6.1%}")
    print()
    print("    핵심: 전통 프로그래밍  규칙 + 데이터 -> 답")
    print("          머신러닝        데이터 + 답  -> 규칙")
    print("    오늘의 '학습'은 for 문 임계값 탐색이었지만, 신경망도 본질은 같습니다.")
    print("    (주의: 지금은 훈련 데이터로 채점했으므로 성적이 낙관적입니다 -> level04)")
