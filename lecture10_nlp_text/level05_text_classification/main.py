"""
level05 — 텍스트 분류 실습: 리뷰 감성 분석

review_corpus(한국어 리뷰 600건)를 TF-IDF 로 벡터화하고
로지스틱 회귀로 긍정/부정을 분류합니다.
학습된 계수(단어별 점수표)를 열어 모델의 판단 근거를 해석하고,
새 리뷰에 대한 판정 근거 리포트까지 출력합니다.
"""

import sys
import pathlib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

SEED = 42


def load_data():
    """[1] 리뷰 600건 로드 후 학습/평가 분리."""
    rows = hjh_data.review_corpus(600, seed=3)
    texts = [r["text"] for r in rows]
    labels = np.array([r["label"] for r in rows])
    print(f"[1] 데이터 준비: 리뷰 {len(texts)}건 "
          f"(긍정 {labels.sum()}건 / 부정 {(labels == 0).sum()}건)")
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels)
    print(f"    학습 {len(X_tr)}건 / 평가 {len(X_te)}건 (stratify 로 비율 유지)")
    print(f"    예시(긍정): {X_tr[int(np.argmax(y_tr))]!r}")
    print(f"    예시(부정): {X_tr[int(np.argmin(y_tr))]!r}\n")
    return X_tr, X_te, y_tr, y_te


def vectorize(X_tr, X_te):
    """[2] TF-IDF 벡터화 — 어휘 사전과 IDF 는 '학습 데이터로만' 만든다."""
    vec = TfidfVectorizer()
    V_tr = vec.fit_transform(X_tr)      # 학습: fit + transform
    V_te = vec.transform(X_te)          # 평가: transform 만 (데이터 누수 방지)
    print(f"[2] 벡터화: 어휘 {len(vec.get_feature_names_out())}종, "
          f"학습 행렬 {V_tr.shape}, 평가 행렬 {V_te.shape}")
    print("    주의: 평가 데이터에는 fit 을 하지 않았습니다 (누수 방지).\n")
    return vec, V_tr, V_te


def train_and_eval(V_tr, y_tr, V_te, y_te):
    """[3] 로지스틱 회귀 학습과 평가."""
    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(V_tr, y_tr)
    pred = model.predict(V_te)
    acc = accuracy_score(y_te, pred)
    cm = confusion_matrix(y_te, pred)
    print(f"[3] 학습 완료. 평가 정확도: {acc:.1%}")
    print("    혼동 행렬 (행=실제, 열=예측 / 0=부정, 1=긍정)")
    print(f"        부정예측 긍정예측")
    print(f"    실제부정 {cm[0, 0]:5d} {cm[0, 1]:6d}")
    print(f"    실제긍정 {cm[1, 0]:5d} {cm[1, 1]:6d}\n")
    return model


def show_scorecard(model, vec, top: int = 8):
    """[4] 계수 = 모델의 '채점 기준표'. 긍정/부정 증거 단어 공개."""
    words = vec.get_feature_names_out()
    coef = model.coef_[0]
    order = np.argsort(coef)
    print(f"[4] 모델이 배운 채점 기준표 (로지스틱 회귀 계수)")
    print("    긍정 증거 top8              | 부정 증거 top8")
    print("    " + "-" * 55)
    for pos_i, neg_i in zip(order[::-1][:top], order[:top]):
        print(f"    {words[pos_i]:12s} {coef[pos_i]:+6.2f}      | "
              f"{words[neg_i]:12s} {coef[neg_i]:+6.2f}")
    print("    -> 사람이 규칙을 적지 않았는데, 데이터가 기준표를 만들었습니다.\n")


def judge_new_reviews(model, vec):
    """[5] 새 리뷰 판정 + 근거 단어 리포트."""
    new_reviews = [
        "직원분이 친절하고 포장이 꼼꼼해서 만족합니다",
        "배송이 일주일이나 걸렸고 상자도 찢어진 채로 왔어요",
        "성능이 기대 이상이라 재구매 의사 있습니다",
        "설명과 달라 실망했고 문의 답도 없네요",
    ]
    words = vec.get_feature_names_out()
    coef = model.coef_[0]
    V = vec.transform(new_reviews)
    proba = model.predict_proba(V)[:, 1]
    print("[5] 새 리뷰 판정 리포트 (긍정 확률 + 판정 근거 단어)")
    for text, p, row in zip(new_reviews, proba, V.toarray()):
        present = np.where(row > 0)[0]                       # 이 리뷰에 등장한 단어
        contrib = row[present] * coef[present]               # 단어별 기여도
        order = np.argsort(np.abs(contrib))[::-1][:3]
        evidence = ", ".join(
            f"{words[present[i]]}({contrib[i]:+.2f})" for i in order)
        verdict = "긍정" if p >= 0.5 else "부정"
        print(f"    [{verdict} {p:.0%}] {text}")
        print(f"        근거: {evidence}")
    print("\n결론: '몇 % 맞춘다'를 넘어 '왜 그렇게 판단했나'까지 보고할 수 있습니다.")


if __name__ == "__main__":
    print("=" * 70)
    print("텍스트 분류 — TF-IDF + 로지스틱 회귀 감성 분석기")
    print("=" * 70 + "\n")
    np.random.seed(SEED)
    X_tr, X_te, y_tr, y_te = load_data()
    vec, V_tr, V_te = vectorize(X_tr, X_te)
    model = train_and_eval(V_tr, y_tr, V_te, y_te)
    show_scorecard(model, vec)
    judge_new_reviews(model, vec)
