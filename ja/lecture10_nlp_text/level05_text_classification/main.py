"""
level05 — テキスト分類の実習: レビューの感情分析

review_corpus (日本語レビュー 600 件) を TF-IDF でベクトル化し、
ロジスティック回帰でポジティブ/ネガティブを分類します。
学習した係数 (トークンごとの採点表) を開いてモデルの判断根拠を解釈し、
新しいレビューに対する判定根拠レポートまで出力します。
"""

import re
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


def tokenize(text: str) -> list[str]:
    """日本語の連続文字列を文字 2-gram に (level03-04 と同じ方式)。"""
    tokens = []
    for run in re.findall(r"[ぁ-んァ-ヶー一-龯々]+|[a-z0-9]+", text.lower()):
        if re.match(r"[a-z0-9]", run) or len(run) < 2:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


def load_data():
    """[1] レビュー 600 件をロードして学習/評価に分割。"""
    rows = hjh_data.review_corpus(600, seed=3)
    texts = [r["text"] for r in rows]
    labels = np.array([r["label"] for r in rows])
    print(f"[1] データ準備: レビュー {len(texts)}件 "
          f"(ポジティブ {labels.sum()}件 / ネガティブ {(labels == 0).sum()}件)")
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels)
    print(f"    学習 {len(X_tr)}件 / 評価 {len(X_te)}件 (stratify で比率を維持)")
    print(f"    例 (ポジティブ): {X_tr[int(np.argmax(y_tr))]!r}")
    print(f"    例 (ネガティブ): {X_tr[int(np.argmin(y_tr))]!r}\n")
    return X_tr, X_te, y_tr, y_te


def vectorize(X_tr, X_te):
    """[2] TF-IDF ベクトル化 — 語彙辞書と IDF は「学習データだけ」で作る。"""
    vec = TfidfVectorizer(analyzer=tokenize)
    V_tr = vec.fit_transform(X_tr)      # 学習: fit + transform
    V_te = vec.transform(X_te)          # 評価: transform のみ (データリーク防止)
    print(f"[2] ベクトル化: 語彙 {len(vec.get_feature_names_out())}種、"
          f"学習行列 {V_tr.shape}、評価行列 {V_te.shape}")
    print("    注意: 評価データには fit していません (リーク防止)。\n")
    return vec, V_tr, V_te


def train_and_eval(V_tr, y_tr, V_te, y_te):
    """[3] ロジスティック回帰の学習と評価。"""
    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(V_tr, y_tr)
    pred = model.predict(V_te)
    acc = accuracy_score(y_te, pred)
    cm = confusion_matrix(y_te, pred)
    print(f"[3] 学習完了。評価精度: {acc:.1%}")
    print("    混同行列 (行=実際、列=予測 / 0=ネガ、1=ポジ)")
    print(f"          ネガ予測 ポジ予測")
    print(f"    実際ネガ {cm[0, 0]:5d} {cm[0, 1]:6d}")
    print(f"    実際ポジ {cm[1, 0]:5d} {cm[1, 1]:6d}\n")
    return model


def show_scorecard(model, vec, top: int = 8):
    """[4] 係数 = モデルの「採点基準表」。ポジ/ネガの証拠トークンを公開。"""
    words = vec.get_feature_names_out()
    coef = model.coef_[0]
    order = np.argsort(coef)
    print(f"[4] モデルが学んだ採点基準表 (ロジスティック回帰の係数)")
    print("    ポジティブの証拠 top8       | ネガティブの証拠 top8")
    print("    " + "-" * 55)
    for pos_i, neg_i in zip(order[::-1][:top], order[:top]):
        print(f"    {words[pos_i]:10s} {coef[pos_i]:+6.2f}      | "
              f"{words[neg_i]:10s} {coef[neg_i]:+6.2f}")
    print("    -> 人間はルールを 1 行も書いていないのに、データが基準表を作りました。\n")


def judge_new_reviews(model, vec):
    """[5] 新しいレビューの判定 + 根拠トークンのレポート。"""
    new_reviews = [
        "スタッフが親切で梱包も丁寧なので満足しています",
        "配送に一週間もかかって、箱も破れたまま届きました",
        "性能が期待以上なのでリピート購入するつもりです",
        "説明と違ってがっかりしましたし、問い合わせの返事もありません",
    ]
    words = vec.get_feature_names_out()
    coef = model.coef_[0]
    V = vec.transform(new_reviews)
    proba = model.predict_proba(V)[:, 1]
    print("[5] 新しいレビューの判定レポート (ポジティブ確率 + 判定根拠トークン)")
    for text, p, row in zip(new_reviews, proba, V.toarray()):
        present = np.where(row > 0)[0]                       # このレビューに登場したトークン
        contrib = row[present] * coef[present]               # トークンごとの寄与度
        order = np.argsort(np.abs(contrib))[::-1][:3]
        evidence = ", ".join(
            f"{words[present[i]]}({contrib[i]:+.2f})" for i in order)
        verdict = "ポジティブ" if p >= 0.5 else "ネガティブ"
        print(f"    [{verdict} {p:.0%}] {text}")
        print(f"        根拠: {evidence}")
    print("\n結論: 「何 % 当たる」を超えて「なぜそう判断したか」まで報告できます。")


if __name__ == "__main__":
    print("=" * 70)
    print("テキスト分類 — TF-IDF + ロジスティック回帰の感情分析器")
    print("=" * 70 + "\n")
    np.random.seed(SEED)
    X_tr, X_te, y_tr, y_te = load_data()
    vec, V_tr, V_te = vectorize(X_tr, X_te)
    model = train_and_eval(V_tr, y_tr, V_te, y_te)
    show_scorecard(model, vec)
    judge_new_reviews(model, vec)
