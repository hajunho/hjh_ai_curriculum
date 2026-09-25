"""
벡터 데이터베이스의 원리: '의미 좌표로 찾는 도서관'을 NumPy 로 직접 구현.
- MiniVectorStore: add / search / save / load + 메타데이터 필터
- 사내 문서 조각을 넣고 의미 검색, 저장 후 다시 불러 같은 결과 확인
- 데이터가 커질 때의 검색 시간을 재고, 근사 검색(ANN) 개념을 소개
Chroma/FAISS 같은 실전 도구가 내부에서 하는 일의 뼈대입니다.
"""

import json
import os
import pathlib
import sys
import time

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import hjh_data
from mock_llm import MockEmbedding, split_sentences

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


class MiniVectorStore:
    """교육용 미니 벡터 저장소.
    실전 벡터 DB(Chroma, FAISS, pgvector...)의 핵심 4기능을 담았습니다:
    add(추가) / search(유사도 검색+필터) / save(영속화) / load(복원)"""

    def __init__(self, embedder: MockEmbedding):
        self.embedder = embedder
        self.vectors = np.zeros((0, embedder.dim))
        self.texts: list[str] = []
        self.metas: list[dict] = []

    def __len__(self) -> int:
        return len(self.texts)

    def add(self, texts: list[str], metas: list[dict]) -> None:
        """문서 조각들을 임베딩해 저장합니다 (색인 = indexing)."""
        new_vecs = self.embedder.embed_batch(texts)
        self.vectors = np.vstack([self.vectors, new_vecs])
        self.texts.extend(texts)
        self.metas.extend(metas)

    def search(self, query: str, k: int = 3, where: dict | None = None):
        """질문을 임베딩해 코사인 유사도 상위 k개를 반환합니다.
        where={"category": "규정"} 처럼 메타데이터 필터를 걸 수 있습니다."""
        qv = self.embedder.embed(query)
        sims = self.vectors @ qv                      # 정규화 벡터: 내적=코사인
        if where:
            mask = np.array([all(m.get(key) == val for key, val in where.items())
                             for m in self.metas])
            sims = np.where(mask, sims, -np.inf)      # 필터 밖 후보는 제외
        order = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.texts[i], self.metas[i])
                for i in order if np.isfinite(sims[i])]

    def save(self, path: str) -> None:
        """벡터는 .npz, 원문·메타데이터는 .json 으로 디스크에 저장합니다."""
        np.savez_compressed(path + ".npz", vectors=self.vectors)
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump({"texts": self.texts, "metas": self.metas},
                      f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str, embedder: MockEmbedding) -> "MiniVectorStore":
        store = cls(embedder)
        store.vectors = np.load(path + ".npz")["vectors"]
        with open(path + ".json", encoding="utf-8") as f:
            payload = json.load(f)
        store.texts = payload["texts"]
        store.metas = payload["metas"]
        return store


def main() -> None:
    np.random.seed(42)

    print("=" * 62)
    print("Level 07 | 벡터 DB — 의미로 찾는 도서관을 직접 만들기")
    print("=" * 62)

    # [1] 문서 조각 준비 (level06 의 문장 단위 청킹)
    chunks, metas = [], []
    for name, doc in hjh_data.SAMPLE_DOCS.items():
        category = "규정" if "규정" in name else "매뉴얼"
        for i, sent in enumerate(split_sentences(doc)):
            chunks.append(sent)
            metas.append({"source": name, "category": category, "chunk_id": i})
    print(f"\n[1] 색인할 조각: {len(chunks)}개 (메타데이터: 출처/분류/번호)")

    # [2] 저장소 구축 (add = 색인)
    embedder = MockEmbedding(dim=512).fit(chunks)
    store = MiniVectorStore(embedder)
    store.add(chunks, metas)
    print(f"\n[2] MiniVectorStore 구축 완료: {len(store)}개 조각, "
          f"벡터 행렬 {store.vectors.shape}")

    # [3] 의미 검색
    query = "수리를 받으려면 무엇이 필요한가요?"
    print(f"\n[3] 검색: \"{query}\"")
    for sim, text, meta in store.search(query, k=3):
        print(f"    {sim:.3f} {text[:36]}... ({meta['source']})")
    print("    -> '수리'라는 단어로 제품매뉴얼의 보증 조항을 정확히 찾았습니다.")

    # [4] 메타데이터 필터: 같은 질문을 '사내 규정' 안에서만 찾기
    print("\n[4] 필터 검색: 같은 질문, where={'category': '규정'}")
    for sim, text, meta in store.search(query, k=2, where={"category": "규정"}):
        print(f"    {sim:.3f} {text[:36]}... ({meta['source']})")
    print("    -> 매뉴얼이 제외되고 규정 문서만 후보가 됩니다. 부서·기간·문서종류로")
    print("       검색 범위를 좁히는 메타데이터 필터는 벡터 DB 의 기본기입니다.")

    # [5] 영속화: 저장 -> 새 프로세스라고 가정하고 -> 복원
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "mini_store")
    store.save(path)
    restored = MiniVectorStore.load(path, embedder)
    q = "연차는 며칠 전에 신청하나요?"
    same = store.search(q, k=1)[0][1] == restored.search(q, k=1)[0][1]
    print(f"\n[5] save/load 검증: {path}.npz/.json 저장 후 복원")
    print(f"    복원 저장소 검색 결과 일치 여부: {same}")
    print("    -> 임베딩은 비싼 작업이라 '한 번 색인, 계속 재사용'이 핵심입니다.")

    # [6] 규모가 커지면? 전수 비교 시간 측정 + 근사 검색 개념
    print("\n[6] 규모 실험: 가짜 벡터를 늘려가며 전수 검색 시간 측정")
    rng = np.random.default_rng(0)
    qv = store.vectors[0]
    for n in (1_000, 20_000, 200_000):
        big = rng.standard_normal((n, embedder.dim))
        big /= np.linalg.norm(big, axis=1, keepdims=True)
        t0 = time.perf_counter()
        np.argsort(-(big @ qv))[:3]
        ms = (time.perf_counter() - t0) * 1000
        print(f"    {n:>7,}개 벡터 전수 비교: {ms:6.1f} ms")
    print("    -> 수십만 개까지는 전수 비교(NumPy)로도 충분히 빠릅니다.")
    print("       수천만 개 규모가 되면 근사 최근접 탐색(ANN: 도서관을 구역으로")
    print("       나눠 유망한 구역만 뒤지는 방식)을 쓰는 Chroma/FAISS/pgvector")
    print("       같은 전용 벡터 DB 를 도입합니다. 사용법은 오늘 API 와 거의 같습니다:")
    print("       collection.add(...) / collection.query(...)  (Chroma 의 경우)")

    print("\n요약: 벡터 DB = 임베딩 행렬 + 원문/메타 + (규모가 크면) 근사 인덱스.")
    print("      오늘 만든 add/search/save/load 가 모든 벡터 DB 의 공통 뼈대입니다.")


if __name__ == "__main__":
    main()
