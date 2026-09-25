"""
사전학습 데이터 파이프라인을 미니어처로 구현합니다.
문서들을 <s>...</s> 로 이어붙여(패킹) 하나의 긴 토큰 스트림을 만들고,
uint16 바이너리 샤드로 저장한 뒤 numpy memmap 으로 즉시 로딩하여
고정 길이(block_size) 학습 배치를 만들어 내는 전 과정을 실행합니다.
실제 대형 학습에서 수 TB 데이터를 다루는 방식의 축소판입니다.
"""

import os
import sys
import pathlib
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"

# 특수 토큰 — id 를 앞번호로 고정해 둡니다.
BOS, EOS, PAD, UNK = "<s>", "</s>", "<pad>", "<unk>"
SPECIALS = [BOS, EOS, PAD, UNK]


def make_documents(n_docs: int = 200, sents_per_doc: int = 5):
    """tiny_corpus 를 문장으로 쪼갠 뒤 5문장씩 묶어 '문서'를 만듭니다."""
    corpus = hjh_data.tiny_corpus()
    sents = [s.strip() + "." for s in corpus.split(".") if s.strip()]
    docs = []
    for i in range(n_docs):
        chunk = sents[i * sents_per_doc:(i + 1) * sents_per_doc]
        docs.append(" ".join(chunk))
    return docs


def build_vocab(docs):
    """글자 단위 vocab: 특수 토큰 4개 + 코퍼스의 모든 글자."""
    chars = sorted(set("".join(docs)))
    itos = SPECIALS + chars
    stoi = {ch: i for i, ch in enumerate(itos)}
    return stoi, itos


def tokenize_and_pack(docs, stoi):
    """각 문서를 <s> 토큰들 </s> 로 감싸 하나의 긴 스트림으로 잇습니다."""
    stream = []
    for doc in docs:
        stream.append(stoi[BOS])
        stream.extend(stoi.get(ch, stoi[UNK]) for ch in doc)
        stream.append(stoi[EOS])
    return np.array(stream, dtype=np.uint16)  # vocab < 65536 이므로 2바이트면 충분


def get_batch(mm, block_size: int, batch_size: int, rng):
    """memmap 스트림에서 무작위 위치를 골라 (x, y) 배치를 만듭니다.
    y 는 x 를 한 칸 민 것 — '다음 토큰 맞히기' 정답지입니다."""
    starts = rng.integers(0, len(mm) - block_size - 1, size=batch_size)
    x = np.stack([np.asarray(mm[s:s + block_size]) for s in starts])
    y = np.stack([np.asarray(mm[s + 1:s + block_size + 1]) for s in starts])
    return x, y


if __name__ == "__main__":
    rng = np.random.default_rng(42)  # 시드 고정

    # [1] 원시 텍스트 -> 문서 목록
    docs = make_documents()
    total_chars = sum(len(d) for d in docs)
    print(f"[1] 문서 준비: {len(docs)}개 문서, 총 {total_chars:,}자")
    print(f"    예시 문서: \"{docs[0][:40]}...\"")

    # [2] vocab 구축 (글자 단위 — level01 의 BPE 를 쓰면 더 짧아집니다)
    stoi, itos = build_vocab(docs)
    print(f"\n[2] vocab 구축: {len(itos)}개"
          f" (특수 토큰 {len(SPECIALS)} + 글자 {len(itos) - len(SPECIALS)})")
    print(f"    특수 토큰 id: " +
          ", ".join(f"{t}={stoi[t]}" for t in SPECIALS))

    # [3] 토큰화 + <s>...</s> 패킹
    stream = tokenize_and_pack(docs, stoi)
    print(f"\n[3] 패킹 완료: 문서 {len(docs)}개 -> 토큰 스트림 {len(stream):,}개")
    print("    문서를 그냥 자르면 짧은 문서마다 <pad> 낭비가 생기므로,")
    print("    전부 이어붙인 뒤 고정 길이로 자르는 쪽이 GPU 를 꽉 채웁니다.")

    # [4] uint16 바이너리 샤드 저장
    os.makedirs(OUT_DIR, exist_ok=True)
    shard_path = OUT_DIR / "shard_000.bin"
    stream.tofile(shard_path)
    disk = os.path.getsize(shard_path)
    text_bytes = sum(len(d.encode("utf-8")) for d in docs)
    print(f"\n[4] 샤드 저장: {shard_path}")
    print(f"    크기 {disk:,}바이트 (토큰당 2바이트)"
          f" / 원문 UTF-8 {text_bytes:,}바이트")
    print("    실제 파이프라인은 이런 샤드를 수천 개 만들어 병렬로 읽습니다.")

    # [5] memmap 로딩 — 파일 전체를 RAM 에 올리지 않고 즉시 사용
    mm = np.memmap(shard_path, dtype=np.uint16, mode="r")
    print(f"\n[5] memmap 로딩: {len(mm):,}토큰 — 필요한 조각만 그때그때 읽음")
    print("    수백 GB 샤드도 로딩이 0초에 가까운 이유입니다.")

    # [6] 고정 길이 배치 생성
    block_size, batch_size = 64, 8
    x, y = get_batch(mm, block_size, batch_size, rng)
    print(f"\n[6] 배치 생성: x{x.shape}, y{y.shape}"
          f" (block_size={block_size}, batch_size={batch_size})")
    print(f"    x[0][:8] = {x[0][:8].tolist()}")
    print(f"    y[0][:8] = {y[0][:8].tolist()}  <- x 를 한 칸 민 정답지")

    # [7] 디코딩으로 검증 — 문서 경계 토큰이 배치 안에 섞여 들어옵니다.
    def decode(ids):
        return "".join(itos[i] for i in ids)

    row = decode(x[0])
    print(f"\n[7] 배치 첫 줄 복원: \"{row[:60]}\"")
    boundary = [i for i in range(batch_size)
                if stoi[BOS] in x[i] or stoi[EOS] in x[i]]
    print(f"    문서 경계(<s>/</s>)가 포함된 행: {len(boundary)}/{batch_size}개")
    print("    경계를 넘어 이어진 조각도 그대로 학습합니다 — <s> 가 '여기서")
    print("    새 문서 시작'을 알려 주므로 모델이 알아서 문맥을 끊어 읽습니다.")
