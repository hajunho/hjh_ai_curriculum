"""
모델 서빙(serving) 실습.
tiny_corpus 로 만든 문자 단위 마르코프 미니 모델을 http.server 로 감싸
추론 API 서버를 띄우고, 클라이언트로 여러 번 호출해 지연시간(p50/p95)과
처리량을 측정한 뒤 서버를 깨끗이 종료합니다.
순차 호출 vs 동시 호출 비교로 스레드 서버의 처리량 차이도 확인합니다.
"""

import json
import pathlib
import random
import statistics
import sys
import threading
import time
import urllib.request
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

random.seed(42)
ORDER = 3  # 앞 3글자를 보고 다음 글자를 고르는 마르코프 모델


def build_model(text: str) -> dict:
    """문자 3-gram 마르코프 표: '앞 3글자' -> [다음 글자 후보들]."""
    table = defaultdict(list)
    for i in range(len(text) - ORDER):
        table[text[i:i + ORDER]].append(text[i + ORDER])
    return table


def generate(model: dict, prompt: str, n_chars: int, rng: random.Random) -> str:
    out = prompt
    for _ in range(n_chars):
        cands = model.get(out[-ORDER:])
        if not cands:
            break
        out += rng.choice(cands)
    return out[len(prompt):]


class InferenceHandler(BaseHTTPRequestHandler):
    """POST /generate 로 프롬프트를 받아 이어질 글자를 생성하는 핸들러."""
    model = None  # 서버 기동 시 주입

    def log_message(self, *args):  # 콘솔을 어지럽히는 접근 로그 끄기
        pass

    def do_GET(self):
        if self.path == "/health":  # 로드밸런서가 두드리는 생존 확인 엔드포인트
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._send(404, {"error": "not found"})
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        t0 = time.perf_counter()
        rng = random.Random(body.get("seed", 0))  # 요청별 seed 로 재현 가능
        text = generate(self.model, body["prompt"], body.get("n_chars", 60), rng)
        # 실제 LLM 추론은 요청당 수십~수백 ms GPU 계산이 걸립니다.
        # 그동안 CPU 는 다른 요청을 받을 수 있으므로 sleep 으로 흉내 냅니다.
        time.sleep(0.005)
        self._send(200, {"completion": text,
                         "model_ms": (time.perf_counter() - t0) * 1000})

    def _send(self, code: int, payload: dict):
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def call(url: str, prompt: str, seed: int) -> tuple[float, dict]:
    """API 를 한 번 호출하고 (왕복 지연 ms, 응답) 을 돌려줍니다."""
    req = urllib.request.Request(
        url + "/generate",
        data=json.dumps({"prompt": prompt, "n_chars": 60, "seed": seed}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as r:
        resp = json.loads(r.read())
    return (time.perf_counter() - t0) * 1000, resp


if __name__ == "__main__":
    print("[1] 모델 준비 — tiny_corpus 로 문자 3-gram 마르코프 모델 구축")
    text = hjh_data.tiny_corpus()
    InferenceHandler.model = build_model(text)
    print(f"    코퍼스 {len(text):,}자 → 상태 {len(InferenceHandler.model):,}개")
    print("    (실제 서빙에서는 이 자리에 학습된 트랜스포머 가중치가 들어갑니다)")

    print("\n[2] 서버 기동 — port=0 으로 빈 포트를 OS 에게 자동 할당받기")
    server = ThreadingHTTPServer(("127.0.0.1", 0), InferenceHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    print(f"    추론 API 가동: {base}  (엔드포인트: GET /health, POST /generate)")
    with urllib.request.urlopen(base + "/health", timeout=5) as r:
        print(f"    헬스체크 응답: {r.read().decode()}")

    print("\n[3] 첫 호출 — 프롬프트를 보내고 생성 결과 받기")
    ms, resp = call(base, "오늘 학생이 ", seed=1)
    print(f"    프롬프트: '오늘 학생이 '")
    print(f"    생성 결과: '{resp['completion'][:40]}...'")
    print(f"    왕복 지연 {ms:.1f}ms (그중 모델 계산 {resp['model_ms']:.1f}ms)")
    print("    → 지연시간 = 네트워크 + 대기열 + 모델 계산. 사용자는 합계만 느낍니다.")

    print("\n[4] 지연시간 통계 — 순차 호출 40회")
    lat = []
    t_start = time.perf_counter()
    for i in range(40):
        ms, _ = call(base, "어제 회사원이 ", seed=i)
        lat.append(ms)
    seq_sec = time.perf_counter() - t_start
    lat_sorted = sorted(lat)
    p50 = statistics.median(lat_sorted)
    p95 = lat_sorted[int(len(lat_sorted) * 0.95) - 1]
    print(f"    p50={p50:.2f}ms  p95={p95:.2f}ms  최대={lat_sorted[-1]:.2f}ms")
    print(f"    순차 처리량: {40 / seq_sec:.1f} 요청/초")
    print("    → 평균이 아니라 p95/p99 를 보는 이유: 느린 5%가 고객 불만을 만듭니다.")

    print("\n[5] 동시 호출 40회 — 스레드 서버의 처리량 확인 (클라이언트 8개)")
    results = []
    lock = threading.Lock()

    def worker(seeds):
        for s in seeds:
            ms, _ = call(base, "주말에 개발자가 ", seed=s)
            with lock:
                results.append(ms)

    t_start = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(range(k, 40, 8),))
               for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    con_sec = time.perf_counter() - t_start
    print(f"    동시 처리량: {40 / con_sec:.1f} 요청/초 (순차 대비 {seq_sec / con_sec:.1f}배)")
    print("    → 실전 서버(vLLM 등)는 여기에 배칭 + KV 캐시로 처리량을 더 끌어올립니다.")

    print("\n[6] 서버 종료 — 배포의 마지막 예절, 우아한 종료(graceful shutdown)")
    server.shutdown()
    server.server_close()
    print("    서버를 정상 종료했습니다. (실서비스라면 새 버전으로 교체하는 순간입니다)")
