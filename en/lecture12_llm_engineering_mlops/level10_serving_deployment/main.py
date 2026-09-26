"""
Hands-on model serving.
We wrap a character-level Markov mini model built from tiny_corpus in an
http.server-based inference API, call it repeatedly from a client to measure
latency (p50/p95) and throughput, then shut the server down cleanly.
A sequential-vs-concurrent comparison also shows the throughput difference
of a threaded server.
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
ORDER = 3  # a Markov model that picks the next character from the previous 3


def build_model(text: str) -> dict:
    """Character 3-gram Markov table: 'previous 3 chars' -> [next-character candidates]."""
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
    """Handler that takes a prompt via POST /generate and produces a continuation."""
    model = None  # injected at server startup

    def log_message(self, *args):  # silence access logs that clutter the console
        pass

    def do_GET(self):
        if self.path == "/health":  # the liveness endpoint that load balancers knock on
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._send(404, {"error": "not found"})
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        t0 = time.perf_counter()
        rng = random.Random(body.get("seed", 0))  # per-request seed for reproducibility
        text = generate(self.model, body["prompt"], body.get("n_chars", 60), rng)
        # Real LLM inference takes tens to hundreds of ms of GPU compute per request.
        # The CPU can accept other requests meanwhile, so we imitate that with a sleep.
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
    """Call the API once and return (round-trip latency in ms, response)."""
    req = urllib.request.Request(
        url + "/generate",
        data=json.dumps({"prompt": prompt, "n_chars": 60, "seed": seed}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as r:
        resp = json.loads(r.read())
    return (time.perf_counter() - t0) * 1000, resp


if __name__ == "__main__":
    print("[1] Model prep — building a character 3-gram Markov model from tiny_corpus")
    text = hjh_data.tiny_corpus()
    InferenceHandler.model = build_model(text)
    print(f"    corpus {len(text):,} chars -> {len(InferenceHandler.model):,} states")
    print("    (in real serving, trained transformer weights sit in this slot)")

    print("\n[2] Server startup — port=0 asks the OS to auto-assign a free port")
    server = ThreadingHTTPServer(("127.0.0.1", 0), InferenceHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    print(f"    inference API up: {base}  (endpoints: GET /health, POST /generate)")
    with urllib.request.urlopen(base + "/health", timeout=5) as r:
        print(f"    health check response: {r.read().decode()}")

    print("\n[3] First call — send a prompt and receive the generation")
    ms, resp = call(base, "Today a student ", seed=1)
    print(f"    prompt: 'Today a student '")
    print(f"    completion: '{resp['completion'][:40]}...'")
    print(f"    round trip {ms:.1f}ms (of which model compute {resp['model_ms']:.1f}ms)")
    print("    -> latency = network + queueing + model compute. Users feel only the sum.")

    print("\n[4] Latency statistics — 40 sequential calls")
    lat = []
    t_start = time.perf_counter()
    for i in range(40):
        ms, _ = call(base, "Yesterday an office worker ", seed=i)
        lat.append(ms)
    seq_sec = time.perf_counter() - t_start
    lat_sorted = sorted(lat)
    p50 = statistics.median(lat_sorted)
    p95 = lat_sorted[int(len(lat_sorted) * 0.95) - 1]
    print(f"    p50={p50:.2f}ms  p95={p95:.2f}ms  max={lat_sorted[-1]:.2f}ms")
    print(f"    sequential throughput: {40 / seq_sec:.1f} requests/s")
    print("    -> Why we watch p95/p99, not the mean: the slow 5% is what creates complaints.")

    print("\n[5] 40 concurrent calls — the threaded server's throughput (8 clients)")
    results = []
    lock = threading.Lock()

    def worker(seeds):
        for s in seeds:
            ms, _ = call(base, "Over the weekend a developer ", seed=s)
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
    print(f"    concurrent throughput: {40 / con_sec:.1f} requests/s ({seq_sec / con_sec:.1f}x sequential)")
    print("    -> Production servers (vLLM etc.) push throughput further with batching + KV caching.")

    print("\n[6] Server shutdown — deployment's final courtesy: the graceful shutdown")
    server.shutdown()
    server.server_close()
    print("    Server shut down cleanly. (In production, this is the moment you swap in the new version.)")
