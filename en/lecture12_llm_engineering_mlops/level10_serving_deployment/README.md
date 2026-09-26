# Lecture 12 · Level 10 — Model Serving and Deployment

> From a finished model file to a real service — we stand up an inference API server ourselves and measure latency and throughput.

**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level09_quantization / **Estimated time** 50 min

## 1. Why Learn This — The Business View

The model we have built so far is just a file sitting on disk. Customers cannot use a file. What customers use is "something that returns an answer when you send a question" — an API. The work of connecting the model file to an API is serving, and putting that on real servers and beginning operations is deployment.

This stage matters because money and user experience are decided here. However brilliant the model, if a response takes 20 seconds the customer leaves. Conversely, serve well and the same GPU handles 10x the requests, cutting unit cost to a tenth. That is why serving-specific software like vLLM became an industry of its own. When a "chatbot rollout" project starts at a company, half the meetings are not about model quality but about serving questions: "How fast are responses? How many concurrent users? What's the server bill?" Finish this level and you will know exactly what the numbers flying around that meeting mean.

## 2. Grasping It Through an Analogy

Recruiting a brilliant chef (the model) does not open a restaurant. You must design the order counter (the API endpoint), the order-ticket format (the request schema), the flow between dining room and kitchen (the server architecture), plus "how many minutes from order to table" (latency) and "how many plates per hour" (throughput).

Latency and throughput are different concepts. If the chef takes 10 minutes per dish, latency is 10 minutes. But with 8 burners cooking 8 dishes at once, each customer still waits 10 minutes — yet hourly sales (throughput) is 8x. LLM serving's batching is exactly this principle.

The KV cache is the chef's mise en place. When a conversation runs long, rather than recomputing the entire dialogue from scratch each time, you keep the earlier computation in the fridge and cook only the new sentence. It is the same as not re-brewing the stock from scratch every time a customer orders another course.

## 3. Core Concepts

### 3.1 The Journey from Model File to Service

1. **Model file** — the weights, quantized and finished in level09 (e.g., a GGUF file)
2. **Inference engine** — the program that loads the weights into memory and generates tokens
3. **API server** — receives HTTP requests, forwards to the engine, responds in JSON
4. **Deployment infrastructure** — multiple servers, a load balancer, health checks, monitoring

In this exercise we build miniatures of items 2–3 using nothing but the Python standard library. The names you hear at work map on like this: Ollama bundles 1–3 for personal PCs ("say the model's name and it downloads and runs it"), Hugging Face is the warehouse and hosting service for uploading and downloading model files, and vLLM is the high-performance inference engine for large-scale services. Know the concepts; we write all the code ourselves.

### 3.2 Latency — Watch p95, Not the Mean

Latency is the time from sending a request to receiving the response. Make 100 calls, sort them fastest to slowest: the 50th value is p50 (the median) and the 95th is p95. Averages create illusions: if 99 calls take 0.1s and one takes 30s, that one person is an angry customer. Service-level objectives (SLOs) are therefore usually written in percentiles, like "p95 < 1s."

LLMs split latency in two: time to first token (TTFT) and time per generated token. This is why a chatbot that starts streaming its first characters quickly after "thinking…" feels so much faster.

### 3.3 Throughput and Concurrency

Throughput is requests (or tokens) handled per unit time. During GPU inference the CPU mostly sits waiting for results, so accepting other requests in the meantime raises throughput. The threaded server in this exercise demonstrates it. Production engines go further with continuous batching — folding many requests' token generation into one GPU operation — multiplying throughput by tens again.

### 3.4 The KV Cache

To produce one new token, a transformer needs the attention keys (K) and values (V) of every previous token. Recompute them every time and the cost of generating the n-th token blows up in proportion to n. Stash the K·V once computed in memory (the KV cache), and each token becomes "compute 1 new token + look up the cache." The price is GPU memory proportional to conversation length — which is why "concurrent users × context length" is the key variable in sizing server memory.

## 4. Hands-On — main.py

How to run:

```bash
cd lecture12_llm_engineering_mlops/level10_serving_deployment
python3 main.py
```

No internet access involved. Server and client communicate only inside your machine (127.0.0.1).

Stage-by-stage guide:

- **[1]** Builds a character 3-gram Markov model from tiny_corpus (84,505 characters → 174 states). We place an instantly-responding micro model in the "inference engine" slot instead of a real transformer.
- **[2]** Starts a `ThreadingHTTPServer` with `port=0`. Passing 0 lets the operating system pick a free port automatically — no port collisions. We probe `/health` for liveness, the same convention a real load balancer uses to knock on a server.
- **[3]** Sends a prompt to `POST /generate` and inspects the completion and round-trip latency (about 7ms here, of which ~6ms is "model compute"). The handler contains a 5ms `sleep` per request, imitating real GPU compute time (time the CPU spends waiting).
- **[4]** Makes 40 sequential calls and computes p50/p95/max latency and sequential throughput (in this run: p50≈6.9ms, p95≈7.2ms, about 147 requests/s).
- **[5]** Fires the same 40 calls concurrently from 8 client threads. Watch throughput jump — about 889 requests/s here, 6.1x the sequential rate. Individual latencies stay similar while throughput soars: that is the essence of concurrency.
- **[6]** Ends with a graceful shutdown via `server.shutdown()`. Finishing in-flight requests before going down is basic etiquette for deployment swaps.

The part of the code to study is `InferenceHandler`. The 30 lines of `do_POST` — read JSON → call the model → respond in JSON — are the minimal skeleton of a commercial service like the OpenAI API, unchanged.

## 5. Try It Yourself

1. **(Basic)** Raise `n_chars` from 60 to 300 and request longer generations. How does latency change? Hint: edit the request body in the client's `call()`. The proportionality of generation length and latency connects directly to why LLM pricing is per token.
2. **(Intermediate)** Vary the client thread count in [5] across 1, 2, 4, 8, 16 and plot the throughput curve. At what point does adding threads stop helping? Hint: there is a point where the server side saturates.
3. **(Challenge)** Right now, sending malformed JSON to `/generate` raises an exception in the server. Fix it with try/except to return a 400 response. Hint: wrap `json.loads` and use `self._send(400, {"error": ...})`. Input validation is half of all serving code.

## 6. Common Mistakes

- **Reporting only mean latency.** The average looks pretty in a demo, but real user complaints come from the tail (p95, p99). Measure in percentiles from day one.
- **Hardcoding a fixed port number.** Pin port 8000 and you collide with other programs — the maddening "fails only sometimes" bug. In tests and demos, `port=0` auto-assignment is the safe choice.
- **Confusing latency with throughput.** Batching raises throughput but can slightly *increase* individual latency. The right balance depends on your service goals.
- **Shutting down by force.** Kill the process mid-request and your customers see error screens. Follow the order: shutdown → drain → swap.
- **Loading the model on every request.** Weights should be loaded into memory once, at server startup. Load them inside the request handler and latency multiplies by tens.

## Next Level Preview

The service is out in the world. But deployment is not the end — it is the beginning. In the final level we build a monitoring dashboard that detects quality drift from operational logs, review cost, privacy, and security governance, and close the twelve-lecture journey with a realistic roadmap for growing from a non-CS background into an AI professional.
