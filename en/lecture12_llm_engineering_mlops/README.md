# Lecture 12 — LLM Engineering and MLOps

> We rebuild, in miniature, the *entire* process by which a large language model like ChatGPT
> comes into existence. From building a tokenizer through pre-training, SFT, DPO, quantization,
> serving, and monitoring — this is the capstone of the curriculum, the lecture that takes you
> from "someone who uses LLMs" to "someone who understands how they are made."

## What You Will Learn in This Lecture

- The full pipeline by which a large language model (LLM) rolls off the line like a factory product: pre-training → annealing → SFT → preference alignment → quantization → deployment
- The real work of the "data factory": BPE tokenizers, data packing, deduplication, and contamination checks
- Training a two-layer mini GPT yourself and witnessing the moment when "predicting the next token alone gives rise to grammar"
- A feel for estimating "what would it cost to train this model" using scaling laws
- Implementing LoRA, SFT, and DPO from scratch with no external libraries, so you understand the principles
- int8 quantization, an HTTP inference server, drift monitoring — how a model becomes a service and gets operated
- A realistic career roadmap for non-CS professionals growing into AI talent (the curriculum's closing chapter)

Each stage of a real large-scale training run (billions to trillions of tokens, thousands of GPUs)
is reproduced here at miniature scale with **models of tens to hundreds of thousands of parameters**.
Every level is designed so training finishes within 90 seconds on a laptop CPU.
Because we implement everything ourselves instead of using libraries like transformers,
you come to understand the LLM not as a black box but as "a machine you can hold in your hands."

## Prerequisite Lectures

- **lecture08 — Deep Learning Foundations** (you should be able to read tensors, gradient descent, and a training loop)
- **lecture10 — Natural Language Processing (NLP)** (the concepts of tokens, embeddings, and attention)
- **lecture11 — Working with LLMs and RAG** (experience with prompts and LLM APIs makes the "builder's side" of the story far more vivid)

Concept-first levels like level00, 05, and 11 are readable even if you are missing some prerequisites.

## Level Contents

| Level | Title | Difficulty |
|---|---|---|
| [level00](level00_the_big_picture/README.md) | What It Means to Build a Model — The Big Picture | ⭐ |
| [level01](level01_bpe_tokenizer/README.md) | Building a Tokenizer — Implementing BPE from Scratch | ⭐⭐⭐ |
| [level02](level02_data_pipeline_packing/README.md) | Training Data Pipelines and Packing | ⭐⭐⭐ |
| [level03](level03_data_quality_contamination/README.md) | Data Quality, Deduplication, and Contamination Checks | ⭐⭐⭐⭐ |
| [level04](level04_mini_gpt_pretraining/README.md) | Pre-training a Mini GPT | ⭐⭐⭐⭐⭐ |
| [level05](level05_scaling_laws_costs/README.md) | Scaling Laws and Estimating Training Costs | ⭐⭐⭐ |
| [level06](level06_finetuning_lora/README.md) | Fine-tuning and LoRA | ⭐⭐⭐⭐ |
| [level07](level07_sft/README.md) | Instruction Tuning (SFT) | ⭐⭐⭐⭐ |
| [level08](level08_dpo/README.md) | Preference Alignment (DPO) | ⭐⭐⭐⭐⭐ |
| [level09](level09_quantization/README.md) | Quantization and Model Compression | ⭐⭐⭐⭐ |
| [level10](level10_serving_deployment/README.md) | Model Serving and Deployment | ⭐⭐⭐⭐ |
| [level11](level11_monitoring_governance_career/README.md) | Monitoring, Cost, Governance — and Your Career Roadmap | ⭐⭐⭐ |

## Fast Track (Short on Time? Do These 5)

1. **level00** — The big picture: start with how a model is born, stage by stage
2. **level04** — Pre-training a mini GPT: the heart of this lecture. The moment a hand-built model learns to form sentences
3. **level05** — Scaling laws: answering "what would it cost to build a GPT" with actual numbers
4. **level07** — SFT: the decisive step where a "continuation machine" turns into an "assistant"
5. **level09** — Quantization: why an LLM can run on your company laptop

## Where This Lecture Shows Up at Work

- **AI adoption decision meetings**: judging "should we build from scratch, fine-tune, or use an API" with a stage-by-stage sense of cost (level00, 05)
- **Working with vendors and AI teams**: understanding a report like "we fine-tuned with LoRA and serve it in int4," and asking the right follow-up questions (level06, 09)
- **Customizing a model on company data**: preparing and quality-controlling the data for a fine-tuning project that teaches the model your documents and tone (level02, 03, 07)
- **Operating an AI service**: watching response latency, cost, and quality drift on a dashboard, and catching anomalies early (level10, 11)
- **Security and compliance reviews**: checking risk factors such as training-data contamination, PII in logs, and API-key management (level03, 11)
- **Hiring and career moves**: explaining the vocabulary of LLM-engineering job postings (pre-training, RLHF/DPO, quantization, serving) in your own words (all levels)

## Tips for Working Through It

- We recommend going through the levels in order. The lecture is designed as a single "model factory tour":
  levels 01–03 prepare the raw material (data), level04 bakes the model, levels 06–08 refine it,
  and levels 09–11 handle packaging, shipping, and after-sales care.
- Every level is meant to be read while you run `main.py` yourself. To run: use the repo-root virtual
  environment and `python3 main.py` (see the root `SETUP.md` for environment setup).
- The training data is the mini English corpus from `common/hjh_data.py`, so no internet connection is needed.
- Clumsy sentences from the mini models are perfectly normal. The point is to watch the process of
  gradual improvement, not to reach ChatGPT-level quality. Each level's notes quantify, in numbers,
  how far these minis are from the real thing.
