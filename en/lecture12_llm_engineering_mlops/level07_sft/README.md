# Lecture 12 · Level 07 — Instruction Tuning (SFT)

> With a chat template and loss masking, we teach the base model — a "continuation machine" — how to answer questions.

**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level06 / **Estimated time** 60 min

## 1. Why Learn This — The Business View

The chatbot-style LLMs you use every day are actually made in two phases. A base model fresh out of pre-training knows only how to "continue the next word" of internet text. Type "tell me the refund policy" and it does not answer — it continues with whatever text plausibly follows that sentence, perhaps by listing more similar questions.

The process that turns this continuation machine into an "assistant that answers when asked" is instruction tuning (SFT, Supervised Fine-Tuning). You create tens to hundreds of thousands of (instruction, model-answer) pairs and teach the model: "when input arrives in this format, respond in this format." When a company says "we'll SFT on our FAQ data" while building its own chatbot, this is exactly what they mean — and the quality of the SFT data effectively determines the quality of the chatbot. Getting a feel for how much behavior a handful of examples creates puts you in a far stronger position when negotiating data-construction budgets and timelines.

## 2. Grasping It Through an Analogy

Picture a new hire who is encyclopedically knowledgeable but has no situational sense. Knowledge (pre-training) is already ample. But when a customer asks "how do I track my delivery?", instead of answering, they start writing an essay about logistics. What this hire needs is not knowledge but **role-play training for customer service**.

SFT is script-based role-play. You drill hundreds of scripts of the form "when the customer says this → you answer this." One rule matters: grading (loss computation) happens **only on the parts the new hire is supposed to say**. The customer's lines (the prompt) are not something the hire will generate, so we do not grade how well they "predicted" those. That is loss masking.

A script also needs stage directions: "customer's line starts here," "your line starts here," "line ends." Those are the chat template's special tokens (<|user|>, <|assistant|>, <|end|>).

## 3. Core Concepts

### 3-1. The Chat Template and Special Tokens

To the model, a conversation is just one long token sequence. Roles are marked with special tokens.

```
<|user|>What is 3 plus 5?<|assistant|>3 plus 5 is 8.<|end|>
```

A special token must be a **single whole token** that cannot collide with ordinary characters. Real models' template grammars (ChatML and friends) differ slightly, but the structure is the same. At inference you fill in everything up through `<|assistant|>` and let the model generate the rest, stopping when `<|end|>` appears. One of the things SFT teaches is precisely "emit <|end|> when the answer is done" — a model that fails to learn this rambles on forever.

### 3-2. Loss Masking — Why Grade Only the Response

The training example is one line of prompt+response, but applying cross entropy to all of it makes the model spend capacity on "predicting the user's question" too. At worst it picks up the habit of mimicking the user or parroting the question back. So the prompt span's target labels are set to -100 (PyTorch cross_entropy's ignore_index) and excluded from grading. These are the two key lines of the exercise:

```python
Y[k, :] = -100                      # default: exclude everything from grading
Y[k, plen-1:] = response tokens     # grade only the response (and <|end|>)
```

Why the index is plen-1: because of the one-position shift (x=ids[:-1], y=ids[1:]), the prediction of the *first response token* happens at the position of the prompt's last token (<|assistant|>).

### 3-3. The Three Requirements of SFT Data

When building real SFT data, what matters is (1) **diversity** — many phrasings of the same intent, (2) **exemplariness** — responses must be good enough to copy verbatim (the model imitates responses literally), (3) **format consistency** — consistent template, tone, and end token. "Garbage in, garbage out" (level03) shows up most dramatically in SFT: 100 bad responses can poison 10,000 good ones.

### 3-4. Learning Format vs. Learning Ability

What SFT actually teaches is mostly **format and behavior**. Most knowledge and ability were already built during pre-training; SFT wraps an interface around them. So with a skimpy pre-training run, like our mini model's, the post-SFT answers come out "perfect in format, wrong in content." This is why industry cares so much about base-model size.

## 4. Hands-On — main.py

Run:

```bash
cd lecture12_llm_engineering_mlops/level07_sft
python3 main.py
```

- **[1]** 34 toy instruction pairs are generated in code: greetings (8 names), addition (combinations of 1–5 with a≠b only — a=b is held out for testing), and stock lookups (6 items). A character-level tokenizer with 4 special tokens is built (vocab 55).
- **[2]** The base model pre-trains on tiny_corpus for 300 steps (loss 4.21 → 0.21).
- **[3]** See the base model's limits with your own eyes. Given "Today an office worker " it continues plausibly, but put "What is 3 plus 5?" into the chat template and out comes corpus-flavored, beside-the-point continuation: `'hecked a program. This evening a developer c'`. **A base model does not answer; it continues.**
- **[4]** One templated line is printed alongside its loss mask (- excluded, # graded). Confirm the 19 prompt tokens are excluded and only the 15 response tokens are trained.
- **[5]** SFT runs 700 steps grading only the response part. The loss falls to 0.003.
- **[6]** The responses to the same instructions transform completely. "What is 3 plus 5?" → "3 plus 5 is 8." — and it stops by itself at <|end|>. On the never-seen "What is 4 plus 4?" it answers "4 plus 4 is 6." — **perfect format, wrong content** — the classic small-model limitation explained in section 3-4.

## 5. Try It Yourself

1. **End-token experiment**: in make_sft_tensors, mask out the <|end|> token as well. After SFT, watch why answer() keeps generating all the way to max_new. (Hint: the model loses its chance to learn the "stop" signal.)
2. **Masking-off experiment**: fill Y with all tokens, unmasked, and train. How do response quality and the loss curve change? (Hint: in this mini exercise the difference may be small. The longer and more varied prompts are relative to responses, the bigger the difference.)
3. **Add a new task**: add 10 pairs of a new instruction type, like "Spell ○○ backwards", and retrain. How few pairs does it take to learn the format? (Hint: new characters mean rebuilding the vocab, so make sure the new data is included in all_text.)

## 6. Common Mistakes

- **Splitting special tokens into characters**: encode "<|user|>" as an ordinary string and it becomes 8 character tokens. Template separators must be registered in the vocab as whole tokens. In real frameworks, forgetting add_special_tokens is exactly this mistake.
- **Off-by-one**: because of the x=ids[:-1], y=ids[1:] shift, the mask boundary is plen-1, not plen. Get the boundary wrong by one and the first token after <|assistant|> is never trained, making response openings unstable.
- **Evaluating only on training data**: always include never-seen instructions, as in [6]. Answering only the training examples well is memorization, not ability.
- **Leaving bad responses in SFT data**: the model imitates responses verbatim. Let a "dunno lol" slip in and it learns that tone too.

## Next Level Preview

A model that has finished SFT "does answer" — but has no idea which of two answers is better. The technique that tilts the model toward the answer humans prefer, between a polite one and a rude one — we move on to level08, preference alignment (DPO).
