# Lecture 10 — Natural Language Processing (NLP)

> We build, from the ground up, the technology that lets computers handle human language.
> Starting with string cleanup and regular expressions, we move through TF-IDF, sentiment
> analysis, and embeddings, implement attention and a mini transformer ourselves, and
> arrive at the GPT/BERT pre-training paradigm.

## What You Learn in This Lecture

- Why language is uniquely hard for computers — ambiguity, context, and how word forms
  shift (with agglutinative languages like Korean as the extreme case)
- The preprocessing pipeline and regular expressions that turn messy text into something analyzable
- The three generations of turning text into numbers: BoW/TF-IDF → word embeddings → contextual embeddings
- Building a review sentiment analyzer yourself, and interpreting which words the model used as evidence
- Assembling RNNs, attention, and transformers part by part in numpy and torch
- The heart of the pre-training paradigm that changed the world: GPT (continuation) and BERT (fill-in-the-blank)

This lecture **deliberately avoids** NLP-specific packages like nltk, spaCy, or
transformers. Implementing tokenizers, TF-IDF, and attention yourself is what turns
"what the library does inside" from a black box into a glass box — so that when you
use the tools in the next lecture (lecture11 — Working with LLMs and RAG), you know
exactly what they're doing.

## Prerequisite Lectures

- **lecture02 — Python Programming Basics** (strings, lists, dictionaries, functions)
- **lecture03 — Working with Data** (NumPy array operations)
- **lecture06 — Introduction to Machine Learning** (classification, logistic regression, train/test split)
- **lecture08 — Deep Learning Foundations** (needed for the torch levels from level07 on; levels 00–06 don't require it)

## Level Contents

| Level | Title | Difficulty |
|---|---|---|
| [level00](level00_why_language_is_hard/README.md) | Why Language Is Hard for Computers | ⭐ |
| [level01](level01_text_preprocessing/README.md) | Text Preprocessing Basics | ⭐⭐ |
| [level02](level02_regex/README.md) | Regular Expressions | ⭐⭐ |
| [level03](level03_tokenization_korean/README.md) | Tokenization and the Quirks of Korean | ⭐⭐⭐ |
| [level04](level04_bow_tfidf/README.md) | Bag of Words and TF-IDF | ⭐⭐⭐ |
| [level05](level05_text_classification/README.md) | Hands-On Text Classification — Review Sentiment Analysis | ⭐⭐⭐ |
| [level06](level06_word_embeddings/README.md) | Word Embeddings | ⭐⭐⭐ |
| [level07](level07_rnn_sequences/README.md) | RNNs and Sequence Models | ⭐⭐⭐⭐ |
| [level08](level08_attention/README.md) | The Attention Mechanism | ⭐⭐⭐⭐ |
| [level09](level09_transformer_anatomy/README.md) | Dissecting the Transformer Architecture | ⭐⭐⭐⭐ |
| [level10](level10_mini_transformer/README.md) | Implementing a Mini Transformer from Scratch | ⭐⭐⭐⭐⭐ |
| [level11](level11_pretraining_paradigm/README.md) | The Pre-training Paradigm — BERT and GPT | ⭐⭐⭐⭐ |

## The Fast Track (short on time? just these 5)

1. **level01 Text preprocessing** — where every text task begins. The #1 most-used skill in practice.
2. **level04 BoW & TF-IDF** — the standard for "text to numbers". The base of search, classification, keyword extraction.
3. **level05 Text classification** — the full review-sentiment pipeline in one sitting.
4. **level08 Attention** — if you can pick only one key to understanding today's AI, it's this.
5. **level11 The pre-training paradigm** — why GPT/BERT matter; the big picture of the LLM era.

## Where This Lecture Shows Up at Work

- **VOC (voice of customer) analysis**: automatically classify the thousands of daily
  reviews and inquiries as positive/negative, extract the complaint keywords, and turn
  them into a weekly report. (level01, 04, 05)
- **Extracting information from documents**: bulk-extract amounts, dates, and contacts
  from stacks of contracts and quotes with regex and drop them into a spreadsheet. (level02)
- **The base of internal search and summarization**: the intranet search and RAG
  chatbot that answer "where's the vacation policy again?" stand entirely on this
  lecture's tokenization, TF-IDF, and embeddings. (level03, 04, 06)
- **An informed eye for LLMs**: why billing is per token, why context length is capped,
  why prompts behave the way they do — knowing the transformer explains all of it. (level08–11)

## How to Run

```bash
cd lecture10_nlp_text/level00_why_language_is_hard
python3 main.py
```

Every exercise works without an internet connection; the data is generated on the
spot by `common/hjh_data.py`. Even the torch levels (07, 10, 11) are designed small
enough to finish within 90 seconds on CPU.
