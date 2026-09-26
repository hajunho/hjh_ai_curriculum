# Lecture 10 · Level 01 — Text Preprocessing Basics

> We build, step by step, the cleaning pipeline that turns messy review text into something you can actually analyze.
**Difficulty** ⭐⭐ / **Prerequisites** level00 / **Estimated time** 35 min

## 1. Why Learn This — The Business View

In a text-analysis project, most of the time goes not into modeling but into
**cleaning**. Customer reviews arrive full of emoticons, stretched-out words
("sooooo good"), URLs, stray whitespace, and inconsistent capitalization
("RMA", "rma", "R.M.A."). Count them as-is and "good" and "goooood" get tallied
as different words, and every statistic comes out wrong.

Preprocessing isn't glamorous, but its effect is real. With the exact same model,
classification accuracy routinely moves by several percentage points depending on
preprocessing quality — and above all, it makes the counts **fit to put in a
report**. You can't exactly publish "this month's most-mentioned keyword: lolll".

## 2. Grasping It Through an Analogy

Preprocessing is **prepping ingredients before cooking**. The vegetables you bring
home from the market (raw text) have dirt on them (stray symbols), come in uneven
sizes (upper/lower case, full-width characters), and carry roots and outer leaves
you won't eat (stopwords). Whatever the dish (analysis), the prep almost always
runs in the same order: wash → trim → cut to size → discard the inedible parts.

The key point is that **the prep depends on the dish**. You devein shrimp for a
delicate soup but leave them whole for the grill. Likewise, in sentiment analysis
"!!!" can be a precious signal (strong emotion), so deleting it blindly is wrong —
while for keyword counting, deleting it is exactly right. There is no "standard
preprocessing", only "preprocessing that fits the purpose".

## 3. Core Concepts

### 3-1. Normalization — make same things the same

Gather together strings that differ only in surface form.

| Step | Example |
|---|---|
| Lowercasing (case folding) | "RMA Center", "rma center" → "rma center" |
| Whitespace cleanup | "delivery&nbsp;&nbsp;was&nbsp;&nbsp;" → "delivery was" (runs of spaces, ends trimmed) |
| Collapsing repeats | "sooooo good", "lolllll" → "soo good", "loll" |
| Special characters | "★highly recommend★" → "highly recommend" (keep or drop depends on purpose) |

In English the single biggest normalizer is lowercasing; collapsing character
repeats handles the "sooooo"/"lolll" style of social-media emphasis.

### 3-2. Stopword removal — throw away words with no signal

Words like "and", "the", "but" appear in nearly every sentence, so they have no
power to distinguish documents. We call them stopwords and exclude them from
counting and search.

Two cautions:
- Stopword lists are **domain-specific**. In shopping reviews, "product" is
  effectively a stopword; in general news it is not.
- In sentiment analysis, deleting negators like "not" or "never" as stopwords
  turns "not good" into "good". What to delete is always decided together with
  the purpose.

### 3-3. Word endings — crude stemming and its dangers

Even in English, inflection scatters one concept across many strings:
"deliver, delivers, delivered, delivery". If you don't peel the endings,
"delivered" and "delivery" get counted as unrelated words.

This level uses a **crude rule** that strips common suffixes ("-ing", "-ed",
"-es", "-ly", "-s") from word endings. The rule is deliberately naive and
misfires — it turns "rating" into "rat" and "was" into "wa" (see section 6) —
much like a naive particle-stripper in Korean turns non-particles into casualties.
(Korean, being agglutinative, needs this even more desperately: without particle
handling, one noun splinters into four or more strings.) The proper fix —
real tokenization and subwords — is level03. Today's goal is the instinct:
*handle word endings or your counts lie*.

### 3-4. Thinking in pipelines

Preprocessing is a **fixed sequence** of steps. Change the order and the result
changes. For example, if you strip special characters first, the URL-removal rule
("starts with http…") stops working. So in practice each step becomes a small
function, chained in order into a pipeline that every text passes through the same
way. A bonus: you can pull out "the result after step 3" for easy debugging.

## 4. Hands-On — main.py

Run:

```bash
python3 main.py
```

- **[1] Dirty-review generation**: takes hjh_data reviews and deliberately smears
  on emoticons, URLs, repeated characters, and stray whitespace to make 8
  "realistic" messy reviews (fixed seed).
- **[2] Step-by-step cleaning**: one sample passes through the 5 pipeline steps
  (lowercase + strip URLs → clean special chars → collapse repeats → normalize
  whitespace → stopwords/crude suffixes), printing before/after at each step.
  You watch the text get cleaner in front of you.
- **[3] Batch cleaning**: all 8 reviews go through the pipeline, original → result.
- **[4] Frequency comparison**: top-8 word counts on the same batch, before vs
  after. Before, "My" and "my" and "honest." are tallied separately and
  "[Photo" sneaks in as a word; after, the same concepts merge.

Actual output of [4]:

```text
    before                   | after
    --------------------------------------------------
    is             x6      | is           x6
    [Photo         x5      | photo        x6
    review]        x5      | my           x5
    My             x4      | review       x5
    star           x4      | star         x4
    rating         x4      | rat          x4
    completely     x4      | complete     x4
    honest.        x4      | honest       x4
```

The heart of the code is the list `PIPELINE = [(step name, function), ...]`.
Each step is a pure function taking a string and returning a string, so
reordering or removing steps is a one-line list edit.

## 5. Try It Yourself

1. **(easy)** Add "my" and "is" to `STOPWORDS` and watch how the top of the
   frequency table in [4] changes.
2. **(medium)** Comment out the `collapse_repeats` step in the pipeline, run it,
   see how "sooooo" contaminates the counts, then put it back.
3. **(challenge)** Build an emotion-preserving version: instead of deleting "!"
   and ":(", add a step that replaces them with `<exclaim>` and `<sad>` tokens.
   Hint: adapt the pattern `re.sub(r"!+", " <exclaim> ", text)`.

## 6. Common Mistakes

- **Deleting everything on reflex**: reflexively stripping symbols and emoticons
  erases emotional signal. Always weigh the three options: delete / keep / replace.
- **Copying a stopword list from another domain**: a news stopword list used on
  reviews contains oddballs like "reporter" while missing "product".
- **Trusting crude suffix-stripping too much**: it clips "rating" into "rat" and
  "was" into "wa" — word-end letters mistaken for suffixes. Crude rules are a
  stopgap, nothing more (see level03).
- **Overwriting the original**: if the cleaned text overwrites the raw column,
  you can never undo a rule change. Always keep the original untouched.

## Next Level Preview

We quietly used a regular expression to delete URLs in this level. In level02 we
learn regular expressions head-on and turn them into a tool that bulk-extracts
phone numbers, emails, and amounts from business documents.
