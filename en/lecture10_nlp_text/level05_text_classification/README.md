# Lecture 10 · Level 05 — Hands-On Text Classification: Review Sentiment Analysis

> Build a sentiment analyzer with TF-IDF + logistic regression, then interpret the words the model used as evidence through its coefficients.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level04, lecture06 (logistic regression) / **Estimated time** 50 min

## 1. Why Learn This — The Business View

"How are customers reacting since the promotion?" — you cannot answer that by
reading 3,000 reviews one by one. Sentiment analysis classifies reviews
positive/negative automatically, feeding dashboards like "positive-share trend"
and "top drivers of negative reviews". VOC analysis, brand monitoring, support
ticket triage — text classification is the most-used NLP application in industry.

The other goal of this level is **explainability**. "The model is 87% accurate"
does not finish the report. Only when you can answer "why did it call this
review negative?" will the business trust the result. Logistic regression lets
you open the per-word coefficients, which is why it remains a beloved starting
point in practice.

## 2. Grasping It Through an Analogy

A sentiment analyzer is **a grader with a scoring rubric**. The rubric assigns
points per word: "friendly" +3, "satisfied" +2, "disappointed" −4, "broke" −3...
When a review arrives, the points of the words it contains are summed; above
zero means positive, below means negative.

Logistic regression is exactly this structure. The one difference: **no human
writes the rubric — the model works the scores out itself from labeled review
data**. After training, open the rubric (the coefficients) and you see plainly
"which words the model learned as evidence of positive or negative".
The keyword rules we wrote by hand in level00 are now written by the data.

## 3. Core Concepts

### 3-1. The whole pipeline at a glance

```
review text → [tokenize & vectorize: TF-IDF] → numeric vector → [logistic regression] → positive probability
```

No new parts to learn. Level04's TF-IDF turns text into vectors, and the
logistic regression from lecture06 classifies the vectors. Much of practical
NLP is exactly this: assembling parts you already own.

### 3-2. Train/test split — don't show the exam in advance

The data splits into training and evaluation sets (a lecture06 level04 revisit).
The rule that matters most for text: **the TF-IDF vocabulary and IDF are built
from the training data only, and merely applied to the test data**. Building
the vocabulary on all the data leaks future information (data leakage) and
inflates measured performance. That is precisely sklearn's distinction between
`fit_transform` (train) and `transform` (test).

### 3-3. Reading the coefficients — opening the model's rubric

A trained logistic regression has one coefficient per word.

- Large positive coefficient → the word raises the positive probability (evidence of positive)
- Large negative coefficient → evidence of negative
- Near 0 → the word barely matters

List the top and bottom coefficients and you get "excellent, helpful, friendly,
lovely" on one side and "cramped, noisy, doubt, torn" on the other. That list is
itself excellent report material — e.g. "the biggest driver of negative reviews
is delivery-delay language".

### 3-4. Reading probabilities — using the confidence too

Logistic regression outputs not 0/1 but **the probability of positive**.
0.98 is a confident positive; 0.52 is a borderline review. In practice you
split the workload on this confidence: automate the confident ones, send only
the ambiguous 0.4–0.6 band to humans. That slashes review headcount while
protecting quality.

### 3-5. The limits of this model

Being bag-of-words based, "not good" computes 'not' as a token unrelated to
'good', and sarcasm ("everyone said it was great, so I bought it…") stays hard.
Expressions absent from the training data are also out of reach. Still, within
a homogeneous domain (one shop's reviews) this simple model works remarkably
well, and the textbook way to decide whether a fancier model is worth it is
"by how much does it beat this baseline?".

## 4. Hands-On — main.py

Run:

```bash
python3 main.py
```

- **[1] Data prep**: loads 600 hjh_data.review_corpus reviews (300 positive /
  300 negative), splits 480 train / 120 test (fixed seed).
- **[2] Vectorization**: fits TfidfVectorizer on the training data only —
  115 vocabulary types — and merely transforms the test data.
- **[3] Train & evaluate**: logistic regression reaches **100.0% test accuracy**
  here, with a spotless confusion matrix (60/0/0/60). Don't celebrate too hard:
  our synthetic reviews are built from clean templates, so the classes are
  perfectly separable. Real-world review data lands far lower — the pipeline,
  not the score, is the lesson.
- **[4] The rubric revealed**: top-8 positive-evidence words (excellent +1.66,
  helpful +1.46, staff +1.46, friendly +1.46...) and top-8 negative-evidence
  words (cramped −1.75, noisy −1.75, doubt −1.27, torn −1.23...). The moment you
  see what the model actually learned.
- **[5] Judging new reviews**: 4 sentences never seen in training get a positive
  probability and their evidence words. For instance "Delivery took a whole week
  and the box arrived torn" → negative 7%, evidence torn(−0.49), arrived(−0.49),
  whole(−0.46). A report format that answers "why did it decide that?".

## 5. Try It Yourself

1. **(easy)** Add 2 reviews of your own to [5] and check the verdicts and
   evidence. Try a deliberately ambiguous sentence and see whether the
   probability lands near 0.5.
2. **(medium)** Shrink the training set from 480 to 100. What happens to
   accuracy? Observe the relationship between data volume and performance.
3. **(challenge)** Soften the "not good" problem: before vectorizing, add a
   preprocessing step that glues negations to the following word, turning
   "not good" into the single token "not_good", and compare the effect.
   Hint: `re.sub(r"\bnot (\S+)", r"not_\1", text)` — negation joining is a
   simple, effective trick used in real systems.

## 6. Common Mistakes

- **Fitting on the test data**: running `fit_transform` on the full dataset is
  data leakage. Fit on train, transform on test.
- **Stopping at accuracy**: on real data with skewed class ratios, always read
  the confusion matrix and per-class metrics (lecture06 level06) alongside.
- **Stretching coefficients into causation**: "the coefficient of 'friendly' is
  big" means "the word friendly appeared often in positive reviews" — it is not
  proof that a friendliness campaign will lift sales.
- **Using the model outside its training domain**: a model trained on shop
  reviews collapses on app-store reviews. New domain, retrain — that's the default.

## Next Level Preview

In TF-IDF, "delivery" and "shipping" are unrelated dimensions. In level06 we
give words **coordinates**, placing similar meanings close together — word
embeddings, trained ourselves from a co-occurrence matrix with SVD.
