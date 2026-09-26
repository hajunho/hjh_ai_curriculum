# Lecture 10 · Level 00 — Why Language Is Hard for Computers

> Computers are great with numbers — so why do they stumble over sentences? You'll run the failure cases yourself and feel the problem first-hand.
**Difficulty** ⭐ / **Prerequisites** none (the string basics from lecture02 help) / **Estimated time** 25 min

## 1. Why Learn This — The Business View

Most of a company's data is not tables — it is **text**. Customer reviews, support
tickets, emails, contracts, meeting notes — the industry rule of thumb says more than
80% of all data is unstructured text like this. You can total up a sales table in a
spreadsheet, but a question like "are customers unhappy with delivery?" can only be
answered by reading. There is far too much for a human to read, and if you hand it to
a computer — the computer doesn't actually *understand* text.

The goal of this level is to **know the exact size of the problem** before learning
any techniques. If you've ever thought "can't you just type a word into the search
box?", watching plain string matching collapse — where and how — will make it
obvious why you need the tools in the eleven levels that follow.

## 2. Grasping It Through an Analogy

To a computer, language is like **asking someone who doesn't speak a word of English
to run your customer-support desk, armed with nothing but a dictionary**.

This new hire is flawless at dictionary lookup (string matching). Whether the word
"refund" appears in a document — found in 0.001 seconds. And yet:

- When a customer writes "I want my money back", they find nothing. The dictionary
  has no link saying 'refund' and 'money back' mean the same thing. (**the synonym problem**)
- They can't tell whether "bank" means the place that holds your savings or the
  grassy edge of a river. They don't even know that the same letters can carry
  several meanings. (**the ambiguity problem**)
- Faced with "Wow, shipping was fast — never ordering again", they see "fast" and
  file it as praise. Context flips the meaning, and they miss it. (**the context problem**)
- Words like "rizz" or "bussin'" simply aren't in the dictionary. (**the new-word problem**)

NLP (Natural Language Processing), which we study from here on, is the process of
teaching this new hire something beyond dictionary lookup: the statistical
*usage* of words.

## 3. Core Concepts

### 3-1. Ambiguity — same letters, different meanings

The English word "bank" is both a financial institution and the edge of a river.
A "check" is something you deposit and something you do to a fact. A human
disambiguates the instant they see the sentence, but to string matching, the two
"bank"s are exactly the same sequence of bytes.

| Sentence | Human reading | String matching's reading |
|---|---|---|
| The fisherman sat on the bank | bank (riverside) | "bank" found (can't tell) |
| The bank approved my loan | bank (financial) | "bank" found (can't tell) |

The clue that separates the meanings is not in the letters but in the
**surrounding words (context)**. Next to "fisherman" it's the riverside; next to
"loan" it's the financial kind. This intuition — *a word's meaning is decided by its
neighbors* — becomes the theoretical root (the distributional hypothesis) of word
embeddings in level06.

### 3-2. Context and sarcasm — the whole is not the sum of the parts

Containing the word "great" does not make a review positive.

- "The quality is great" → positive
- "Everyone said it was great, but I'm so disappointed" → negative
- "Not that great" → negative (just one extra word)

Counting words one by one (the BoW of level04) hits its limit exactly here, and
models that look at order and context (level07 RNN, level08 attention) are what
fill the gap.

### 3-3. Words keep changing shape — and some languages take it to the extreme

Even in English, where spaces separate words fairly cleanly, one concept hides
behind many strings: *ship, ships, shipped, shipping, shipment*. Search for the
exact token "ship" and you miss "shipped"; search for the substring and you catch
"member**ship**" and "relation**ship**" by accident.

Now consider Korean, an **agglutinative language**, where particles and endings
glue onto the stem and the surface form changes constantly. The single verb
meaning "to eat" shows up in real sentences as
*meogeotda, meokneunda, meogeosseumnida, meokgo, meogeoseo, meogeunikka...* —
dozens of distinct strings for one word. Nouns too: the Korean word for
"delivery" appears with four different particles as four different strings.
English inflection is a gentle slope; agglutinative languages are a cliff.
Either way, the cure is the same family of tools: tokenization, stemming, and
morphological analysis (level03).

### 3-4. Language keeps being born

A 2015 dictionary contained neither "ghosting" nor "doomscrolling". Whatever rule
book you write, language changes faster than the rule book. That is why modern NLP
turned away from hand-written rules toward **learning patterns from data** (the
machine-learning philosophy from lecture06), and invented subword methods
(level03) that handle even unknown words by breaking them into pieces.

## 4. Hands-On — main.py

Run:

```bash
python3 main.py
```

main.py builds a "plain string-matching support bot" and then deliberately breaks it.

- **[1] Synonym failure**: the "refund" keyword search missing "I want my money
  back". It prints the hit rate over 6 genuine refund requests (2 of 6, 33%).
- **[2] Ambiguity failure**: 6 sentences containing "bank" — string matching treats
  them all identically, but neighboring-word hints split money/river correctly (6/6).
- **[3] The betrayal of sentiment keywords**: a rule that calls a review positive on
  "great/best/satisfied" misclassifies negative sentences ("said it was great...
  disappointed") — a confusion table shows 3 of 6 wrong.
- **[4] The word-form experiment**: exact-token search for "ship" vs substring
  search — how they treat inflected forms ("shipping", "shipped") differently, and
  how even substring search drags in impostors like "membership".

The core code is the one-line function `keyword_match(text, keywords)`. How far
this simple function goes and where it collapses — that is this whole level. The
final output [5] prints a roadmap of which level in this lecture solves each failure.

## 5. Try It Yourself

1. **(easy)** Add 2 sentences with the word "check" (bank check vs. checking a fact)
   to the `AMBIGUOUS_SENTENCES` list, fill in the hint dictionary
   (`CONTEXT_HINTS`), and see whether the senses separate.
   Hint: in "I deposited the check", the clue is 'deposited'.
2. **(medium)** Add a "flip if a negation word is nearby" rule to the positive-keyword
   rule in [3] (e.g. negative if 'not', 'but', or 'disappointed' appears in the same
   sentence). Does accuracy improve? Which sentences still slip through?
3. **(challenge)** Write 3 lines, in your own words, on why no amount of added rules
   ever reaches 100%. Hint: every rule you add creates its own new exceptions —
   this is the essential limit of rule-based approaches.

## 6. Common Mistakes

- **"But search engines work fine?"** — Google and your intranet search work because
  morphological analysis, synonym dictionaries, and embeddings are running behind
  the curtain. It is not pure string matching.
- **Treating substring matching (`in`) as a silver bullet** — "ship" also hits
  "membership" and "relationship". Cast the net wide and junk swims in; cast it
  narrow and you miss what you need.
- **Forgetting case and whitespace** — "RMA request", "rma request", and
  "R.M.A. request" are all different strings. The next level (preprocessing)
  tackles this head-on.
- **Assuming every language is equally easy** — English gets surprisingly far on
  whitespace splitting; agglutinative languages like Korean lose serious accuracy
  if you skip particle/ending handling. Tools must fit the language.

## Next Level Preview

Now that we know the problem, we start with the cleaning. In level01 we take real
review text littered with typos and stray symbols and scrub it through a
step-by-step pipeline. You will experience in your own hands why people say 80%
of practical NLP is preprocessing.
