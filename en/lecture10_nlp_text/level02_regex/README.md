# Lecture 10 · Level 02 — Regular Expressions

> Learn the search condition language that auto-extracts phone numbers, emails, and amounts from a pile of documents with a single pattern.
**Difficulty** ⭐⭐ / **Prerequisites** level01 / **Estimated time** 40 min

## 1. Why Learn This — The Business View

"Pull the contract amount and the contact person's phone number out of these 30
contracts and put them in a spreadsheet." Done by hand, that's half a day. With
regular expressions (regex), it's five minutes. A regex expresses "things shaped
like a phone number" or "things shaped like an amount" as a one-line pattern and
lets the computer do the finding.

Regex shows up all over the NLP pipeline too. The URL removal in level01 was
already a regex, and it powers personal-data masking (turning a phone number into
212-****-0134), date standardization, and log parsing — anywhere text lives.
If your spreadsheet's Find & Replace is a flashlight, regex is a searchlight.

## 2. Grasping It Through an Analogy

A regular expression is like a **real-estate search filter**. Just as "near the
station AND 2+ bedrooms AND rent under $1,800" filters thousands of listings down
to the matches, a regex sets the condition "3 digits - 3 digits - 4 digits" and
filters a text down to just the phone numbers.

A filter is built from only three kinds of parts:

- **What**: which characters to find — `\d` (a digit), `[A-Za-z]` (a letter), `.` (any character)
- **How many**: how often it repeats — `+` (1 or more), `*` (0 or more), `{3}` (exactly 3)
- **Where**: position constraints — `^` (line start), `$` (line end), `\b` (word boundary)

That combination is the whole language. Even intimidating patterns, taken apart,
are just a sequence of "what · how many · where".

## 3. Core Concepts

### 3-1. Mini syntax table

| Pattern | Meaning | Example |
|---|---|---|
| `\d` | one digit | `\d\d\d` = 3 digits |
| `\d{2,3}` | 2–3 digits | 21 or 212 |
| `[A-Za-z]+` | 1+ letters | "Bennett" |
| `a\|b` | a or b | `won\|dollars` |
| `?` | optional | `-?` = hyphen may be absent |
| `( )` | grouping / partial capture | just the digits in `(\d+) won` |
| `\.` | a literal period | `.` is special, so it needs `\` |

### 3-2. The big three business patterns — phone, email, amount

**Phone number**: `\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}`
— area code with optional parentheses, separators optional, so "212-555-0134",
"(917) 555-0188", and "8005550123" all match.

**Email**: `[\w.]+@[\w.-]+\.[a-z]{2,}`
— letters/dots before the @, then a domain and a top-level domain (2+ letters).
(Note: a regex covering 100% of the international email standard is absurdly
long; in practice this "good enough" pattern is what people use.)

**Amount**: `KRW\s?\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?\s?million won|\d+\s?won`
— joins several notations with `|`: "KRW 1,200,000", "3.5 million won",
"9900 won". Money notation varies (these documents settle in Korean won, KRW),
so the pattern needs several branches.

### 3-3. The four verbs of Python's re module

| Function | What it does | When |
|---|---|---|
| `re.search(p, s)` | find the first match | existence check |
| `re.findall(p, s)` | list of all matches | bulk extraction |
| `re.sub(p, r, s)` | replace matches | masking, cleanup |
| `re.compile(p)` | prepare a reusable pattern | repeated use |

With groups `( )`, `findall` returns only the inside of the group. Handy for
"match the whole thing, extract just a part" — and if you need a group but not
its capture, use `(?: )` (a non-capturing group).

### 3-4. Greedy matching — regex's biggest trap

`*` and `+` grab **as much as possible** by default. In `"<Mark> <Sarah>"`,
the pattern `<.*>` matches not `<Mark>` but the entire string
`<Mark> <Sarah>`. If you want "as short as possible", append `?`, as in
`<.*?>` (lazy matching). When a replacement mysteriously deletes text in
huge chunks, greedy matching is the culprit nine times out of ten.

## 4. Hands-On — main.py

Run:

```bash
python3 main.py
```

The material is three fictional internal documents (a PO confirmation email, a
vendor contact sheet, an expense notice — creative text written inside the code),
and there are four exercises.

- **[1] Syntax warm-up**: check each basic part — `\d+`, `[A-Za-z]+`, `{n}`
  repetition — on a short example, printing what each pattern caught.
- **[2] Information extraction**: `findall` bulk-extracts phones, emails, and
  amounts from the 3 documents, organized per document. The summary in [5]
  comes out as `po_confirmation_email.txt: 2 phones, 1 email, 1 amount /
  vendor_contacts.txt: 3, 2, 0 / expense_notice.txt: 1, 1, 3`.
- **[3] Personal-data masking**: `re.sub` with group references turns the middle
  digits of 3 phone numbers into `****` (e.g. `917-****-4321`). A pattern you
  will use verbatim before publishing any material.
- **[4] Greedy-matching accident**: comparing `<.*>` with `<.*?>` shows greedy
  matching swallowing all three attendee names in one bite.

The part of the code worth studying is the `PATTERNS` dictionary. Keeping named
patterns in one place makes maintenance like "fix only the amount pattern" easy.

## 5. Try It Yourself

1. **(easy)** Add a date pattern to `PATTERNS` that catches both `2026-09-26` and
   `2026.9.26`. Hint: make the separator `[-.]` and the month/day digits
   `\d{1,2}` and one pattern covers both notations.
2. **(medium)** Build email masking: `mark.bennett@example.com` →
   `m****@example.com`. Hint: split groups like `([\w.])[\w.]*(@...)` and use
   `\1****\2` in the replacement.
3. **(challenge)** Write a function `to_won(text)` that converts
   "3.5 million won", "KRW 1,200,000", and "9900 won" all into integer won.
   Hint: multiply by 1,000,000 when "million" appears; remove commas with
   `replace(",", "")`.

## 6. Common Mistakes

- **Trying to match `.` `+` `?` literally and failing**: escape special
  characters like `\.`, or use `re.escape()`.
- **Leaving greedy matching alone**: replacing with `".*"` and losing half a
  sentence. To match short, use `.*?`.
- **Over-engineering the pattern**: trying to fit every phone format on Earth
  into one pattern produces a pattern nobody can read. Cover "the formats that
  actually occur in our data", log the exceptions, and add them later.
- **Assuming `[A-Za-z]` covers all letters**: it misses accented characters
  ("café", "naïve") and other scripts entirely — Korean needs an explicit range
  like `[가-힣]`. For multilingual text, reach for `\w` or explicit Unicode ranges.

## Next Level Preview

Regex finds by "shape", so it knows nothing of meaning. In level03 we learn
tokenization — splitting a sentence into meaningful units. We compare strategies
for the messy bits of English (contractions, hyphens, casing), see why
agglutinative languages like Korean are a harder beast, and taste the subword
idea used by models like GPT.
