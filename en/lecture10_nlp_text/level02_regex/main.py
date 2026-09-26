"""
level02 — Regular Expressions

Uses the re module to extract phone numbers, emails, and amounts from
fictional internal documents, then practices personal-data masking and
the greedy-matching trap. Standard library only.
"""

import re

# ---------------------------------------------------------------------------
# Practice documents (fictional text written just for this lecture)
# ---------------------------------------------------------------------------

DOCUMENTS = {
    "po_confirmation_email.txt": (
        "Hello, this is Mark Bennett in General Affairs, confirming the "
        "September purchase order: 20 office chairs, total KRW 1,200,000. "
        "Please send the tax invoice to mark.bennett@example.com. "
        "If it's urgent, call 212-555-0134 or (917) 555-0188."
    ),
    "vendor_contacts.txt": (
        "Hanbit Logistics — Sarah Kim 917-555-4321 / sarah@hanbit-logi.com, "
        "Nuri Print main line 646-555-0912, quotes via quote@nuri-print.com. "
        "For after-hours dispatch emergencies, text 8005550123."
    ),
    "expense_notice.txt": (
        "Q3 event expense settlement: booth rental 3.5 million won, plus 300 "
        "promo tote bags at 9900 won each (KRW 2,970,000 total). For missing "
        "receipts contact Finance at fin.help@example.com (ext. 212-555-9999)."
    ),
}

# Named pattern collection — kept in one place for easy maintenance.
PATTERNS = {
    "phone": r"\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}",
    "email": r"[\w.]+@[\w.-]+\.[a-z]{2,}",
    "amount": r"KRW\s?\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?\s?million won|\d+\s?won",
}


def demo_basics() -> None:
    """[1] Warming up the building blocks — what / how many / where"""
    print("[1] Syntax warm-up: what each pattern piece catches")
    sample = "Order no. A-2093, qty 15 units, owner Mark Bennett, memo: ships Sept 26"
    drills = [
        (r"\d+", "runs of digits"),
        (r"[A-Za-z]+", "runs of letters"),
        (r"[A-Z]-\d{4}", "capital-dash-4digits (order-no shape)"),
        (r"\d+ units", "digits + 'units' (a quantity)"),
    ]
    print(f"    target: {sample}")
    for pattern, meaning in drills:
        found = re.findall(pattern, sample)
        print(f"    {pattern:12s} ({meaning:34s}) -> {found}")
    print()


def demo_extraction() -> None:
    """[2] Bulk-extract phones, emails, amounts from all 3 documents"""
    print("[2] Information extraction: sweeping contacts and amounts out of a document pile")
    for name, text in DOCUMENTS.items():
        print(f"    -- {name}")
        for label, pattern in PATTERNS.items():
            found = re.findall(pattern, text)
            print(f"       {label:6s}: {found}")
    print("    -> What used to be a highlighter job is now three findall calls.\n")


def mask_phone(text: str) -> str:
    """Mask the middle digits of phone numbers with ****. Uses group refs \\1, \\3."""
    return re.sub(r"(\(?\d{3}\)?[- ]?)(\d{3})([- ]?\d{4})", r"\1****\3", text)


def demo_masking() -> None:
    """[3] Personal-data masking — re.sub with group references"""
    print("[3] Personal-data masking: a must before any material goes public")
    original = DOCUMENTS["vendor_contacts.txt"]
    masked = mask_phone(original)
    print(f"    original: {original[:60]}...")
    print(f"    masked  : {masked[:60]}...")
    n = len(re.findall(r"\*{4}", masked))
    print(f"    -> The middle digits of {n} phone numbers became ****.\n")


def demo_greedy() -> None:
    """[4] Reproducing a greedy-matching accident — .* vs .*?"""
    print("[4] The greedy trap: by default, the star grabs 'as much as possible'")
    text = "Attendees: <Mark Bennett> <Sarah Kim> <David Cole>"
    greedy = re.findall(r"<.*>", text)
    lazy = re.findall(r"<.*?>", text)
    print(f"    target         : {text}")
    print(f"    <.*>  (greedy) : {greedy}   <- one giant chunk!")
    print(f"    <.*?> (lazy)   : {lazy}")
    # Replacement accident: trying to delete tags deletes the names too
    broken = re.sub(r"<.*>", "", text)
    fixed = re.sub(r"<.*?>", "", text)
    print(f"    delete tags (greedy): {broken!r}  <- data evaporates in one bite")
    print(f"    delete tags (lazy)  : {fixed!r}")
    print("    -> If a replacement mysteriously swallowed half your text, "
          "it's almost always greedy matching.\n")


def demo_summary_table() -> None:
    """[5] Extraction results as a CSV-style summary — the real-world deliverable"""
    print("[5] Wrap-up: per-document extraction summary (paste straight into a spreadsheet)")
    print("    document,phones,emails,amounts")
    for name, text in DOCUMENTS.items():
        counts = [len(re.findall(p, text)) for p in PATTERNS.values()]
        print(f"    {name},{counts[0]},{counts[1]},{counts[2]}")
    print("\nBottom line: regex finds things by their 'shape'. Finding by meaning starts next level.")


if __name__ == "__main__":
    print("=" * 70)
    print("Regular expressions — automatic information extraction from business documents")
    print("=" * 70 + "\n")
    demo_basics()
    demo_extraction()
    demo_masking()
    demo_greedy()
    demo_summary_table()
