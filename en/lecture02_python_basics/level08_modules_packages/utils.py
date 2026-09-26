"""Our team's shared utility module.

Procedures used repeatedly across scripts (currency formatting, VAT,
business-day arithmetic) collected in one file — the 'procedures binder'.
main.py pulls it in with `import utils`.
"""

import datetime

VAT_RATE = 0.1          # module-level constant: importers can access it as utils.VAT_RATE

# The moment a module is imported, its code runs top to bottom, ONCE.
# This print is an educational marker so you can see that with your own eyes.
print("  (utils module loaded — this line runs exactly once, at import)")


def format_krw(amount):
    """Format an amount as a Korean-won string like '1,234,567 KRW'."""
    return f"{amount:,} KRW"


def calc_vat(amount):
    """Compute VAT (10%) as a whole-KRW integer."""
    return int(amount * VAT_RATE)


def add_business_days(start_date, days):
    """Return the date `days` business days later, skipping weekends (Sat/Sun)."""
    current = start_date
    remaining = days
    while remaining > 0:
        current += datetime.timedelta(days=1)
        if current.weekday() < 5:        # 0=Mon ... 4=Fri / skip 5=Sat, 6=Sun
            remaining -= 1
    return current


if __name__ == "__main__":
    # This block runs only when 'executed directly' via `python3 utils.py`.
    # It does NOT run when main.py imports the module — definition vs execution.
    print("[utils self-demo] output visible only when run directly")
    print(f"  format_krw(1234567) = {format_krw(1234567)}")
    print(f"  calc_vat(50000)     = {calc_vat(50000)}")
    demo_day = datetime.date(2026, 9, 25)   # a Friday
    print(f"  {demo_day} (Fri) + 3 business days = {add_business_days(demo_day, 3)}")
