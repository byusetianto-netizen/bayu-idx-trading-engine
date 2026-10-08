from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "fundamental" / "financial_statements_pit.csv"
OUT = ROOT / "data" / "fundamental" / "pit_reconciliation_validation.json"

if not INPUT.exists():
    raise FileNotFoundError(f"Missing input: {INPUT}")

df = pd.read_csv(INPUT, dtype=str).fillna("")

required = {"ticker", "period_end", "publication_date"}
missing = sorted(required - set(df.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

period = pd.to_datetime(df["period_end"], errors="coerce", utc=True)
pub = pd.to_datetime(df["publication_date"], errors="coerce", utc=True)

publication_missing = int(pub.isna().sum())
period_missing = int(period.isna().sum())

# IMPORTANT:
# Publication date is expected to be AFTER period_end for financial reports.
# Therefore publication_date > period_end is NOT an error and is NOT a PIT failure.
# PIT readiness here means the availability evidence exists and is valid.
duplicates = (
    int(df.duplicated(subset=["ticker", "period_end", "metric"], keep=False).sum())
    if "metric" in df.columns else 0
)

pit_ready_mask = pub.notna() & period.notna()
pit_ready_rows = int(pit_ready_mask.sum())
pit_ready_tickers = int(df.loc[pit_ready_mask, "ticker"].nunique())

status = "PASS" if (
    publication_missing == 0
    and period_missing == 0
    and duplicates == 0
) else "REVIEW"

result = {
    "status": status,
    "engine_changed": False,
    "financial_rows": int(len(df)),
    "tickers": int(df["ticker"].nunique()),
    "pit_ready_rows": pit_ready_rows,
    "pit_ready_tickers": pit_ready_tickers,
    "publication_missing": publication_missing,
    "period_end_missing": period_missing,
    "duplicate_key_rows": duplicates,
    "publication_after_period_end": int(
        (pub.notna() & period.notna() & (pub > period)).sum()
    ),
    "publication_after_period_end_is_expected": True,
    "timezone_normalization": "UTC-aware comparison",
    "notes": [
        "Publication date is availability evidence and is expected to occur after period_end for financial reports.",
        "Publication date is never required to be <= period_end.",
        "PIT readiness requires valid period_end and publication evidence.",
        "For a historical analysis date, the separate PIT rule is publication_date <= analysis_date.",
        "No investment decision is performed by this validation layer."
    ]
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
