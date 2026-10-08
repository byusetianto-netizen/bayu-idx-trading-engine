#!/usr/bin/env python3
"""V2.2 PIT Diagnostic compatibility fix.

This diagnostic intentionally does not depend on a private helper function
inside valuation_engine.py. It reads the reconciled PIT dataset directly.
It is diagnostic-only and never changes valuation outputs.
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRICE = ROOT / "data" / "idx_stock_prices.csv"
FS_PIT = ROOT / "data" / "fundamental" / "financial_statements_pit.csv"
OUT = ROOT / "data" / "valuation"
OUT.mkdir(parents=True, exist_ok=True)

REQUIRED = [
    "ticker", "metric", "value", "period_end",
    "publication_date"
]


def read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def norm_cols(df):
    if df.empty:
        return df
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def main():
    prices = norm_cols(read_csv(PRICE))
    fs = norm_cols(read_csv(FS_PIT))

    result = {
        "status": "DIAGNOSTIC_COMPLETE",
        "analysis_date": None,
        "financial_source_expected": "data/fundamental/financial_statements_pit.csv",
        "financial_source_exists": bool(FS_PIT.exists()),
        "financial_rows_loaded": int(len(fs)),
        "financial_columns": list(fs.columns),
        "tickers_checked": {},
        "pit_verified_tickers": [],
        "pit_unknown_tickers": [],
        "notes": [
            "Diagnostic reads the V2.1F reconciled PIT dataset directly.",
            "No private helper from valuation_engine.py is required.",
            "PIT readiness requires publication_date <= analysis_date.",
            "period_end is never substituted for publication_date."
        ]
    }

    if prices.empty or "date" not in prices.columns:
        analysis_date = pd.Timestamp.today().normalize()
    else:
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        analysis_date = prices["date"].max()
        if pd.isna(analysis_date):
            analysis_date = pd.Timestamp.today().normalize()

    result["analysis_date"] = analysis_date.isoformat()

    if fs.empty:
        result["status"] = "DIAGNOSTIC_COMPLETE_NO_PIT_ROWS"
        (OUT / "v22_pit_diagnostic.json").write_text(
            json.dumps(result, indent=2, default=str),
            encoding="utf-8"
        )
        print(json.dumps(result, indent=2, default=str))
        return

    missing = [c for c in REQUIRED if c not in fs.columns]
    if missing:
        result["status"] = "DIAGNOSTIC_SCHEMA_ERROR"
        result["missing_required_columns"] = missing
        (OUT / "v22_pit_diagnostic.json").write_text(
            json.dumps(result, indent=2, default=str),
            encoding="utf-8"
        )
        print(json.dumps(result, indent=2, default=str))
        raise SystemExit(1)

    fs["ticker"] = fs["ticker"].astype(str).str.upper().str.strip()
    fs["period_end"] = pd.to_datetime(fs["period_end"], errors="coerce")
    fs["publication_date"] = pd.to_datetime(fs["publication_date"], errors="coerce")

    if "pit_ready" in fs.columns:
        pit_flag = fs["pit_ready"].astype(str).str.strip().str.lower().isin(
            ["true", "1", "yes", "y"]
        )
    else:
        pit_flag = pd.Series(True, index=fs.index)

    available = fs["publication_date"].notna() & (
        fs["publication_date"] <= analysis_date
    )
    eligible = fs[pit_flag & available].copy()

    for ticker, g in fs.groupby("ticker"):
        eg = eligible[eligible["ticker"] == ticker]
        sample_cols = [
            c for c in [
                "ticker", "metric", "period_end", "publication_date",
                "publication_timestamp", "pit_ready", "source"
            ] if c in eg.columns
        ]

        entry = {
            "rows_loaded": int(len(g)),
            "pit_rows_available": int(len(eg)),
            "pit_ready": bool(len(eg) > 0),
            "sample": eg[sample_cols].head(10).to_dict("records")
        }
        result["tickers_checked"][ticker] = entry

        if len(eg) > 0:
            result["pit_verified_tickers"].append(ticker)
        else:
            result["pit_unknown_tickers"].append(ticker)

    result["pit_verified_tickers"].sort()
    result["pit_unknown_tickers"].sort()
    result["pit_verified_rows"] = int(len(eligible))
    result["pit_verified_ticker_count"] = len(result["pit_verified_tickers"])
    result["pit_unknown_ticker_count"] = len(result["pit_unknown_tickers"])

    (OUT / "v22_pit_diagnostic.json").write_text(
        json.dumps(result, indent=2, default=str),
        encoding="utf-8"
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
