#!/usr/bin/env python3
"""V2.2 PIT Diagnostic.

Reads the V2.1F reconciled PIT dataset directly. Diagnostic-only; never
changes valuation outputs. publication_timestamp is authoritative.
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRICE = ROOT / "data" / "idx_stock_prices.csv"
FS_PIT = ROOT / "data" / "fundamental" / "financial_statements_pit.csv"
OUT = ROOT / "data" / "valuation"
OUT.mkdir(parents=True, exist_ok=True)

REQUIRED = ["ticker", "metric", "value", "period_end", "publication_timestamp", "pit_ready"]

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
        "analysis_timestamp": None,
        "analysis_timezone": "Asia/Jakarta (+07:00)",
        "financial_source_expected": "data/fundamental/financial_statements_pit.csv",
        "financial_source_exists": bool(FS_PIT.exists()),
        "financial_rows_loaded": int(len(fs)),
        "financial_columns": list(fs.columns),
        "tickers_checked": {},
        "pit_verified_tickers": [],
        "pit_unknown_tickers": [],
        "publication_timestamp_missing_rows": 0,
        "publication_timestamp_invalid_rows": 0,
        "future_publication_rows": 0,
        "publication_date_fallback": False,
        "notes": [
            "Diagnostic reads the V2.1F reconciled PIT dataset directly.",
            "publication_timestamp is authoritative; publication_date is metadata only and never a fallback.",
            "PIT availability requires pit_ready=true and publication_timestamp <= analysis_timestamp.",
            "Daily price data supplies only a date, so analysis_timestamp is conservatively set to 23:59:59 WIB on the latest price date.",
            "period_end is never substituted for publication evidence."
        ]
    }

    if prices.empty or "date" not in prices.columns:
        # Fallback is only for the analysis clock, never for publication evidence.
        analysis_date = pd.Timestamp.now(tz="Asia/Jakarta").normalize()
    else:
        price_dates = pd.to_datetime(prices["date"], errors="coerce")
        max_date = price_dates.max()
        if pd.isna(max_date):
            analysis_date = pd.Timestamp.now(tz="Asia/Jakarta").normalize()
        else:
            analysis_date = pd.Timestamp(max_date).normalize().tz_localize("Asia/Jakarta")

    analysis_timestamp = analysis_date + pd.Timedelta(hours=23, minutes=59, seconds=59)
    analysis_timestamp_utc = analysis_timestamp.tz_convert("UTC")
    result["analysis_date"] = analysis_date.strftime("%Y-%m-%d")
    result["analysis_timestamp"] = analysis_timestamp.isoformat()

    if fs.empty:
        result["status"] = "DIAGNOSTIC_COMPLETE_NO_PIT_ROWS"
        (OUT / "v22_pit_diagnostic.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(json.dumps(result, indent=2, default=str))
        return

    missing = [c for c in REQUIRED if c not in fs.columns]
    if missing:
        result["status"] = "DIAGNOSTIC_SCHEMA_ERROR"
        result["missing_required_columns"] = missing
        (OUT / "v22_pit_diagnostic.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(json.dumps(result, indent=2, default=str))
        raise SystemExit(1)

    fs["ticker"] = fs["ticker"].astype(str).str.upper().str.strip()
    fs["period_end"] = pd.to_datetime(fs["period_end"], errors="coerce", utc=True)

    raw_ts = fs["publication_timestamp"].fillna("").astype(str).str.strip()
    fs["_publication_timestamp"] = pd.to_datetime(raw_ts, errors="coerce", utc=True)
    fs["_pit_flag"] = fs["pit_ready"].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])

    missing_ts = raw_ts.eq("")
    invalid_ts = raw_ts.ne("") & fs["_publication_timestamp"].isna()
    future_ts = fs["_publication_timestamp"].notna() & (fs["_publication_timestamp"] > analysis_timestamp_utc)

    result["publication_timestamp_missing_rows"] = int(missing_ts.sum())
    result["publication_timestamp_invalid_rows"] = int(invalid_ts.sum())
    result["future_publication_rows"] = int(future_ts.sum())

    available = fs["_publication_timestamp"].notna() & (fs["_publication_timestamp"] <= analysis_timestamp_utc)
    eligible = fs[fs["_pit_flag"] & available].copy()

    for ticker, g in fs.groupby("ticker"):
        eg = eligible[eligible["ticker"] == ticker]
        sample_cols = [c for c in [
            "ticker", "metric", "period_end", "publication_date",
            "publication_timestamp", "pit_ready", "source"
        ] if c in eg.columns]
        result["tickers_checked"][ticker] = {
            "rows_loaded": int(len(g)),
            "pit_rows_available": int(len(eg)),
            "pit_ready": bool(len(eg) > 0),
            "sample": eg[sample_cols].head(10).to_dict("records")
        }
        if len(eg) > 0:
            result["pit_verified_tickers"].append(ticker)
        else:
            result["pit_unknown_tickers"].append(ticker)

    result["pit_verified_tickers"].sort()
    result["pit_unknown_tickers"].sort()
    result["pit_verified_rows"] = int(len(eligible))
    result["pit_verified_ticker_count"] = len(result["pit_verified_tickers"])
    result["pit_unknown_ticker_count"] = len(result["pit_unknown_tickers"])

    (OUT / "v22_pit_diagnostic.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()
