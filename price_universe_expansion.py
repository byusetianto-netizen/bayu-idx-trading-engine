"""
V1.3B-1 — Safe historical price expansion for the current BEI universe.

Purpose
-------
Download a recent multi-year historical window for current-universe tickers that
are missing from data/idx_stock_prices.csv.

IMPORTANT:
- This script NEVER overwrites idx_stock_prices.csv.
- It writes a candidate expanded dataset to:
    data/idx_stock_prices_expansion.csv
- It writes per-ticker diagnostics to:
    data/price_expansion_audit.csv
- It only marks a ticker ACCEPTED when it has enough rows and a recent last date.
- This is a research-data expansion step, not proof of historical PIT membership.
- Yahoo/yfinance is a practical research source, not an official IDX feed.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import time

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MASTER = DATA / "security_master.csv"
STOCKS = DATA / "idx_stock_prices.csv"
OUT_STOCKS = DATA / "idx_stock_prices_expansion.csv"
OUT_AUDIT = DATA / "price_expansion_audit.csv"
OUT_STATUS = DATA / "price_expansion_status.json"

DEFAULT_YEARS = 3
MAX_WORKERS = 6
MIN_ROWS = 200
MAX_LAG_DAYS = 10
RETRIES = 3


def clean_ticker(x):
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    return s or None


def load_inputs():
    if not MASTER.exists():
        raise FileNotFoundError("data/security_master.csv not found.")
    if not STOCKS.exists():
        raise FileNotFoundError("data/idx_stock_prices.csv not found.")

    master = pd.read_csv(MASTER)
    stocks = pd.read_csv(STOCKS, parse_dates=["date"])

    if "ticker" not in master.columns:
        raise ValueError("security_master.csv has no ticker column.")
    required = {"ticker", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(stocks.columns)
    if missing:
        raise ValueError(f"idx_stock_prices.csv missing columns: {sorted(missing)}")

    master["ticker"] = master["ticker"].map(clean_ticker)
    stocks["ticker"] = stocks["ticker"].map(clean_ticker)
    stocks["date"] = pd.to_datetime(stocks["date"], errors="coerce")

    stocks = (
        stocks.dropna(subset=["ticker", "date", "close"])
        .drop_duplicates(["ticker", "date"], keep="last")
        .sort_values(["ticker", "date"])
    )

    return master.dropna(subset=["ticker"]).drop_duplicates("ticker"), stocks


def download_one(ticker, start, end, retries=RETRIES):
    try:
        import yfinance as yf
    except Exception as exc:
        return ticker, None, f"yfinance import error: {exc}"

    symbol = f"{ticker}.JK"

    for attempt in range(retries):
        try:
            x = yf.download(
                symbol,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=20,
            )

            if x is None or x.empty:
                raise ValueError("empty response")

            x = x.reset_index()

            if isinstance(x.columns, pd.MultiIndex):
                x.columns = [
                    c[0] if isinstance(c, tuple) else c
                    for c in x.columns
                ]

            x.columns = [
                str(c).lower().replace(" ", "_") for c in x.columns
            ]

            required = ["date", "open", "high", "low", "close", "volume"]
            missing = [c for c in required if c not in x.columns]
            if missing:
                raise ValueError(f"missing columns: {missing}")

            x = x[required].copy()
            x["ticker"] = ticker
            x["date"] = pd.to_datetime(x["date"], errors="coerce")

            try:
                x["date"] = x["date"].dt.tz_localize(None)
            except TypeError:
                pass

            for c in ["open", "high", "low", "close", "volume"]:
                x[c] = pd.to_numeric(x[c], errors="coerce")

            x = (
                x.dropna(subset=["date", "close"])
                .drop_duplicates(["ticker", "date"], keep="last")
                .sort_values("date")
            )

            if x.empty:
                raise ValueError("no valid OHLC rows after cleaning")

            return ticker, x[
                ["date", "ticker", "open", "high", "low", "close", "volume"]
            ], None

        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return ticker, None, str(exc)

    return ticker, None, "unknown error"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--min-rows", type=int, default=MIN_ROWS)
    parser.add_argument("--max-lag-days", type=int, default=MAX_LAG_DAYS)
    args = parser.parse_args()

    master, stocks = load_inputs()

    existing = set(stocks["ticker"].dropna())
    universe = set(master["ticker"].dropna())
    missing = sorted(universe - existing)

    latest_local = pd.Timestamp(stocks["date"].max()).normalize()
    end_date = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None).normalize()
    start_date = end_date - pd.DateOffset(years=int(args.years))

    print(f"Current universe: {len(universe)} tickers")
    print(f"Existing price tickers: {len(existing)}")
    print(f"Missing price tickers to probe: {len(missing)}")
    print(f"Download window: {start_date.date()} -> {end_date.date()}")

    rows = []
    diagnostics = []

    workers = max(1, min(int(args.workers), MAX_WORKERS))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_one,
                ticker,
                str(start_date.date()),
                str((end_date + pd.Timedelta(days=1)).date()),
            ): ticker
            for ticker in missing
        }

        for future in as_completed(futures):
            ticker = futures[future]
            try:
                t, x, error = future.result()
            except Exception as exc:
                t, x, error = ticker, None, str(exc)

            if x is None or x.empty:
                diagnostics.append({
                    "ticker": t,
                    "status": "DOWNLOAD_FAILED",
                    "rows": 0,
                    "first_date": None,
                    "last_date": None,
                    "lag_days": None,
                    "accepted": False,
                    "error": error,
                })
                continue

            first_date = pd.Timestamp(x["date"].min()).normalize()
            last_date = pd.Timestamp(x["date"].max()).normalize()
            lag_days = int((latest_local - last_date).days)

            accepted = (
                len(x) >= int(args.min_rows)
                and lag_days <= int(args.max_lag_days)
            )

            diagnostics.append({
                "ticker": t,
                "status": "ACCEPTED" if accepted else "INSUFFICIENT_COVERAGE",
                "rows": int(len(x)),
                "first_date": str(first_date.date()),
                "last_date": str(last_date.date()),
                "lag_days": lag_days,
                "accepted": bool(accepted),
                "error": None,
            })

            if accepted:
                rows.append(x)

    audit = pd.DataFrame(diagnostics).sort_values("ticker")
    audit.to_csv(OUT_AUDIT, index=False)

    accepted_data = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])
    )

    # Candidate expanded dataset only. Existing 95-ticker dataset remains untouched.
    expanded = pd.concat([stocks, accepted_data], ignore_index=True)
    expanded["date"] = pd.to_datetime(expanded["date"], errors="coerce")
    expanded = (
        expanded.dropna(subset=["ticker", "date", "close"])
        .drop_duplicates(["ticker", "date"], keep="last")
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    expanded.to_csv(OUT_STOCKS, index=False)

    accepted = audit[audit["accepted"] == True]
    failed = audit[audit["accepted"] == False]

    status = {
        "version": "V1.3B-1",
        "status": "PROBE_COMPLETE__MAIN_DATASET_UNCHANGED",
        "universe_tickers": int(len(universe)),
        "existing_price_tickers": int(len(existing)),
        "missing_price_tickers": int(len(missing)),
        "accepted_new_tickers": int(len(accepted)),
        "rejected_or_failed_tickers": int(len(failed)),
        "download_start": str(start_date.date()),
        "download_end": str(end_date.date()),
        "minimum_rows": int(args.min_rows),
        "maximum_allowed_lag_days": int(args.max_lag_days),
        "candidate_rows": int(len(expanded)),
        "candidate_tickers": int(expanded["ticker"].nunique()),
        "output_dataset": str(OUT_STOCKS.relative_to(ROOT)),
        "audit_file": str(OUT_AUDIT.relative_to(ROOT)),
        "important": [
            "idx_stock_prices.csv was NOT overwritten.",
            "This is a current-universe price-coverage expansion, not a point-in-time security master.",
            "Do not feed the candidate dataset into the trading engine until coverage and data-quality review is complete.",
        ],
    }

    OUT_STATUS.write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
