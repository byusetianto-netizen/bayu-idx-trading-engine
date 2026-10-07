#!/usr/bin/env python3
"""
V1.3B-4 Session Coverage Diagnostic

Diagnoses why long-history tickers can show session_coverage > 1.0
when compared with the IHSG calendar.

This is diagnostic only:
- does not modify idx_stock_prices.csv
- does not modify idx_stock_prices_expansion.csv
- does not change trading-engine eligibility

Outputs:
- data/session_coverage_diagnostic.csv
- data/session_coverage_summary.json
- data/session_coverage_examples.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def norm_date(s):
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def read_stock(path):
    df = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    tcol = cols.get("ticker") or cols.get("symbol")
    dcol = cols.get("date") or cols.get("datetime")
    if not tcol or not dcol:
        raise ValueError(f"Stock file needs ticker/date columns. Found: {list(df.columns)}")
    out = pd.DataFrame({
        "ticker": df[tcol].astype(str).str.upper().str.strip(),
        "date": norm_date(df[dcol]),
    }).dropna().drop_duplicates(["ticker", "date"])
    return out


def read_ihsg(path):
    df = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    dcol = cols.get("date") or cols.get("price")
    if not dcol:
        raise ValueError(f"IHSG file needs date/Price column. Found: {list(df.columns)}")
    return norm_date(df[dcol]).dropna().drop_duplicates().sort_values().reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--top-examples", type=int, default=30)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    stock_path = data_dir / "idx_stock_prices_expansion.csv"
    ihsg_path = data_dir / "idx_ihsg_index.csv"

    stock = read_stock(stock_path)
    ihsg = read_ihsg(ihsg_path)
    ihsg_set = set(ihsg)

    rows = []
    example_rows = []

    for ticker, g in stock.groupby("ticker", sort=True):
        dates = set(g["date"])
        common = dates & ihsg_set
        stock_only = sorted(dates - ihsg_set)
        ihsg_only_in_range = [d for d in ihsg if g["date"].min() <= d <= g["date"].max() and d not in dates]

        first = g["date"].min()
        last = g["date"].max()
        ihsg_in_range = [d for d in ihsg if first <= d <= last]

        stock_sessions = len(dates)
        ihsg_sessions = len(ihsg_in_range)
        common_sessions = len(common)
        stock_only_count = len(stock_only)
        ihsg_only_count = len(ihsg_only_in_range)

        coverage = (stock_sessions / ihsg_sessions) if ihsg_sessions else None

        rows.append({
            "ticker": ticker,
            "stock_sessions": stock_sessions,
            "ihsg_sessions_same_range": ihsg_sessions,
            "common_sessions": common_sessions,
            "stock_only_sessions": stock_only_count,
            "ihsg_only_sessions": ihsg_only_count,
            "stock_minus_ihsg": stock_sessions - ihsg_sessions,
            "coverage_ratio": coverage,
            "first_date": first.date().isoformat(),
            "last_date": last.date().isoformat(),
        })

    audit = pd.DataFrame(rows)
    audit["coverage_excess_pct"] = (audit["coverage_ratio"] - 1.0) * 100.0
    audit["diagnostic_flag"] = audit.apply(
        lambda r: (
            "STOCK_CALENDAR_DIFFERENCE"
            if r["stock_only_sessions"] > 0 or r["ihsg_only_sessions"] > 0
            else "CALENDAR_MATCH"
        ),
        axis=1,
    )

    # Focus examples where coverage exceeds 1 or there is a calendar mismatch.
    focus = audit.sort_values(
        ["coverage_excess_pct", "stock_only_sessions", "ihsg_only_sessions"],
        ascending=[False, False, False],
    ).head(args.top_examples)

    for _, r in focus.iterrows():
        ticker = r["ticker"]
        g = stock[stock["ticker"] == ticker]
        dates = set(g["date"])
        first = g["date"].min()
        last = g["date"].max()
        ihsg_range = [d for d in ihsg if first <= d <= last]
        stock_only = sorted(dates - set(ihsg_range))
        ihsg_only = [d for d in ihsg_range if d not in dates]

        # Limit date lists to keep CSV compact.
        example_rows.append({
            "ticker": ticker,
            "stock_only_dates_sample": ";".join(d.strftime("%Y-%m-%d") for d in stock_only[:25]),
            "ihsg_only_dates_sample": ";".join(d.strftime("%Y-%m-%d") for d in ihsg_only[:25]),
            "stock_only_count": len(stock_only),
            "ihsg_only_count": len(ihsg_only),
        })

    examples = pd.DataFrame(example_rows)

    summary = {
        "version": "V1.3B-4",
        "status": "DIAGNOSTIC_COMPLETE__DATASETS_UNCHANGED",
        "stock_unique_tickers": int(stock["ticker"].nunique()),
        "ihsg_unique_sessions": int(len(ihsg)),
        "tickers_with_coverage_above_1": int((audit["coverage_ratio"] > 1.0).sum()),
        "tickers_with_stock_only_sessions": int((audit["stock_only_sessions"] > 0).sum()),
        "tickers_with_ihsg_only_sessions": int((audit["ihsg_only_sessions"] > 0).sum()),
        "tickers_calendar_match": int((audit["diagnostic_flag"] == "CALENDAR_MATCH").sum()),
        "mean_coverage_ratio": float(audit["coverage_ratio"].mean()),
        "median_coverage_ratio": float(audit["coverage_ratio"].median()),
        "max_coverage_ratio": float(audit["coverage_ratio"].max()),
        "outputs": [
            "data/session_coverage_diagnostic.csv",
            "data/session_coverage_summary.json",
            "data/session_coverage_examples.csv",
        ],
        "important": [
            "This diagnostic does not declare stock-only dates invalid.",
            "It does not modify any price dataset.",
            "Calendar differences must be understood before using session coverage as a rejection rule.",
            "This does not solve point-in-time universe membership or survivorship bias.",
        ],
    }

    audit.to_csv(data_dir / "session_coverage_diagnostic.csv", index=False)
    examples.to_csv(data_dir / "session_coverage_examples.csv", index=False)
    (data_dir / "session_coverage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Stock unique tickers: {summary['stock_unique_tickers']}")
    print(f"IHSG unique sessions: {summary['ihsg_unique_sessions']}")
    print(f"Tickers coverage > 1.0: {summary['tickers_with_coverage_above_1']}")
    print(f"Tickers with stock-only sessions: {summary['tickers_with_stock_only_sessions']}")
    print(f"Tickers with IHSG-only sessions: {summary['tickers_with_ihsg_only_sessions']}")
    print(f"Calendar match: {summary['tickers_calendar_match']}")
    print(f"Mean coverage ratio: {summary['mean_coverage_ratio']:.6f}")
    print(f"Median coverage ratio: {summary['median_coverage_ratio']:.6f}")
    print(f"Max coverage ratio: {summary['max_coverage_ratio']:.6f}")
    print("STATUS: DIAGNOSTIC_COMPLETE__DATASETS_UNCHANGED")


if __name__ == "__main__":
    main()
