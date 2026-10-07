#!/usr/bin/env python3
"""
V1.3B-3 OHLCV Integrity Audit

Reads data/idx_stock_prices_expansion.csv and checks row-level OHLCV integrity.
Does NOT modify idx_stock_prices.csv or the expansion dataset.

Outputs:
- data/ohlcv_integrity_audit.csv
- data/ohlcv_integrity_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def find_col(df, candidates):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--min-rows-a", type=int, default=500)
    ap.add_argument("--max-gap-days", type=int, default=45)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    src = data_dir / "idx_stock_prices_expansion.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}")

    df = pd.read_csv(src)
    original_rows = len(df)

    # Normalize expected names without changing the source file.
    colmap = {}
    for target, candidates in {
        "ticker": ["ticker", "symbol"],
        "date": ["date", "datetime"],
        "open": ["open"],
        "high": ["high"],
        "low": ["low"],
        "close": ["close", "adj close", "adj_close"],
        "volume": ["volume"],
    }.items():
        c = find_col(df, candidates)
        if c is None:
            raise ValueError(f"Required column not found for {target}. Columns: {list(df.columns)}")
        colmap[target] = c

    work = pd.DataFrame({
        "ticker": df[colmap["ticker"]].astype(str).str.upper().str.strip(),
        "date": pd.to_datetime(df[colmap["date"]], errors="coerce"),
        "open": pd.to_numeric(df[colmap["open"]], errors="coerce"),
        "high": pd.to_numeric(df[colmap["high"]], errors="coerce"),
        "low": pd.to_numeric(df[colmap["low"]], errors="coerce"),
        "close": pd.to_numeric(df[colmap["close"]], errors="coerce"),
        "volume": pd.to_numeric(df[colmap["volume"]], errors="coerce"),
    })

    # Row-level integrity flags.
    work["bad_date"] = work["date"].isna()
    work["missing_ohlcv"] = work[["open","high","low","close","volume"]].isna().any(axis=1)
    work["nonpositive_price"] = work[["open","high","low","close"]].le(0).any(axis=1)
    work["negative_volume"] = work["volume"].lt(0)
    work["high_below_low"] = work["high"] < work["low"]
    work["high_below_open"] = work["high"] < work["open"]
    work["high_below_close"] = work["high"] < work["close"]
    work["low_above_open"] = work["low"] > work["open"]
    work["low_above_close"] = work["low"] > work["close"]

    price_cols = ["open","high","low","close"]
    work["finite_price"] = np.isfinite(work[price_cols]).all(axis=1)
    work["finite_volume"] = np.isfinite(work["volume"])

    # A duplicate is any repeated ticker/date. We separately report exact
    # duplicates and conflicting duplicates.
    work["_row_order"] = np.arange(len(work))
    dup_key = work.duplicated(["ticker","date"], keep=False)
    work["duplicate_ticker_date"] = dup_key
    exact_dup = work.duplicated(keep=False)

    # Sort for gap calculations.
    work = work.sort_values(["ticker", "date", "_row_order"])
    work["gap_days"] = work.groupby("ticker")["date"].diff().dt.days
    work["gap_gt_threshold"] = work["gap_days"] > args.max_gap_days

    # Per-ticker summary.
    rows = []
    for ticker, g in work.groupby("ticker", sort=True):
        n = len(g)
        bad_logic = (
            g["high_below_low"] |
            g["high_below_open"] |
            g["high_below_close"] |
            g["low_above_open"] |
            g["low_above_close"]
        )
        hard = (
            g["bad_date"] |
            g["missing_ohlcv"] |
            g["nonpositive_price"] |
            g["negative_volume"] |
            (~g["finite_price"]) |
            (~g["finite_volume"]) |
            bad_logic
        )

        dup_count = int(g["duplicate_ticker_date"].sum())
        exact_dup_count = int(g.drop(columns=["_row_order"]).duplicated(keep=False).sum())

        max_gap = g["gap_days"].max()
        gap_count = int(g["gap_gt_threshold"].sum())

        hard_count = int(hard.sum())
        issue_count = int(
            g["missing_ohlcv"].sum()
            + g["nonpositive_price"].sum()
            + g["negative_volume"].sum()
            + bad_logic.sum()
            + g["bad_date"].sum()
        )

        # Integrity grade is independent from the V1.3B-2 coverage grade.
        # A: no hard issues, no large gap, sufficient rows.
        # B: no hard issues but a reviewable gap / short history / duplicates.
        # C: any hard integrity issue or no usable observations.
        if n == 0 or hard_count > 0:
            grade = "C"
        elif n < args.min_rows_a or gap_count > 0 or dup_count > 0:
            grade = "B"
        else:
            grade = "A"

        rows.append({
            "ticker": ticker,
            "rows": n,
            "first_date": g["date"].min().date().isoformat() if g["date"].notna().any() else "",
            "last_date": g["date"].max().date().isoformat() if g["date"].notna().any() else "",
            "missing_ohlcv_rows": int(g["missing_ohlcv"].sum()),
            "bad_date_rows": int(g["bad_date"].sum()),
            "nonpositive_price_rows": int(g["nonpositive_price"].sum()),
            "negative_volume_rows": int(g["negative_volume"].sum()),
            "ohlc_logic_error_rows": int(bad_logic.sum()),
            "duplicate_ticker_date_rows": dup_count,
            "exact_duplicate_rows": exact_dup_count,
            "max_gap_days": float(max_gap) if pd.notna(max_gap) else None,
            "gaps_over_threshold": gap_count,
            "hard_issue_rows": hard_count,
            "issue_rows_total": issue_count,
            "integrity_grade": grade,
        })

    audit = pd.DataFrame(rows)

    # Join the prior coverage audit if available, but never overwrite it.
    prior = data_dir / "price_expansion_audit.csv"
    if prior.exists():
        try:
            p = pd.read_csv(prior)
            if "ticker" in p.columns:
                keep = [c for c in ["ticker","status","rows","first_date","last_date","lag_days","accepted","error"] if c in p.columns]
                p = p[keep].copy()
                p = p.rename(columns={c: f"coverage_{c}" for c in keep if c != "ticker"})
                audit = audit.merge(p, on="ticker", how="left")
        except Exception as e:
            audit["coverage_audit_join_error"] = str(e)

    summary = {
        "version": "V1.3B-3",
        "status": "AUDIT_COMPLETE__MAIN_DATASET_UNCHANGED",
        "source_dataset": str(src),
        "source_rows": int(original_rows),
        "unique_tickers": int(audit["ticker"].nunique()),
        "integrity_grade_A": int((audit["integrity_grade"] == "A").sum()),
        "integrity_grade_B": int((audit["integrity_grade"] == "B").sum()),
        "integrity_grade_C": int((audit["integrity_grade"] == "C").sum()),
        "tickers_with_hard_issues": int((audit["hard_issue_rows"] > 0).sum()),
        "tickers_with_large_gaps": int((audit["gaps_over_threshold"] > 0).sum()),
        "tickers_with_duplicates": int((audit["duplicate_ticker_date_rows"] > 0).sum()),
        "max_gap_threshold_days": int(args.max_gap_days),
        "min_rows_for_grade_A": int(args.min_rows_a),
        "outputs": [
            "data/ohlcv_integrity_audit.csv",
            "data/ohlcv_integrity_summary.json",
        ],
        "important": [
            "Integrity grade is separate from V1.3B-2 coverage grade.",
            "No trading-engine dataset was replaced.",
            "Passing integrity does not solve point-in-time universe membership or survivorship bias.",
        ],
    }

    out_csv = data_dir / "ohlcv_integrity_audit.csv"
    out_json = data_dir / "ohlcv_integrity_summary.json"
    audit.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Expansion rows: {original_rows}")
    print(f"Unique tickers: {audit['ticker'].nunique()}")
    print(f"Integrity Grade A: {summary['integrity_grade_A']}")
    print(f"Integrity Grade B: {summary['integrity_grade_B']}")
    print(f"Integrity Grade C: {summary['integrity_grade_C']}")
    print(f"Tickers with hard issues: {summary['tickers_with_hard_issues']}")
    print(f"Tickers with gaps > {args.max_gap_days} days: {summary['tickers_with_large_gaps']}")
    print(f"Tickers with duplicate ticker/date: {summary['tickers_with_duplicates']}")
    print("STATUS: AUDIT_COMPLETE__MAIN_DATASET_UNCHANGED")


if __name__ == "__main__":
    main()
