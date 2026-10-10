"""
V1.3F Raw Execution Price Probe

Purpose
-------
Build a small, isolated proof-of-concept for an unadjusted execution-price
layer.

This script:
1. Downloads raw/unadjusted OHLCV using yfinance auto_adjust=False.
2. Downloads corporate-action information when available.
3. Compares raw close with the existing adjusted research dataset.
4. Performs basic OHLCV integrity checks.
5. Writes audit-only outputs.

IMPORTANT:
- Does NOT modify idx_stock_prices.csv.
- Does NOT modify idx_stock_prices_expansion.csv.
- Does NOT modify actual_engine.py.
- Does NOT generate BUY/SELL signals.
- This is NOT yet a production execution dataset.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

ADJUSTED_FILE = DATA / "idx_stock_prices.csv"

OUTPUT_DIR = DATA / "execution_probe"
RAW_OUTPUT = OUTPUT_DIR / "raw_execution_prices_probe.csv"
ACTIONS_OUTPUT = OUTPUT_DIR / "corporate_actions_probe.csv"
COMPARISON_OUTPUT = OUTPUT_DIR / "raw_vs_adjusted_probe.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "raw_execution_probe_summary.json"

TICKERS = ["BBRI", "TLKM", "PTBA"]

START_DATE = "2015-01-01"


def fail(message):
    raise RuntimeError(message)


def flatten_columns(df):
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    # For a single yfinance ticker the useful field normally appears
    # in one of the MultiIndex levels.
    known = {
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Dividends",
        "Stock Splits",
    }

    new_columns = []

    for col in df.columns:
        values = [str(x) for x in col]

        match = next(
            (x for x in values if x in known),
            None,
        )

        new_columns.append(match if match is not None else values[0])

    out = df.copy()
    out.columns = new_columns
    return out


def download_raw(ticker):
    symbol = f"{ticker}.JK"

    df = yf.download(
        symbol,
        start=START_DATE,
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        fail(f"No yfinance data returned for {ticker}")

    df = flatten_columns(df)
    df = df.reset_index()

    date_col = "Date" if "Date" in df.columns else "Datetime"

    required = {
        date_col,
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }

    missing = required - set(df.columns)

    if missing:
        fail(
            f"{ticker}: required raw columns missing: "
            f"{sorted(missing)}"
        )

    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "ticker": ticker,
            "open": pd.to_numeric(df["Open"], errors="coerce"),
            "high": pd.to_numeric(df["High"], errors="coerce"),
            "low": pd.to_numeric(df["Low"], errors="coerce"),
            "close": pd.to_numeric(df["Close"], errors="coerce"),
            "volume": pd.to_numeric(df["Volume"], errors="coerce"),
        }
    )

    raw["date"] = raw["date"].dt.tz_localize(None).dt.normalize()

    actions = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "ticker": ticker,
            "dividends": (
                pd.to_numeric(df["Dividends"], errors="coerce")
                if "Dividends" in df.columns
                else 0.0
            ),
            "stock_splits": (
                pd.to_numeric(df["Stock Splits"], errors="coerce")
                if "Stock Splits" in df.columns
                else 0.0
            ),
        }
    )

    actions["date"] = (
        actions["date"]
        .dt.tz_localize(None)
        .dt.normalize()
    )

    actions["dividends"] = actions["dividends"].fillna(0.0)
    actions["stock_splits"] = actions["stock_splits"].fillna(0.0)

    actions = actions[
        (actions["dividends"] != 0)
        | (actions["stock_splits"] != 0)
    ].copy()

    return raw, actions


def validate_raw(df):
    checks = {}

    checks["rows"] = int(len(df))
    checks["tickers"] = int(df["ticker"].nunique())

    checks["duplicate_ticker_date_rows"] = int(
        df.duplicated(["ticker", "date"]).sum()
    )

    checks["missing_ohlcv_rows"] = int(
        df[
            ["open", "high", "low", "close", "volume"]
        ].isna().any(axis=1).sum()
    )

    checks["nonpositive_price_rows"] = int(
        (
            (df["open"] <= 0)
            | (df["high"] <= 0)
            | (df["low"] <= 0)
            | (df["close"] <= 0)
        ).sum()
    )

    checks["negative_volume_rows"] = int(
        (df["volume"] < 0).sum()
    )

    checks["ohlc_logic_error_rows"] = int(
        (
            (df["high"] < df[["open", "close", "low"]].max(axis=1))
            | (
                df["low"]
                > df[["open", "close", "high"]].min(axis=1)
            )
        ).sum()
    )

    return checks


def load_adjusted():
    if not ADJUSTED_FILE.exists():
        fail(
            f"Adjusted research dataset not found: "
            f"{ADJUSTED_FILE}"
        )

    df = pd.read_csv(ADJUSTED_FILE)

    required = {
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = required - set(df.columns)

    if missing:
        fail(
            "Adjusted dataset missing required columns: "
            f"{sorted(missing)}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return df[df["ticker"].isin(TICKERS)].copy()


def build_comparison(raw, adjusted):
    a = adjusted[
        ["date", "ticker", "close"]
    ].rename(
        columns={
            "close": "adjusted_close"
        }
    )

    r = raw[
        ["date", "ticker", "close"]
    ].rename(
        columns={
            "close": "raw_close"
        }
    )

    merged = r.merge(
        a,
        on=["ticker", "date"],
        how="inner",
        validate="one_to_one",
    )

    merged["raw_to_adjusted_ratio"] = (
        merged["raw_close"]
        / merged["adjusted_close"]
    )

    merged["close_difference_pct"] = (
        merged["raw_close"]
        / merged["adjusted_close"]
        - 1.0
    )

    return merged.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_frames = []
    action_frames = []

    for ticker in TICKERS:
        print(f"Downloading raw execution probe: {ticker}")

        raw, actions = download_raw(ticker)

        raw_frames.append(raw)

        if not actions.empty:
            action_frames.append(actions)

    raw_all = pd.concat(
        raw_frames,
        ignore_index=True,
    )

    raw_all = raw_all.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    if action_frames:
        actions_all = pd.concat(
            action_frames,
            ignore_index=True,
        ).sort_values(
            ["ticker", "date"]
        ).reset_index(drop=True)
    else:
        actions_all = pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "dividends",
                "stock_splits",
            ]
        )

    checks = validate_raw(raw_all)

    hard_issue_count = (
        checks["duplicate_ticker_date_rows"]
        + checks["missing_ohlcv_rows"]
        + checks["nonpositive_price_rows"]
        + checks["negative_volume_rows"]
        + checks["ohlc_logic_error_rows"]
    )

    adjusted = load_adjusted()

    comparison = build_comparison(
        raw_all,
        adjusted,
    )

    raw_all.to_csv(
        RAW_OUTPUT,
        index=False,
        date_format="%Y-%m-%d",
    )

    actions_all.to_csv(
        ACTIONS_OUTPUT,
        index=False,
        date_format="%Y-%m-%d",
    )

    comparison.to_csv(
        COMPARISON_OUTPUT,
        index=False,
        date_format="%Y-%m-%d",
    )

    ticker_summary = {}

    for ticker in TICKERS:
        x = raw_all[
            raw_all["ticker"] == ticker
        ]

        c = comparison[
            comparison["ticker"] == ticker
        ]

        ticker_summary[ticker] = {
            "raw_rows": int(len(x)),
            "first_raw_date": (
                x["date"].min().strftime("%Y-%m-%d")
                if len(x)
                else None
            ),
            "last_raw_date": (
                x["date"].max().strftime("%Y-%m-%d")
                if len(x)
                else None
            ),
            "comparison_rows": int(len(c)),
            "corporate_action_rows": int(
                (
                    actions_all["ticker"] == ticker
                ).sum()
            ),
            "raw_adjusted_ratio_min": (
                float(
                    c["raw_to_adjusted_ratio"].min()
                )
                if len(c)
                else None
            ),
            "raw_adjusted_ratio_max": (
                float(
                    c["raw_to_adjusted_ratio"].max()
                )
                if len(c)
                else None
            ),
            "latest_raw_adjusted_ratio": (
                float(
                    c.iloc[-1][
                        "raw_to_adjusted_ratio"
                    ]
                )
                if len(c)
                else None
            ),
        }

    summary = {
        "version": "V1.3F",
        "status": (
            "PROBE_PASS__NOT_ENABLED_IN_ENGINE"
            if hard_issue_count == 0
            else "PROBE_REVIEW_REQUIRED__NOT_ENABLED_IN_ENGINE"
        ),
        "purpose": (
            "Raw/unadjusted execution-price "
            "proof-of-concept only"
        ),
        "source": "Yahoo Finance via yfinance",
        "yfinance_version": getattr(
            yf,
            "__version__",
            "UNKNOWN",
        ),
        "auto_adjust": False,
        "actions_requested": True,
        "start_date": START_DATE,
        "tickers": TICKERS,
        "raw_integrity": checks,
        "hard_issue_count": int(
            hard_issue_count
        ),
        "corporate_action_rows": int(
            len(actions_all)
        ),
        "comparison_rows": int(
            len(comparison)
        ),
        "ticker_summary": ticker_summary,
        "production_dataset_modified": False,
        "actual_engine_modified": False,
        "pit_safe_for_backtest": False,
        "execution_backtest_ready": False,
        "warning": (
            "Probe output is research evidence only. "
            "It does not establish official IDX historical "
            "prices, PIT universe membership, transaction "
            "cost assumptions, tick-size handling, or "
            "execution-grade backtest validity."
        ),
    }

    SUMMARY_OUTPUT.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    if hard_issue_count > 0:
        print(
            "V1.3F completed with integrity issues. "
            "Review outputs before any promotion.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()