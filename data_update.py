"""IDX market-data updater V3: daily update + automatic repair mode.

Purpose:
- Prevent a partial yfinance response from being accepted as a complete update.
- Repair recent historical gaps even when the repository already contains the
  latest calendar/trading date.
- Preserve the existing dataset and merge validated new/repair rows.
- Fail safely when coverage is still inadequate.

Research use only; not an official IDX market-data feed.
"""

from pathlib import Path
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
STOCKS = DATA / "idx_stock_prices.csv"
IHSG = DATA / "idx_ihsg_index.csv"
MANIFEST = DATA / "update_manifest.json"

# Repair the most recent ~95 calendar days on every run. This catches cases
# where a previous "successful" update contained only a few rows.
REPAIR_DAYS = 95

# Safety gates.
MIN_LATEST_TICKER_COVERAGE = 0.85
MIN_ROW_COVERAGE = 0.50

# If the repair window is expected to contain ~65 trading sessions and
# ~95 securities, a healthy update should be thousands of rows, not dozens.
MIN_REPAIR_ROWS = 1000

MAX_WORKERS = 6


def normalize_existing():
    df = pd.read_csv(STOCKS, parse_dates=["date"])
    required = ["ticker", "date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Stock CSV missing columns: {missing}")

    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (
        df.dropna(subset=["ticker", "date", "close"])
        .drop_duplicates(["ticker", "date"], keep="last")
        .sort_values(["ticker", "date"])
    )

    df.to_csv(STOCKS, index=False)
    return df


def download_one(ticker, start, end, retries=3):
    """Download one ticker independently so one bad symbol cannot poison the batch."""
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
            )

            if x is None or x.empty:
                raise ValueError("empty response")

            x = x.reset_index()

            # yfinance can return MultiIndex columns.
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

            x["date"] = pd.to_datetime(
                x["date"], errors="coerce"
            )

            # Remove timezone if yfinance returns tz-aware timestamps.
            try:
                x["date"] = x["date"].dt.tz_localize(None)
            except TypeError:
                pass

            for c in ["open", "high", "low", "close", "volume"]:
                x[c] = pd.to_numeric(x[c], errors="coerce")

            x = x.dropna(subset=["date", "close"])

            return (
                ticker,
                x[
                    [
                        "date",
                        "ticker",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    ]
                ],
                None,
            )

        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return ticker, None, str(exc)

    return ticker, None, "unknown error"


def fetch_yfinance(tickers, start, end, workers=MAX_WORKERS):
    chunks = []
    diagnostics = []

    workers = max(1, min(int(workers), MAX_WORKERS))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_one, ticker, start, end): ticker
            for ticker in tickers
        }

        for future in as_completed(futures):
            ticker = futures[future]

            try:
                t, x, error = future.result()
            except Exception as exc:
                t, x, error = ticker, None, str(exc)

            if x is not None and not x.empty:
                chunks.append(x)
                diagnostics.append(
                    {
                        "ticker": t,
                        "ok": True,
                        "rows": int(len(x)),
                        "last_date": str(x["date"].max().date()),
                        "error": None,
                    }
                )
            else:
                diagnostics.append(
                    {
                        "ticker": t,
                        "ok": False,
                        "rows": 0,
                        "last_date": None,
                        "error": error,
                    }
                )

    data = pd.concat(chunks, ignore_index=True) if chunks else None
    diagnostics.sort(key=lambda item: item["ticker"])
    return data, diagnostics


def load_ihsg_trading_dates():
    """Return known IHSG trading dates from the repository's IHSG file."""
    ih = pd.read_csv(IHSG)

    if "Price" in ih.columns:
        dates = pd.to_datetime(ih["Price"], errors="coerce")
    elif "date" in ih.columns:
        dates = pd.to_datetime(ih["date"], errors="coerce")
    else:
        return pd.DatetimeIndex([])

    return pd.DatetimeIndex(
        dates.dropna().drop_duplicates().sort_values()
    )


def coverage_report(additions, tickers, start, end):
    if additions is None or additions.empty:
        return {
            "pass": False,
            "latest_ticker_coverage": 0.0,
            "row_coverage": 0.0,
            "tickers_with_data": 0,
            "expected_tickers": len(tickers),
            "rows_received": 0,
            "expected_rows": 0,
            "latest_date": None,
        }

    x = additions.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.dropna(subset=["date", "ticker"])
    x = x.drop_duplicates(["ticker", "date"])

    # Use actual IHSG trading dates when available. This is much more
    # realistic than treating every business day as a trading session.
    trading_dates = load_ihsg_trading_dates()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    period_dates = trading_dates[
        (trading_dates >= start_ts) & (trading_dates <= end_ts)
    ]

    if len(period_dates) == 0:
        period_dates = pd.bdate_range(start, end)

    expected_rows = int(len(period_dates) * len(tickers))
    rows_received = int(len(x))

    row_coverage = (
        rows_received / expected_rows
        if expected_rows > 0
        else 0.0
    )

    latest_date = x["date"].max()
    latest_tickers = int(
        x.loc[x["date"] == latest_date, "ticker"].nunique()
    )

    latest_ticker_coverage = (
        latest_tickers / len(tickers)
        if tickers
        else 0.0
    )

    passed = (
        latest_ticker_coverage >= MIN_LATEST_TICKER_COVERAGE
        and row_coverage >= MIN_ROW_COVERAGE
        and rows_received >= MIN_REPAIR_ROWS
    )

    return {
        "pass": bool(passed),
        "latest_ticker_coverage": round(
            latest_ticker_coverage, 4
        ),
        "row_coverage": round(row_coverage, 4),
        "tickers_with_data": latest_tickers,
        "expected_tickers": len(tickers),
        "rows_received": rows_received,
        "expected_rows": expected_rows,
        "latest_date": str(latest_date.date()),
    }


def normalize_ihsg_file():
    ih = pd.read_csv(IHSG)

    if "Price" in ih.columns:
        ih = ih[
            pd.to_datetime(
                ih["Price"], errors="coerce"
            ).notna()
        ].copy()

        ih = ih.rename(
            columns={
                "Price": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

    elif "date" in ih.columns:
        ih = ih[
            pd.to_datetime(
                ih["date"], errors="coerce"
            ).notna()
        ].copy()

    else:
        raise ValueError(
            "IHSG file has no recognized date column "
            "(expected 'Price' or 'date')."
        )

    ih["date"] = pd.to_datetime(
        ih["date"], errors="coerce"
    )
    ih["close"] = pd.to_numeric(
        ih["close"], errors="coerce"
    )

    return (
        ih.dropna(subset=["date", "close"])
        .drop_duplicates("date", keep="last")
        .sort_values("date")
    )


def update_ihsg(start, end):
    try:
        import yfinance as yf

        x = yf.download(
            "^JKSE",
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if x is None or x.empty:
            return None

        x = x.reset_index()

        if isinstance(x.columns, pd.MultiIndex):
            x.columns = [
                c[0] if isinstance(c, tuple) else c
                for c in x.columns
            ]

        x.columns = [
            str(c).lower().replace(" ", "_")
            for c in x.columns
        ]

        return pd.DataFrame(
            {
                "date": pd.to_datetime(x["date"]),
                "open": pd.to_numeric(
                    x["open"], errors="coerce"
                ),
                "high": pd.to_numeric(
                    x["high"], errors="coerce"
                ),
                "low": pd.to_numeric(
                    x["low"], errors="coerce"
                ),
                "close": pd.to_numeric(
                    x["close"], errors="coerce"
                ),
                "volume": pd.to_numeric(
                    x.get("volume"), errors="coerce"
                ),
            }
        ).dropna(subset=["date", "close"])

    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-days", type=int, default=REPAIR_DAYS)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    stocks = normalize_existing()

    old_last_date = stocks["date"].max().date()

    # IMPORTANT:
    # Always re-fetch a recent historical window. This is what repairs the
    # previous 67-row "successful" update instead of assuming that having
    # 2026-10-02 means the period is complete.
    end_date = pd.Timestamp.now(
        tz="Asia/Jakarta"
    ).date()

    start_date = (
        pd.Timestamp(end_date)
        - pd.Timedelta(days=int(args.repair_days))
    ).date()

    start = str(start_date)
    end = str(end_date)

    tickers = sorted(
        stocks["ticker"].dropna().astype(str).unique()
    )

    print(
        f"REPAIR MODE: {start} -> {end}; "
        f"{len(tickers)} tickers"
    )

    additions, diagnostics = fetch_yfinance(
        tickers,
        start,
        end,
        args.workers,
    )

    coverage = coverage_report(
        additions,
        tickers,
        start,
        end,
    )

    # Hard safety gate. Existing dataset is untouched if this fails.
    if not coverage["pass"]:
        result = {
            "status": "FAILED_DATA_COVERAGE",
            "mode": "repair",
            "old_last_date": str(old_last_date),
            "requested_start": start,
            "requested_end": end,
            "ticker_count": len(tickers),
            "coverage": coverage,
            "message": (
                "Repair rejected. Existing dataset was not "
                "overwritten because recent stock-data coverage "
                "did not pass the validation gate."
            ),
            "diagnostics": diagnostics,
        }

        MANIFEST.write_text(
            json.dumps(result, indent=2, default=str),
            encoding="utf-8",
        )

        raise RuntimeError(
            json.dumps(result, indent=2, default=str)
        )

    # Merge validated repair data with existing data.
    combined = pd.concat(
        [stocks, additions],
        ignore_index=True,
    )

    combined["date"] = pd.to_datetime(
        combined["date"], errors="coerce"
    )

    combined = (
        combined
        .dropna(subset=["ticker", "date", "close"])
        .drop_duplicates(
            ["ticker", "date"],
            keep="last",
        )
        .sort_values(["ticker", "date"])
    )

    combined.to_csv(STOCKS, index=False)

    # Update IHSG using the same recent window.
    ih = normalize_ihsg_file()
    ih_new = update_ihsg(start, end)

    if ih_new is not None and not ih_new.empty:
        ih = (
            pd.concat([ih, ih_new], ignore_index=True)
            .drop_duplicates("date", keep="last")
            .sort_values("date")
        )
        ih.to_csv(IHSG, index=False)

    # Final on-disk validation.
    final_stocks = pd.read_csv(
        STOCKS,
        parse_dates=["date"],
    )
    final_ihsg = pd.read_csv(
        IHSG,
        parse_dates=["date"],
    )

    final_last_date = final_stocks["date"].max().date()
    final_ihsg_date = final_ihsg["date"].max().date()

    latest_ticker_count = int(
        final_stocks.loc[
            final_stocks["date"]
            == pd.Timestamp(final_last_date),
            "ticker",
        ].nunique()
    )

    result = {
        "status": "SUCCESS",
        "mode": "repair",
        "updated_at": pd.Timestamp.now(
            tz="Asia/Jakarta"
        ).isoformat(),
        "source": "yfinance",
        "old_last_date": str(old_last_date),
        "new_last_date": str(final_last_date),
        "ihsg_last_date": str(final_ihsg_date),
        "requested_start": start,
        "requested_end": end,
        "ticker_count": len(tickers),
        "tickers_on_latest_stock_date": latest_ticker_count,
        "rows_added_before_dedup": int(len(additions)),
        "rows_final": int(len(final_stocks)),
        "coverage": coverage,
        "message": (
            "Recent historical repair completed and merged "
            "after passing coverage validation."
        ),
    }

    MANIFEST.write_text(
        json.dumps(result, indent=2, default=str),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
