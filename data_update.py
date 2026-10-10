"""Incrementally update the research dataset.

Primary source: Yahoo Finance via yfinance for the existing ticker universe.
Fallback source: daily all-stock snapshot published by nofendian17/idx_dataset.

Safety principles:
- Never write stock rows without a valid ticker.
- Never silently accept malformed OHLCV rows.
- Preserve the existing research dataset unless new data passes validation.
- Research use only; this is not an official IDX feed.
"""

from pathlib import Path
import argparse
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
STOCKS = DATA / "idx_stock_prices.csv"
IHSG = DATA / "idx_ihsg_index.csv"
MANIFEST = DATA / "update_manifest.json"

FALLBACK_BASE = (
    "https://raw.githubusercontent.com/"
    "nofendian17/idx_dataset/main/data/stock_data_{date}.csv"
)

STOCK_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def validate_stock_rows(df, context):
    """Normalize and validate stock OHLCV rows.

    Fail closed on missing/blank tickers or invalid dates.
    Invalid numeric OHLCV rows are removed.
    """
    if df is None:
        raise ValueError(f"{context}: dataframe is None")

    missing = [c for c in STOCK_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{context}: missing required columns: {missing}"
        )

    x = df[STOCK_COLUMNS].copy()

    x["date"] = pd.to_datetime(x["date"], errors="coerce")

    x["ticker"] = (
        x["ticker"]
        .astype("string")
        .str.strip()
        .str.upper()
        .str.replace(".JK", "", regex=False)
    )

    invalid_ticker = (
        x["ticker"].isna()
        | x["ticker"].eq("")
        | x["ticker"].str.lower().eq("nan")
    )

    if invalid_ticker.any():
        sample = x.loc[invalid_ticker].head(5).to_dict("records")
        raise ValueError(
            f"{context}: blank/invalid ticker detected. "
            f"count={int(invalid_ticker.sum())}, sample={sample}"
        )

    if x["date"].isna().any():
        raise ValueError(
            f"{context}: invalid date detected. "
            f"count={int(x['date'].isna().sum())}"
        )

    for c in ["open", "high", "low", "close", "volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    valid_numeric = (
        x[["open", "high", "low", "close", "volume"]]
        .notna()
        .all(axis=1)
    )

    valid_price = (
        (x["open"] > 0)
        & (x["high"] > 0)
        & (x["low"] > 0)
        & (x["close"] > 0)
        & (x["volume"] >= 0)
    )

    x = x.loc[valid_numeric & valid_price].copy()

    if x.empty:
        raise ValueError(
            f"{context}: no valid OHLCV rows remain after validation"
        )

    x = (
        x.drop_duplicates(["ticker", "date"], keep="last")
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    return x


def normalize_existing():
    """Load existing stock history and remove legacy malformed rows."""
    df = pd.read_csv(STOCKS)

    required = set(STOCK_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"existing stock dataset missing columns: {sorted(missing)}"
        )

    # Legacy cleanup:
    # old updater could create rows whose ticker was blank.
    ticker_text = df["ticker"].astype("string").str.strip()
    bad_ticker = (
        ticker_text.isna()
        | ticker_text.eq("")
        | ticker_text.str.lower().eq("nan")
    )

    removed_blank_tickers = int(bad_ticker.sum())

    if removed_blank_tickers:
        print(
            f"Removing {removed_blank_tickers} legacy row(s) "
            f"with blank ticker."
        )
        df = df.loc[~bad_ticker].copy()

    df = validate_stock_rows(df, "existing stock dataset")

    # Persist cleaned legacy dataset.
    df.to_csv(STOCKS, index=False)

    return df, removed_blank_tickers


def fetch_yfinance(tickers, start, end):
    try:
        import yfinance as yf
    except Exception:
        return None

    symbols = [f"{t}.JK" for t in tickers]

    raw = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if raw is None or raw.empty:
        return None

    rows = []

    for t in tickers:
        sym = f"{t}.JK"

        try:
            x = raw[sym].copy()
        except Exception:
            continue

        if x is None or x.empty:
            continue

        x = x.reset_index()
        x.columns = [
            str(c).lower().replace(" ", "_")
            for c in x.columns
        ]

        required_market = {
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }

        if not required_market.issubset(x.columns):
            continue

        # IMPORTANT:
        # Preserve ticker in the dataframe that is appended.
        x["ticker"] = t

        x = x[
            [
                "date",
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ].copy()

        rows.append(x)

    if not rows:
        return None

    result = pd.concat(rows, ignore_index=True)

    return validate_stock_rows(
        result,
        "yfinance additions",
    )


def fetch_fallback_day(date):
    url = FALLBACK_BASE.format(date=date)

    r = requests.get(url, timeout=30)

    if r.status_code != 200 or not r.text.strip():
        return None, r.status_code

    x = pd.read_csv(io.StringIO(r.text))

    x = x.rename(
        columns={
            "Date": "date",
            "Stock Code": "ticker",
            "Open Price": "open",
            "High Price": "high",
            "Low Price": "low",
            "Last Price": "close",
            "Volume": "volume",
        }
    )

    required = set(STOCK_COLUMNS)
    if not required.issubset(x.columns):
        return None, r.status_code

    x = x[STOCK_COLUMNS].copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")

    return x, r.status_code


def update_fallback(tickers, start, end, workers=8):
    days = pd.date_range(start, end, freq="B")

    chunks = []
    found = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                fetch_fallback_day,
                d.strftime("%Y-%m-%d"),
            ): d
            for d in days
        }

        for fut in as_completed(futs):
            d = futs[fut]

            try:
                x, status = fut.result()
            except Exception:
                x, status = None, None

            if x is None or x.empty:
                continue

            x["ticker"] = (
                x["ticker"]
                .astype("string")
                .str.strip()
                .str.upper()
                .str.replace(".JK", "", regex=False)
            )

            x = x[x["ticker"].isin(tickers)]

            if not x.empty:
                chunks.append(x)
                found.append(d.strftime("%Y-%m-%d"))

    if not chunks:
        return None, []

    result = pd.concat(chunks, ignore_index=True)

    result = validate_stock_rows(
        result,
        "fallback additions",
    )

    return result, sorted(found)


def update_ihsg(start, end):
    try:
        import yfinance as yf

        x = yf.download(
            "^JKSE",
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )

        if x is None or x.empty:
            return None

        x = x.reset_index()

        if hasattr(x.columns, "levels"):
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
                "date": pd.to_datetime(
                    x["date"],
                    errors="coerce",
                ),
                "open": x["open"],
                "high": x["high"],
                "low": x["low"],
                "close": x["close"],
                "volume": x.get("volume"),
            }
        )

    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--workers", type=int, default=8)

    args = ap.parse_args()

    stocks, removed_blank_tickers = normalize_existing()

    old_last = stocks["date"].max().date()

    start = (
        args.start
        or (
            old_last + pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d")
    )

    end = (
        args.end
        or pd.Timestamp.today(
            tz="Asia/Jakarta"
        ).date().strftime("%Y-%m-%d")
    )

    tickers = sorted(
        stocks["ticker"]
        .dropna()
        .astype(str)
        .unique()
    )

    additions = fetch_yfinance(
        tickers,
        start,
        end,
    )

    source = "yfinance"

    if additions is None or additions.empty:
        additions, days = update_fallback(
            tickers,
            start,
            end,
            args.workers,
        )
        source = "github-daily-snapshot"
    else:
        days = sorted(
            pd.to_datetime(additions["date"])
            .dt.strftime("%Y-%m-%d")
            .unique()
        )

    if additions is not None and not additions.empty:
        # Defensive validation before merge.
        additions = validate_stock_rows(
            additions,
            f"{source} additions",
        )

        combined = pd.concat(
            [stocks, additions],
            ignore_index=True,
        )

        # Fail closed before production write.
        combined = validate_stock_rows(
            combined,
            "combined stock dataset",
        )

        combined.to_csv(STOCKS, index=False)

    else:
        days = []

    ih = pd.read_csv(IHSG)

    if "Price" in ih.columns:
        ih = ih[
            pd.to_datetime(
                ih["Price"],
                errors="coerce",
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
                ih["date"],
                errors="coerce",
            ).notna()
        ].copy()

    else:
        raise ValueError(
            "IHSG file has no recognized date column "
            "(expected 'Price' or 'date')."
        )

    ih["date"] = pd.to_datetime(
        ih["date"],
        errors="coerce",
    )

    ihnew = update_ihsg(start, end)

    if ihnew is not None and not ihnew.empty:
        ih = (
            pd.concat(
                [ih, ihnew],
                ignore_index=True,
            )
            .drop_duplicates(
                "date",
                keep="last",
            )
            .sort_values("date")
        )

        ih.to_csv(IHSG, index=False)

    # Final production validation.
    final_stock = pd.read_csv(STOCKS)

    final_stock = validate_stock_rows(
        final_stock,
        "final production stock dataset",
    )

    final_stock.to_csv(STOCKS, index=False)

    final_ihsg = pd.read_csv(
        IHSG,
        parse_dates=["date"],
    )

    new_last_stock = final_stock["date"].max().date()
    new_last_ihsg = final_ihsg["date"].max().date()

    latest_stock_date = final_stock["date"].max()

    latest_ticker_count = int(
        final_stock.loc[
            final_stock["date"].eq(latest_stock_date),
            "ticker",
        ].nunique()
    )

    blank_ticker_rows = int(
        final_stock["ticker"].isna().sum()
        + final_stock["ticker"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if blank_ticker_rows != 0:
        raise ValueError(
            "FAIL-CLOSED: final stock dataset "
            "contains blank ticker rows."
        )

    result = {
        "updated_at": pd.Timestamp.now(
            tz="Asia/Jakarta"
        ).isoformat(),
        "source": source,
        "old_last_date": str(old_last),
        "new_last_date": str(new_last_stock),
        "ihsg_last_date": str(new_last_ihsg),
        "requested_start": start,
        "requested_end": end,
        "new_days": days,
        "ticker_count": len(tickers),
        "latest_date_ticker_count": latest_ticker_count,
        "legacy_blank_ticker_rows_removed": (
            removed_blank_tickers
        ),
        "blank_ticker_rows_final": blank_ticker_rows,
        "rows": int(len(final_stock)),
    }

    MANIFEST.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()