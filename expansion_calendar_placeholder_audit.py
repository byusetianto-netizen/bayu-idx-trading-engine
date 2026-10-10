#!/usr/bin/env python3
"""
V1.3F-4 Expansion Calendar Placeholder Audit

Audits stock observations whose dates are absent from the available
IHSG date series and determines whether they have placeholder-like
characteristics.

IMPORTANT:
- IHSG date presence is contextual evidence only.
- This is NOT an official historical IDX trading calendar.
- No rows are deleted or repaired.
- No production dataset is modified.
- actual_engine.py is not modified.
- This stage does not solve PIT universe membership or survivorship bias.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "V1.3F-4"

STOCK_PATH = Path("data/idx_stock_prices_expansion.csv")
IHSG_PATH = Path("data/idx_ihsg_index.csv")

OUTPUT_DETAIL = Path("data/execution_probe/expansion_calendar_placeholder_audit.csv")
OUTPUT_SUMMARY = Path("data/execution_probe/expansion_calendar_placeholder_summary.json")

FLOAT_RTOL = 1e-12
FLOAT_ATOL = 1e-9


def normalized_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def main():
    if not STOCK_PATH.exists():
        raise FileNotFoundError(STOCK_PATH)

    if not IHSG_PATH.exists():
        raise FileNotFoundError(IHSG_PATH)

    stock = pd.read_csv(STOCK_PATH)
    ihsg = pd.read_csv(IHSG_PATH)

    stock.columns = [str(c).strip().lower() for c in stock.columns]
    ihsg.columns = [str(c).strip().lower() for c in ihsg.columns]

    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = required - set(stock.columns)

    if missing:
        raise ValueError(f"Missing stock columns: {sorted(missing)}")

    ihsg_date_col = "date" if "date" in ihsg.columns else "price" if "price" in ihsg.columns else None

    if ihsg_date_col is None:
        raise ValueError(f"IHSG file has no date/Price column: {list(ihsg.columns)}")

    stock["date"] = normalized_dates(stock["date"])
    ihsg_dates = set(normalized_dates(ihsg[ihsg_date_col]).dropna())

    stock["ticker"] = stock["ticker"].astype(str).str.upper().str.strip()

    for col in ["open", "high", "low", "close", "volume"]:
        stock[col] = pd.to_numeric(stock[col], errors="coerce")

    stock_only = stock[
        stock["date"].notna()
        & ~stock["date"].isin(ihsg_dates)
    ].copy()

    stock_only["zero_volume"] = stock_only["volume"].eq(0)

    stock_only["exact_flat_ohlc"] = (
        stock_only["open"].eq(stock_only["high"])
        & stock_only["high"].eq(stock_only["low"])
        & stock_only["low"].eq(stock_only["close"])
    )

    finite_ohlc = np.isfinite(
        stock_only[["open", "high", "low", "close"]].to_numpy(dtype=float)
    ).all(axis=1)

    close_open = np.isclose(
        stock_only["close"],
        stock_only["open"],
        rtol=FLOAT_RTOL,
        atol=FLOAT_ATOL,
        equal_nan=False,
    )

    high_open = np.isclose(
        stock_only["high"],
        stock_only["open"],
        rtol=FLOAT_RTOL,
        atol=FLOAT_ATOL,
        equal_nan=False,
    )

    low_open = np.isclose(
        stock_only["low"],
        stock_only["open"],
        rtol=FLOAT_RTOL,
        atol=FLOAT_ATOL,
        equal_nan=False,
    )

    stock_only["effective_flat_ohlc"] = (
        finite_ohlc
        & close_open
        & high_open
        & low_open
    )

    stock_only["placeholder_candidate"] = (
        stock_only["zero_volume"]
        & stock_only["effective_flat_ohlc"]
    )

    stock_only["classification"] = np.select(
        [
            stock_only["placeholder_candidate"],
            stock_only["zero_volume"],
        ],
        [
            "PLACEHOLDER_CANDIDATE",
            "ZERO_VOLUME_NONFLAT_REVIEW",
        ],
        default="STOCK_ONLY_NONZERO_VOLUME_REVIEW",
    )

    detail_columns = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "zero_volume",
        "exact_flat_ohlc",
        "effective_flat_ohlc",
        "placeholder_candidate",
        "classification",
    ]

    detail = stock_only[detail_columns].copy()
    detail["date"] = detail["date"].dt.strftime("%Y-%m-%d")

    classification_counts = {
        str(k): int(v)
        for k, v in detail["classification"].value_counts().to_dict().items()
    }

    stock_only_rows = int(len(detail))
    zero_volume_rows = int(detail["zero_volume"].sum())
    exact_flat_zero_volume_rows = int(
        (detail["zero_volume"] & detail["exact_flat_ohlc"]).sum()
    )
    effective_flat_zero_volume_rows = int(
        (detail["zero_volume"] & detail["effective_flat_ohlc"]).sum()
    )
    placeholder_candidate_rows = int(detail["placeholder_candidate"].sum())

    summary = {
        "version": VERSION,
        "status": "PLACEHOLDER_AUDIT_COMPLETE__DATASETS_UNCHANGED",
        "source_stock_dataset": str(STOCK_PATH).replace("\\", "/"),
        "ihsg_context_dataset": str(IHSG_PATH).replace("\\", "/"),
        "stock_rows": int(len(stock)),
        "stock_unique_tickers": int(stock["ticker"].nunique()),
        "stock_only_rows": stock_only_rows,
        "stock_only_unique_tickers": int(detail["ticker"].nunique()) if len(detail) else 0,
        "zero_volume_stock_only_rows": zero_volume_rows,
        "exact_flat_zero_volume_rows": exact_flat_zero_volume_rows,
        "effective_flat_zero_volume_rows": effective_flat_zero_volume_rows,
        "placeholder_candidate_rows": placeholder_candidate_rows,
        "non_placeholder_stock_only_rows": int(
            stock_only_rows - placeholder_candidate_rows
        ),
        "classification_counts": classification_counts,
        "float_comparison_policy": {
            "rtol": FLOAT_RTOL,
            "atol": FLOAT_ATOL,
            "purpose": "Handle insignificant floating-point representation differences only.",
        },
        "policy": {
            "ihsg_date_presence": "CONTEXT_ONLY_NOT_OFFICIAL_CALENDAR_PROOF",
            "placeholder_definition": "DATE_ABSENT_FROM_IHSG_CONTEXT_AND_ZERO_VOLUME_AND_EFFECTIVELY_FLAT_OHLC",
            "placeholder_is_proven_non_trading_session": False,
            "automatic_row_deletion": False,
            "automatic_price_repair": False,
            "production_dataset_modified": False,
            "actual_engine_modified": False,
            "official_idx_calendar_verified": False,
            "pit_safe_for_backtest": False,
            "execution_backtest_ready": False,
        },
        "outputs": [
            str(OUTPUT_DETAIL).replace("\\", "/"),
            str(OUTPUT_SUMMARY).replace("\\", "/"),
        ],
        "warning": (
            "PLACEHOLDER_CANDIDATE is a data-quality classification, not proof "
            "that the date was an official IDX non-trading session."
        ),
    }

    OUTPUT_DETAIL.parent.mkdir(parents=True, exist_ok=True)

    detail.to_csv(OUTPUT_DETAIL, index=False)

    OUTPUT_SUMMARY.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Version: {VERSION}")
    print(f"Stock rows: {summary['stock_rows']}")
    print(f"Unique tickers: {summary['stock_unique_tickers']}")
    print(f"Stock-only rows: {stock_only_rows}")
    print(f"Zero-volume stock-only rows: {zero_volume_rows}")
    print(f"Exact-flat zero-volume rows: {exact_flat_zero_volume_rows}")
    print(f"Effective-flat zero-volume rows: {effective_flat_zero_volume_rows}")
    print(f"Placeholder candidates: {placeholder_candidate_rows}")
    print(f"Non-placeholder stock-only rows: {summary['non_placeholder_stock_only_rows']}")
    print("STATUS: PLACEHOLDER_AUDIT_COMPLETE__DATASETS_UNCHANGED")


if __name__ == "__main__":
    main()
