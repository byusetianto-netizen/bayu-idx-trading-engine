"""
Bayu IDX Trading Engine
V1.3F-11 Stage A — Calendar Integrity Diagnostic

RESEARCH ONLY

Purpose:
- Audit calendar context and carry-forward patterns.
- Separate off-IHSG-date observations from on-IHSG-date observations.
- Preserve all original observations.
- Produce reproducible research reports.

Important:
- IHSG dates are contextual evidence, NOT an official IDX calendar.
- Zero volume does not automatically mean invalid historical data.
- No rows are removed or corrected.
- No production model or trading signals are modified.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


VERSION = "V1.3F-11"
STAGE = "A_CALENDAR_INTEGRITY_DIAGNOSTIC"

STOCK_PATH = Path("data/idx_stock_prices.csv")
IHSG_PATH = Path("data/idx_ihsg_index.csv")

OUTPUT_DIR = Path("data/execution_probe")

SUMMARY_PATH = (
    OUTPUT_DIR / "v13f11_calendar_integrity_summary.json"
)

DETAIL_PATH = (
    OUTPUT_DIR / "v13f11_calendar_integrity_detail.csv"
)

REQUIRED_STOCK_COLUMNS = {
    "date", "ticker", "open", "high", "low", "close", "volume"
}

OHLC_COLUMNS = ["open", "high", "low", "close"]

FLOAT_RTOL = 1e-12
FLOAT_ATOL = 1e-9


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_data():
    if not STOCK_PATH.is_file():
        raise FileNotFoundError(STOCK_PATH)

    if not IHSG_PATH.is_file():
        raise FileNotFoundError(IHSG_PATH)

    stock = pd.read_csv(STOCK_PATH)
    ihsg = pd.read_csv(IHSG_PATH)

    stock.columns = [
        str(column).strip().lower()
        for column in stock.columns
    ]

    ihsg.columns = [
        str(column).strip().lower()
        for column in ihsg.columns
    ]

    missing = REQUIRED_STOCK_COLUMNS - set(stock.columns)

    if missing:
        raise ValueError(
            f"Missing stock columns: {sorted(missing)}"
        )

    if "date" in ihsg.columns:
        ihsg_date_column = "date"
    elif "price" in ihsg.columns:
        ihsg_date_column = "price"
    else:
        raise ValueError(
            "IHSG dataset requires date or Price column"
        )

    stock["date"] = pd.to_datetime(
        stock["date"], errors="coerce"
    ).dt.normalize()

    ihsg["date"] = pd.to_datetime(
        ihsg[ihsg_date_column], errors="coerce"
    ).dt.normalize()

    if stock["date"].isna().any():
        raise ValueError("Invalid stock dates detected")

    stock["ticker"] = (
        stock["ticker"].astype("string").str.strip().str.upper()
    )

    if stock["ticker"].isna().any() or stock["ticker"].eq("").any():
        raise ValueError("Invalid ticker detected")

    for column in OHLC_COLUMNS + ["volume"]:
        stock[column] = pd.to_numeric(
            stock[column], errors="coerce"
        )

    if stock.duplicated(["ticker", "date"]).any():
        raise ValueError(
            "Duplicate ticker-date observations detected. "
            "Resolve through a separate audit, not automatic deletion."
        )

    stock = stock.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    return stock, ihsg


def main():
    stock, ihsg = load_data()

    ihsg_dates = set(ihsg["date"].dropna())

    stock["in_ihsg_context"] = stock["date"].isin(
        ihsg_dates
    )

    stock["previous_close"] = (
        stock.groupby("ticker")["close"].shift(1)
    )

    stock["zero_volume"] = stock["volume"].eq(0)

    finite_ohlc = pd.Series(
        np.isfinite(
            stock[OHLC_COLUMNS].to_numpy(dtype=float)
        ).all(axis=1),
        index=stock.index,
    )

    stock["flat_ohlc"] = finite_ohlc.copy()

    for column in ["high", "low", "close"]:
        stock["flat_ohlc"] &= np.isclose(
            stock["open"],
            stock[column],
            rtol=FLOAT_RTOL,
            atol=FLOAT_ATOL,
            equal_nan=False,
        )

    stock["same_previous_close"] = np.isclose(
        stock["close"],
        stock["previous_close"],
        rtol=FLOAT_RTOL,
        atol=FLOAT_ATOL,
        equal_nan=False,
    )

    stock["carry_forward_pattern"] = (
        stock["zero_volume"]
        & stock["flat_ohlc"]
        & stock["same_previous_close"]
    )

    off_calendar = ~stock["in_ihsg_context"]
    on_calendar = stock["in_ihsg_context"]

    stock["classification"] = np.select(
        [
            off_calendar & stock["carry_forward_pattern"],
            off_calendar,
            on_calendar & stock["carry_forward_pattern"],
            on_calendar & stock["zero_volume"],
        ],
        [
            "OFF_IHSG_CARRY_FORWARD_CANDIDATE",
            "OFF_IHSG_OTHER_REVIEW",
            "ON_IHSG_CARRY_FORWARD_REVIEW",
            "ON_IHSG_ZERO_VOLUME_REVIEW",
        ],
        default="NORMAL_CONTEXT",
    )

    # This is a research-only candidate exclusion.
    # It is NOT an official trading-calendar determination.
    stock["scenario_exclusion_candidate"] = (
        stock["classification"]
        == "OFF_IHSG_CARRY_FORWARD_CANDIDATE"
    )

    counts = {
        str(key): int(value)
        for key, value in (
            stock["classification"].value_counts().items()
        )
    }

    off = stock.loc[off_calendar]
    on = stock.loc[on_calendar]

    summary = {
        "version": VERSION,
        "stage": STAGE,
        "status": "RESEARCH_DIAGNOSTIC_ONLY",
        "sources": {
            "stock_path": str(STOCK_PATH),
            "stock_sha256": sha256_file(STOCK_PATH),
            "ihsg_path": str(IHSG_PATH),
            "ihsg_sha256": sha256_file(IHSG_PATH),
        },
        "statistics": {
            "stock_rows": int(len(stock)),
            "tickers": int(stock["ticker"].nunique()),
            "stock_unique_dates": int(stock["date"].nunique()),
            "ihsg_unique_dates": int(
                ihsg["date"].dropna().nunique()
            ),
            "off_ihsg_dates": int(
                off["date"].nunique()
            ),
            "off_ihsg_rows": int(len(off)),
            "off_ihsg_carry_forward_rows": int(
                off["carry_forward_pattern"].sum()
            ),
            "on_ihsg_rows": int(len(on)),
            "on_ihsg_carry_forward_rows": int(
                on["carry_forward_pattern"].sum()
            ),
            "total_carry_forward_rows": int(
                stock["carry_forward_pattern"].sum()
            ),
            "scenario_exclusion_candidates": int(
                stock["scenario_exclusion_candidate"].sum()
            ),
        },
        "classification_counts": counts,
        "method": {
            "calendar_reference": "IHSG_CONTEXT_ONLY",
            "float_rtol": FLOAT_RTOL,
            "float_atol": FLOAT_ATOL,
            "carry_forward_definition": (
                "ZERO_VOLUME_AND_FLAT_OHLC_AND_"
                "CLOSE_EQUALS_PREVIOUS_CLOSE"
            ),
        },
        "policy": {
            "official_idx_calendar_verified": False,
            "automatic_deletion": False,
            "automatic_price_correction": False,
            "production_dataset_modified": False,
            "actual_engine_modified": False,
            "production_model_replaced": False,
            "pit_backtest_ready": False,
        },
        "limitations": [
            "IHSG dates are not an official IDX trading calendar.",
            "On-IHSG zero-volume rows may represent valid sessions.",
            "Carry-forward patterns do not establish data origin.",
            "Previous close refers to the preceding ticker observation.",
            "Scenario exclusion is hypothetical and research-only.",
            "Feature, label and model impacts are not measured here.",
        ],
    }

    if (
        len(off) + len(on) != len(stock)
        or sum(counts.values()) != len(stock)
    ):
        raise RuntimeError(
            "Calendar classification accounting mismatch"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detail_columns = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "previous_close",
        "in_ihsg_context",
        "zero_volume",
        "flat_ohlc",
        "same_previous_close",
        "carry_forward_pattern",
        "classification",
        "scenario_exclusion_candidate",
    ]

    stock[detail_columns].to_csv(
        DETAIL_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n=== V1.3F-11 STAGE A ===")

    for key, value in summary["statistics"].items():
        print(f"{key}: {value:,}")

    print("\nClassification counts:")

    for key, value in counts.items():
        print(f"{key}: {value:,}")

    print(f"\nSaved: {SUMMARY_PATH}")
    print(f"Saved: {DETAIL_PATH}")

    print("STATUS: RESEARCH_DIAGNOSTIC_COMPLETED")


if __name__ == "__main__":
    main()