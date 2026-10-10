"""
V1.3F-3 Trading Calendar Reconciliation

Research-only reconciliation between:
- V1.3F-2 execution quality gate
- existing IHSG date series

This stage classifies calendar context only.

It does NOT:
- repair prices
- delete observations
- infer official IDX trading calendars
- modify production datasets
- modify actual_engine.py
"""

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

QUALITY_FILE = (
    DATA / "execution_probe" / "execution_quality_gate.csv"
)

IHSG_FILE = DATA / "idx_ihsg_index.csv"

OUTPUT_FILE = (
    DATA
    / "execution_probe"
    / "trading_calendar_reconciliation.csv"
)

SUMMARY_FILE = (
    DATA
    / "execution_probe"
    / "trading_calendar_reconciliation_summary.json"
)


def fail(message):
    raise RuntimeError(message)


def main():
    if not QUALITY_FILE.exists():
        fail(f"Missing quality file: {QUALITY_FILE}")

    if not IHSG_FILE.exists():
        fail(f"Missing IHSG file: {IHSG_FILE}")

    df = pd.read_csv(QUALITY_FILE)
    ihsg = pd.read_csv(IHSG_FILE)

    required_quality = {
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "hard_invalid",
        "volume_review",
        "corporate_action_flag",
        "execution_eligible",
    }

    missing = required_quality - set(df.columns)

    if missing:
        fail(
            f"Quality file missing columns: {sorted(missing)}"
        )

    if "date" not in ihsg.columns:
        fail("IHSG file missing date column")

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    ihsg["date"] = pd.to_datetime(
        ihsg["date"],
        errors="coerce",
    )

    if df["date"].isna().any():
        fail("Invalid date found in quality-gate input")

    ihsg_dates = set(
        ihsg.loc[ihsg["date"].notna(), "date"]
    )

    df["in_ihsg_date_series"] = df["date"].isin(
        ihsg_dates
    )

    df["zero_volume"] = (
        pd.to_numeric(
            df["volume"],
            errors="coerce",
        ).eq(0)
    )

    df["non_trading_calendar_row"] = (
        df["zero_volume"]
        & ~df["in_ihsg_date_series"]
    )

    df["zero_volume_on_ihsg_date"] = (
        df["zero_volume"]
        & df["in_ihsg_date_series"]
    )

    # Important:
    # IHSG date presence is contextual evidence only.
    # It is NOT treated as proof of an official IDX trading session.
    df["calendar_classification"] = "NORMAL_CONTEXT"

    df.loc[
        df["non_trading_calendar_row"],
        "calendar_classification",
    ] = "NON_TRADING_CALENDAR_ROW"

    df.loc[
        df["zero_volume_on_ihsg_date"],
        "calendar_classification",
    ] = "ZERO_VOLUME_ON_IHSG_DATE"

    # Preserve V1.3F-2 safety.
    # Calendar reconciliation can make a row ineligible,
    # but can never promote an already-ineligible row.
    prior_eligible = (
        df["execution_eligible"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    df["calendar_execution_eligible"] = (
        prior_eligible
        & ~df["non_trading_calendar_row"]
        & ~df["zero_volume_on_ihsg_date"]
    )

    if (
        df["calendar_execution_eligible"]
        & ~prior_eligible
    ).any():
        fail(
            "Calendar stage illegally promoted "
            "an ineligible observation"
        )

    df = df.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    class_counts = {
        str(k): int(v)
        for k, v in df[
            "calendar_classification"
        ].value_counts().items()
    }

    zero_market = df[
        df["zero_volume_on_ihsg_date"]
    ].copy()

    if len(zero_market):
        simultaneous = (
            zero_market
            .groupby("date")["ticker"]
            .nunique()
        )

        zero_ihsg_dates = int(
            simultaneous.shape[0]
        )

        all_probe_tickers = int(
            df["ticker"].nunique()
        )

        all_ticker_zero_dates = int(
            simultaneous.eq(
                all_probe_tickers
            ).sum()
        )
    else:
        zero_ihsg_dates = 0
        all_ticker_zero_dates = 0

    summary = {
        "version": "V1.3F-3",
        "status": (
            "CALENDAR_RECONCILIATION_COMPLETE"
            "__NOT_ENABLED_IN_ENGINE"
        ),
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()),
        "calendar_classification_counts": (
            class_counts
        ),
        "zero_volume_rows": int(
            df["zero_volume"].sum()
        ),
        "non_trading_calendar_rows": int(
            df["non_trading_calendar_row"].sum()
        ),
        "zero_volume_on_ihsg_date_rows": int(
            df["zero_volume_on_ihsg_date"].sum()
        ),
        "zero_volume_on_ihsg_unique_dates": (
            zero_ihsg_dates
        ),
        "all_probe_tickers_zero_same_date_count": (
            all_ticker_zero_dates
        ),
        "calendar_execution_eligible_rows": int(
            df["calendar_execution_eligible"].sum()
        ),
        "policy": {
            "ihsg_date_presence": (
                "CONTEXT_ONLY_NOT_OFFICIAL_CALENDAR_PROOF"
            ),
            "non_trading_calendar_row": (
                "EXECUTION_INELIGIBLE"
            ),
            "zero_volume_on_ihsg_date": (
                "REVIEW_AND_EXECUTION_INELIGIBLE"
            ),
            "automatic_price_repair": False,
            "automatic_row_deletion": False,
            "eligibility_promotion_allowed": False,
        },
        "production_dataset_modified": False,
        "actual_engine_modified": False,
        "official_idx_calendar_verified": False,
        "pit_safe_for_backtest": False,
        "execution_backtest_ready": False,
        "warning": (
            "IHSG date presence is used only as contextual "
            "evidence. This stage does not establish an "
            "official historical IDX trading calendar."
        ),
    }

    SUMMARY_FILE.write_text(
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


if __name__ == "__main__":
    main()