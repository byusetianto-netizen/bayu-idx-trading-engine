"""
V1.3F-9 Corporate Action Reconciliation

Compare historical anomaly dates against documented
corporate-action events.

Research only:
- No automatic price correction
- No automatic anomaly clearance
- No changes to actual_engine.py
- No changes to raw market data
"""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent

INPUT_FILE = (
    ROOT / "data" / "execution_probe"
    / "adjusted_price_discontinuity_detail.csv"
)

OUTPUT_DIR = ROOT / "data" / "execution_probe"

DETAIL_OUTPUT = (
    OUTPUT_DIR / "corporate_action_reconciliation_detail.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR / "corporate_action_reconciliation_summary.json"
)


# Evidence records are manually curated.
#
# event_date_type is essential:
# an announcement date is NOT an effective trading date.
#
# Event evidence confirms an event, not the cause
# of any particular price anomaly.

EVENTS = [
    {
        "ticker": "NISP",
        "event_type": "BONUS_SHARES",
        "ratio": "1:1",
        "event_date": "2018-04-20",
        "event_date_type": "CUM_DATE",
        "source": "KSEI",
        "source_url": (
            "https://web.ksei.co.id/ksei_news/read/"
            "14074/Reminder-Corporate-Action-Cum-Date-NISP-AUTO-ADHI"
        ),
    },
    {
        "ticker": "NISP",
        "event_type": "BONUS_SHARES",
        "ratio": "1:1",
        "event_date": "2018-05-04",
        "event_date_type": "DISTRIBUTION_DATE",
        "source": "OCBC_2018_ANNUAL_REPORT",
        "source_url": (
            "https://cdn1.ocbc.id/asset/media/Feature/"
            "AboutOCBC/Hubungan-Investor/Laporan-Tahunan/"
            "2018/OCBC-NISP-AR-2018-Final-Website-Full.pdf"
            "?rev=-1"
        ),
    },
    {
        "ticker": "TOWR",
        "event_type": "STOCK_SPLIT",
        "ratio": "1:5",
        "event_date": "2018-06-25",
        "event_date_type": "ANNOUNCEMENT_DATE",
        "source": "IDNFINANCIALS_ANNOUNCEMENT_INDEX",
        "source_url": (
            "https://www.idnfinancials.com/id/"
            "announcement/towr/pt-sarana-menara-nusantara-tbk/3"
        ),
    },
]


def main():
    print("V1.3F-9 Corporate Action Reconciliation")
    print("---------------------------------------")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Audit detail is missing. Run "
            "adjusted_price_discontinuity_audit.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "ticker",
        "date",
        "close",
        "return_from_previous",
        "severity",
        "scale_pattern_class",
        "scale_reversal_ratio",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing audit columns: {sorted(missing)}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise",
    )
    # Separate price-scale anomalies from OHLC integrity issues.
    # Classification does not establish corporate-action causation.

    df["anomaly_type"] = "OTHER_PRICE_ANOMALY"

    scale_event = df["scale_pattern_class"].isin(
        ["STRONG_SCALE_PATTERN", "POSSIBLE_REVERSAL"]
    )

    ohlc_event = df["severity"].eq(
        "INVALID_PRICE_OR_OHLC"
    )

    df.loc[
        ohlc_event,
        "anomaly_type",
    ] = "OHLC_INTEGRITY_EVENT"

    df.loc[
        scale_event,
        "anomaly_type",
    ] = "PRICE_SCALE_EVENT"

    # First reconciliation scope: two priority tickers.
    df = df.loc[
        df["ticker"].isin(["NISP", "TOWR"])
    ].copy()

    events = pd.DataFrame(EVENTS)

    events["event_date"] = pd.to_datetime(
        events["event_date"],
        errors="raise",
    )

    # Many-to-many is intentional:
    # each anomaly is compared with each documented
    # event for the same ticker.
    comparison = df.merge(
        events,
        on="ticker",
        how="left",
        validate="many_to_many",
    )

    comparison["calendar_days_from_event"] = (
        comparison["date"] - comparison["event_date"]
    ).dt.days

    comparison["date_relationship"] = "UNDETERMINED"

    comparison.loc[
        comparison["calendar_days_from_event"] < 0,
        "date_relationship",
    ] = "ANOMALY_BEFORE_RECORDED_EVENT_DATE"

    comparison.loc[
        comparison["calendar_days_from_event"] == 0,
        "date_relationship",
    ] = "SAME_RECORDED_DATE"

    comparison.loc[
        comparison["calendar_days_from_event"] > 0,
        "date_relationship",
    ] = "ANOMALY_AFTER_RECORDED_EVENT_DATE"

    # Do not assert causation based on event proximity.
    comparison["causal_match_verified"] = False

    comparison["reconciliation_status"] = (
        "SOURCE_RECONCILIATION_REQUIRED"
    )

    comparison["review_note"] = (
        "Documented corporate action exists, but "
        "event-date alignment and adjusted-price "
        "factor consistency are not verified."
    )

    columns = [
        "ticker",
        "date",
        "anomaly_type",
        "close",
        "return_from_previous",
        "severity",
        "scale_pattern_class",
        "scale_reversal_ratio",
        "event_type",
        "ratio",
        "event_date",
        "event_date_type",
        "calendar_days_from_event",
        "date_relationship",
        "source",
        "source_url",
        "causal_match_verified",
        "reconciliation_status",
        "review_note",
    ]


    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison[columns].to_csv(
        DETAIL_OUTPUT,
        index=False,
        date_format="%Y-%m-%d",
    )

    summary = {
        "anomaly_type_counts": {
            str(k): int(v)
            for k, v in (
                df["anomaly_type"]
                .value_counts()
                .items()
            )
        },
        "version": "V1.3F-9",
        "stage": "CORPORATE_ACTION_RECONCILIATION",
        "status": "SOURCE_RECONCILIATION_REQUIRED",
        "scope_tickers": ["NISP", "TOWR"],
        "anomaly_rows_in_scope": int(len(df)),
        "evidence_event_records": int(len(events)),
        "comparison_rows": int(len(comparison)),
        "causal_matches_verified": 0,
        "policy": {
            "original_prices_modified": False,
            "automatic_price_adjustment": False,
            "automatic_quarantine": False,
            "production_engine_modified": False,
        },
        "limitations": [
            "Announcement, cum, and distribution dates "
            "are not interchangeable.",
            "Event proximity is not proof of causation.",
            "TOWR effective split date is not established "
            "by the recorded announcement index.",
            "Independent unadjusted daily price "
            "reconciliation remains pending.",
        ],
    }

    with open(
        SUMMARY_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print(f"Anomaly rows in scope: {len(df)}")
    print(f"Evidence records: {len(events)}")
    print(f"Comparison rows: {len(comparison)}")
    print("Causal matches verified: 0")
    print()
    print("STATUS: SOURCE_RECONCILIATION_REQUIRED")


if __name__ == "__main__":
    main()