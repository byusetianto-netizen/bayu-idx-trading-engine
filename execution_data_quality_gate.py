"""
V1.3F-2 Execution Data Quality Gate

Audit-only quality gate for the V1.3F raw execution-price probe.

Principles
----------
- Never repair source prices automatically.
- Never delete anomalous observations silently.
- Impossible OHLC is quarantined.
- Suspicious volume is review-only and execution-ineligible.
- Corporate actions are preserved as evidence and flagged separately.
- Production datasets and actual_engine.py are not modified.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PROBE_DIR = DATA / "execution_probe"

RAW_FILE = PROBE_DIR / "raw_execution_prices_probe.csv"
ACTIONS_FILE = PROBE_DIR / "corporate_actions_probe.csv"

GATED_OUTPUT = PROBE_DIR / "execution_quality_gate.csv"
ISSUES_OUTPUT = PROBE_DIR / "execution_quality_issues.csv"
SUMMARY_OUTPUT = PROBE_DIR / "execution_quality_summary.json"


def fail(message):
    raise RuntimeError(message)


def load_raw():
    if not RAW_FILE.exists():
        fail(f"Raw probe file not found: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)

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
        fail(f"Raw probe missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["ticker"] = (
        df["ticker"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_actions():
    if not ACTIONS_FILE.exists():
        fail(f"Corporate-action file not found: {ACTIONS_FILE}")

    actions = pd.read_csv(ACTIONS_FILE)

    required = {
        "date",
        "ticker",
        "dividends",
        "stock_splits",
    }

    missing = required - set(actions.columns)

    if missing:
        fail(
            f"Corporate-action file missing columns: {sorted(missing)}"
        )

    actions["date"] = pd.to_datetime(
        actions["date"],
        errors="coerce",
    )

    actions["ticker"] = (
        actions["ticker"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    actions["dividends"] = pd.to_numeric(
        actions["dividends"],
        errors="coerce",
    ).fillna(0.0)

    actions["stock_splits"] = pd.to_numeric(
        actions["stock_splits"],
        errors="coerce",
    ).fillna(0.0)

    if actions.duplicated(["ticker", "date"]).any():
        fail("Duplicate ticker/date found in corporate actions")

    return actions


def main():
    raw = load_raw()
    actions = load_actions()

    if raw.duplicated(["ticker", "date"]).any():
        fail("Duplicate ticker/date found in raw execution probe")

    df = raw.merge(
        actions,
        on=["ticker", "date"],
        how="left",
        validate="one_to_one",
    )

    df["dividends"] = df["dividends"].fillna(0.0)
    df["stock_splits"] = df["stock_splits"].fillna(0.0)

    price_cols = ["open", "high", "low", "close"]

    df["invalid_date"] = df["date"].isna()
    df["invalid_ticker"] = (
        df["ticker"].isna()
        | df["ticker"].str.strip().eq("")
    )

    df["missing_ohlcv"] = df[
        ["open", "high", "low", "close", "volume"]
    ].isna().any(axis=1)

    df["nonpositive_price"] = (
        df[price_cols].le(0).any(axis=1)
    )

    df["negative_volume"] = df["volume"] < 0

    row_max = df[["open", "close", "low"]].max(axis=1)
    row_min = df[["open", "close", "high"]].min(axis=1)

    df["ohlc_logic_error"] = (
        (df["high"] < row_max)
        | (df["low"] > row_min)
    )

    # Conservative review rules for this probe.
    # These are flags, not claims that such observations are universally wrong.
    df["zero_volume"] = df["volume"].eq(0)

    df["suspicious_low_volume"] = (
        df["volume"].gt(0)
        & df["volume"].le(100)
    )

    df["dividend_flag"] = df["dividends"].ne(0)
    df["stock_split_flag"] = df["stock_splits"].ne(0)

    df["corporate_action_flag"] = (
        df["dividend_flag"]
        | df["stock_split_flag"]
    )

    hard_invalid_cols = [
        "invalid_date",
        "invalid_ticker",
        "missing_ohlcv",
        "nonpositive_price",
        "negative_volume",
        "ohlc_logic_error",
    ]

    df["hard_invalid"] = df[hard_invalid_cols].any(axis=1)

    df["volume_review"] = (
        df["zero_volume"]
        | df["suspicious_low_volume"]
    )

    df["quality_status"] = np.select(
        [
            df["hard_invalid"],
            df["volume_review"],
            df["corporate_action_flag"],
        ],
        [
            "QUARANTINE",
            "REVIEW",
            "PASS_WITH_CORPORATE_ACTION_FLAG",
        ],
        default="PASS",
    )

    # Corporate action alone does not make an observation invalid.
    # Hard-invalid or volume-review rows cannot be used for execution.
    df["execution_eligible"] = (
        ~df["hard_invalid"]
        & ~df["volume_review"]
    )

    issue_cols = [
        "hard_invalid",
        "volume_review",
        "corporate_action_flag",
    ]

    issues = df[df[issue_cols].any(axis=1)].copy()

    df = df.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    issues = issues.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    df.to_csv(
        GATED_OUTPUT,
        index=False,
        date_format="%Y-%m-%d",
    )

    issues.to_csv(
        ISSUES_OUTPUT,
        index=False,
        date_format="%Y-%m-%d",
    )

    status_counts = {
        str(k): int(v)
        for k, v in df["quality_status"].value_counts().items()
    }

    ticker_summary = {}

    for ticker, x in df.groupby("ticker"):
        ticker_summary[str(ticker)] = {
            "rows": int(len(x)),
            "execution_eligible_rows": int(
                x["execution_eligible"].sum()
            ),
            "quarantine_rows": int(
                x["quality_status"].eq("QUARANTINE").sum()
            ),
            "review_rows": int(
                x["quality_status"].eq("REVIEW").sum()
            ),
            "corporate_action_rows": int(
                x["corporate_action_flag"].sum()
            ),
            "zero_volume_rows": int(
                x["zero_volume"].sum()
            ),
            "suspicious_low_volume_rows": int(
                x["suspicious_low_volume"].sum()
            ),
            "ohlc_logic_error_rows": int(
                x["ohlc_logic_error"].sum()
            ),
        }

    summary = {
        "version": "V1.3F-2",
        "status": "QUALITY_GATE_COMPLETE__NOT_ENABLED_IN_ENGINE",
        "source_file": str(
            RAW_FILE.relative_to(ROOT)
        ).replace("\\", "/"),
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()),
        "status_counts": status_counts,
        "execution_eligible_rows": int(
            df["execution_eligible"].sum()
        ),
        "execution_ineligible_rows": int(
            (~df["execution_eligible"]).sum()
        ),
        "quarantine_rows": int(
            df["quality_status"].eq("QUARANTINE").sum()
        ),
        "review_rows": int(
            df["quality_status"].eq("REVIEW").sum()
        ),
        "ohlc_logic_error_rows": int(
            df["ohlc_logic_error"].sum()
        ),
        "zero_volume_rows": int(
            df["zero_volume"].sum()
        ),
        "suspicious_low_volume_rows": int(
            df["suspicious_low_volume"].sum()
        ),
        "corporate_action_rows": int(
            df["corporate_action_flag"].sum()
        ),
        "ticker_summary": ticker_summary,
        "policy": {
            "impossible_ohlc": "QUARANTINE",
            "missing_or_nonpositive_price": "QUARANTINE",
            "negative_volume": "QUARANTINE",
            "zero_volume": "REVIEW_AND_EXECUTION_INELIGIBLE",
            "volume_1_to_100": "REVIEW_AND_EXECUTION_INELIGIBLE",
            "corporate_action": (
                "FLAG_ONLY_UNLESS_ANOTHER_QUALITY_RULE_FAILS"
            ),
            "automatic_price_repair": False,
            "automatic_anomaly_deletion": False,
        },
        "production_dataset_modified": False,
        "actual_engine_modified": False,
        "pit_safe_for_backtest": False,
        "execution_backtest_ready": False,
        "warning": (
            "Quality-gated probe remains research-only. "
            "Low-volume threshold is a conservative probe rule, "
            "not yet a validated universe-wide execution rule. "
            "PIT universe, official-price reconciliation, fees, "
            "slippage, tick size, and execution assumptions remain unresolved."
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

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()