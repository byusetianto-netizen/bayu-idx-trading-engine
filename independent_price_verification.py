"""
V1.3F-9 Independent Historical Price Verification

Research-only historical price comparison.

Rules:
- Never fabricate external prices.
- Never treat a source label as proof of independence.
- Never automatically repair historical prices.
- Never modify actual_engine.py or the raw dataset.
- Missing evidence must remain INSUFFICIENT_EVIDENCE.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

INTERNAL_FILE = ROOT / "data" / "idx_stock_prices.csv"

EVIDENCE_FILE = (
    ROOT / "data" / "execution_probe"
    / "independent_price_evidence.csv"
)

OUTPUT_DIR = ROOT / "data" / "execution_probe"

DETAIL_FILE = (
    OUTPUT_DIR / "independent_price_verification_detail.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "independent_price_verification_summary.json"
)


PERIODS = {
    "TOWR": ("2018-04-30", "2018-05-08"),
    "NISP": ("2018-04-16", "2018-04-25"),
}

EVIDENCE_COLUMNS = [
    "ticker",
    "date",
    "external_close",
    "price_type",
    "source_name",
    "source_url",
    "source_document",
    "verification_note",
]


def fail(message):
    raise RuntimeError(message)


def load_internal_prices():
    if not INTERNAL_FILE.exists():
        fail(f"Missing internal prices: {INTERNAL_FILE}")

    df = pd.read_csv(INTERNAL_FILE)

    required = {"ticker", "date", "close"}

    if not required.issubset(df.columns):
        fail("Internal dataset missing required columns.")

    df["date"] = pd.to_datetime(
        df["date"], errors="raise"
    )

    df["close"] = pd.to_numeric(
        df["close"], errors="coerce"
    )

    df["ticker"] = (
        df["ticker"].astype("string")
        .str.strip().str.upper()
    )

    if df.duplicated(["ticker", "date"]).any():
        fail("Duplicate ticker/date in internal dataset.")

    parts = []

    for ticker, (start, end) in PERIODS.items():
        subset = df.loc[
            df["ticker"].eq(ticker)
            & df["date"].between(start, end)
        ].copy()

        parts.append(subset)

    result = pd.concat(parts, ignore_index=True)

    if result.empty:
        fail("No internal observations in investigation periods.")

    result = result.rename(
        columns={"close": "internal_close"}
    )

    return result[["ticker", "date", "internal_close"]]


def load_external_evidence():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not EVIDENCE_FILE.exists():
        pd.DataFrame(
            columns=EVIDENCE_COLUMNS
        ).to_csv(EVIDENCE_FILE, index=False)

        print("External evidence template created.")
        print(f"Template: {EVIDENCE_FILE}")

        return pd.DataFrame(columns=EVIDENCE_COLUMNS)

    evidence = pd.read_csv(EVIDENCE_FILE, dtype="string")

    missing = set(EVIDENCE_COLUMNS) - set(evidence.columns)

    if missing:
        fail(
            "External evidence file missing columns: "
            f"{sorted(missing)}"
        )

    evidence = evidence[EVIDENCE_COLUMNS].copy()

    if evidence.empty:
        return evidence

    for column in EVIDENCE_COLUMNS:
        evidence[column] = (
            evidence[column].astype("string").str.strip()
        )

    evidence["ticker"] = evidence["ticker"].str.upper()

    evidence["date"] = pd.to_datetime(
        evidence["date"], errors="coerce"
    )

    evidence["external_close"] = pd.to_numeric(
        evidence["external_close"], errors="coerce"
    )

    allowed_price_types = {
        "RAW",
        "ADJUSTED",
        "UNKNOWN",
    }

    evidence["price_type"] = (
        evidence["price_type"].str.upper()
    )

    invalid = (
        evidence["date"].isna()
        | evidence["external_close"].isna()
        | evidence["external_close"].le(0)
        | ~evidence["price_type"].isin(allowed_price_types)
        | evidence["ticker"].isna()
        | evidence["source_name"].isna()
        | evidence["source_url"].isna()
    )

    if invalid.any():
        fail(
            "External evidence has invalid values or "
            "missing source metadata. No comparisons generated."
        )

    if evidence.duplicated(
        ["ticker", "date", "price_type", "source_name"]
    ).any():
        fail("Duplicate external evidence keys detected.")

    return evidence


def main():
    print("V1.3F-9 Independent Price Verification")
    print("-------------------------------------")

    internal = load_internal_prices()
    evidence = load_external_evidence()

    if evidence.empty:
        result = internal.copy()

        for column in EVIDENCE_COLUMNS:
            if column not in {"ticker", "date"}:
                result[column] = pd.NA

        result["close_ratio"] = np.nan
        result["absolute_percentage_difference"] = np.nan
        result["verification_status"] = "INSUFFICIENT_EVIDENCE"

        evidence_rows = 0

    else:
        result = internal.merge(
            evidence,
            on=["ticker", "date"],
            how="left",
            validate="one_to_many",
        )

        has_evidence = result["external_close"].notna()

        result["close_ratio"] = (
            result["internal_close"]
            / result["external_close"]
        ).where(has_evidence)

        result["absolute_percentage_difference"] = (
            result["close_ratio"] - 1
        ).abs() * 100

        # A numerical comparison is not independent verification.
        result["verification_status"] = np.where(
            has_evidence,
            "COMPARISON_AVAILABLE_REVIEW_REQUIRED",
            "INSUFFICIENT_EVIDENCE",
        )

        evidence_rows = int(len(evidence))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        DETAIL_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    status_counts = {
        str(k): int(v)
        for k, v in (
            result["verification_status"]
            .value_counts()
            .items()
        )
    }

    summary = {
        "version": "V1.3F-9",
        "stage": "INDEPENDENT_PRICE_VERIFICATION",
        "investigation_periods": PERIODS,
        "internal_observations": int(len(internal)),
        "external_evidence_records": evidence_rows,
        "comparison_rows": int(len(result)),
        "status_counts": status_counts,
        "corporate_action_causation_verified": False,
        "independent_source_authenticated": False,
        "production_price_correction_authorized": False,
        "policy": {
            "raw_dataset_modified": False,
            "actual_engine_modified": False,
            "automatic_price_repair": False,
            "automatic_quarantine": False,
        },
        "limitations": [
            "Numerical price agreement does not prove independence.",
            "RAW and ADJUSTED prices are not interchangeable.",
            "Source documents must be inspected independently.",
            "Corporate action effective dates require verification.",
            "No trading decision follows automatically.",
        ],
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print(f"Internal observations: {len(internal)}")
    print(f"External evidence records: {evidence_rows}")
    print(f"Comparison rows: {len(result)}")

    print()
    print("Verification status")
    print("-------------------")

    for status, count in status_counts.items():
        print(f"{status}: {count}")

    print()
    print("STATUS: EVIDENCE_REVIEW_REQUIRED")


if __name__ == "__main__":
    main()