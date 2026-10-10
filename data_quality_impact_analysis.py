"""
V1.3F-10 — Data Quality Impact Assessment
Stage A: Feature and Forward-Label Exposure Audit

Research-only diagnostic.

No price correction.
No deletion of historical rows.
No model retraining.
No changes to actual_engine.py.

This audit measures potential dependency exposure,
not a confirmed numerical change in features or labels.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

PRICE_FILE = ROOT / "data" / "idx_stock_prices.csv"

AUDIT_FILE = (
    ROOT / "data" / "execution_probe"
    / "adjusted_price_discontinuity_detail.csv"
)

OUT_DIR = ROOT / "data" / "execution_probe"

SUMMARY_FILE = OUT_DIR / "data_quality_impact_summary.json"

DETAIL_FILE = OUT_DIR / "data_quality_impact_detail.csv"


# The largest stock-price rolling window in actual_engine.py.
MAX_FEATURE_LOOKBACK = 200

# Forward labels in actual_engine.py use five next observations.
FORWARD_HORIZON = 5


def fail(message):
    raise RuntimeError(message)


def load_prices():
    if not PRICE_FILE.exists():
        fail(f"Missing price file: {PRICE_FILE}")

    df = pd.read_csv(
        PRICE_FILE,
        usecols=["ticker", "date"],
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise",
    )

    df["ticker"] = (
        df["ticker"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if df["ticker"].isna().any():
        fail("Missing ticker in price data.")

    if df.duplicated(["ticker", "date"]).any():
        fail("Duplicate ticker/date in price data.")

    return (
        df.sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )


def load_audit():
    if not AUDIT_FILE.exists():
        fail(
            "Missing audit detail CSV. Run "
            "adjusted_price_discontinuity_audit.py first."
        )

    audit = pd.read_csv(AUDIT_FILE)

    required = {
        "ticker",
        "date",
        "invalid_ohlc",
        "scale_pattern_class",
        "extreme_return",
    }

    missing = required - set(audit.columns)

    if missing:
        fail(f"Missing audit columns: {sorted(missing)}")

    audit["date"] = pd.to_datetime(
        audit["date"],
        errors="raise",
    )

    audit["ticker"] = (
        audit["ticker"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if audit.duplicated(["ticker", "date"]).any():
        fail("Duplicate ticker/date in audit.")

    for col in ["invalid_ohlc", "extreme_return"]:
        values = (
            audit[col]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        if not values.isin(["true", "false"]).all():
            fail(f"Invalid boolean values in {col}.")

        audit[col] = values.eq("true")

    audit["scale_pattern_class"] = (
        audit["scale_pattern_class"]
        .fillna("NOT_DETECTED")
        .astype(str)
    )

    allowed = {
        "NOT_DETECTED",
        "POSSIBLE_REVERSAL",
        "STRONG_SCALE_PATTERN",
    }

    if not audit["scale_pattern_class"].isin(allowed).all():
        fail("Unknown scale pattern classification.")

    return audit


def build_exposure(prices, audit):
    x = prices.merge(
        audit[
            [
                "ticker",
                "date",
                "invalid_ohlc",
                "extreme_return",
                "scale_pattern_class",
            ]
        ],
        on=["ticker", "date"],
        how="left",
        validate="one_to_one",
    )

    x["invalid_ohlc"] = (
        x["invalid_ohlc"].fillna(False).astype(bool)
    )

    x["extreme_return"] = (
        x["extreme_return"].fillna(False).astype(bool)
    )

    x["scale_pattern_class"] = (
        x["scale_pattern_class"]
        .fillna("NOT_DETECTED")
    )

    # Conservative investigation scenario:
    # invalid OHLC or a strong scale-pattern observation.
    #
    # This is NOT proof that the row contains wrong prices.
    x["quality_review_anchor"] = (
        x["invalid_ohlc"]
        | x["scale_pattern_class"].eq(
            "STRONG_SCALE_PATTERN"
        )
    )

    x["quality_review_reason"] = np.select(
        [
            x["invalid_ohlc"]
            & x["scale_pattern_class"].eq(
                "STRONG_SCALE_PATTERN"
            ),
            x["invalid_ohlc"],
            x["scale_pattern_class"].eq(
                "STRONG_SCALE_PATTERN"
            ),
        ],
        [
            "BOTH",
            "INVALID_OHLC",
            "STRONG_SCALE_PATTERN",
        ],
        default="NONE",
    )

    g = x.groupby("ticker", sort=False)

    anchor = x["quality_review_anchor"]

    # Stock-price features use current and earlier rows.
    #
    # For conservative exposure analysis, each anchor
    # affects its own row and up to 199 following rows.
    feature_exposure = anchor.copy()

    for lag in range(1, MAX_FEATURE_LOOKBACK):
        previous_anchor = (
            anchor.groupby(x["ticker"])
            .shift(lag)
            .fillna(False)
            .astype(bool)
        )

        feature_exposure |= previous_anchor

    x["potential_feature_exposure"] = feature_exposure

    # Forward labels use the next five observations,
    # plus the current close as the reference price.
    #
    # Therefore an anchor can affect its own label and
    # labels on up to five preceding observations.
    label_exposure = anchor.copy()

    for lead in range(1, FORWARD_HORIZON + 1):
        future_anchor = (
            anchor.groupby(x["ticker"])
            .shift(-lead)
            .fillna(False)
            .astype(bool)
        )

        label_exposure |= future_anchor

    x["potential_label_exposure"] = label_exposure

    x["potential_any_exposure"] = (
        x["potential_feature_exposure"]
        | x["potential_label_exposure"]
    )

    x["sample_period"] = np.select(
        [
            x["date"] < "2023-01-01",
            x["date"].between(
                "2023-01-01",
                "2024-12-31",
            ),
        ],
        [
            "TRAIN_PERIOD",
            "VALIDATION_PERIOD",
        ],
        default="POST_VALIDATION",
    )

    return x


def count_metrics(df):
    return {
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()),
        "review_anchor_rows": int(
            df["quality_review_anchor"].sum()
        ),
        "potential_feature_exposure_rows": int(
            df["potential_feature_exposure"].sum()
        ),
        "potential_label_exposure_rows": int(
            df["potential_label_exposure"].sum()
        ),
        "potential_any_exposure_rows": int(
            df["potential_any_exposure"].sum()
        ),
    }


def main():
    print("V1.3F-10 Data Quality Impact — Stage A")
    print("-------------------------------------")

    prices = load_prices()
    audit = load_audit()

    missing_keys = (
        audit[["ticker", "date"]]
        .merge(
            prices,
            on=["ticker", "date"],
            how="left",
            indicator=True,
        )
    )

    if missing_keys["_merge"].ne("both").any():
        fail("Audit contains keys missing from price data.")

    result = build_exposure(prices, audit)

    periods = {}

    for period, part in result.groupby("sample_period"):
        periods[str(period)] = count_metrics(part)

    anchor_by_reason = {
        str(key): int(value)
        for key, value in (
            result.loc[
                result["quality_review_anchor"],
                "quality_review_reason",
            ]
            .value_counts()
            .items()
        )
    }

    summary = {
        "version": "V1.3F-10",
        "stage": "A_FEATURE_LABEL_EXPOSURE",
        "status": "RESEARCH_DIAGNOSTIC_ONLY",
        "overall": count_metrics(result),
        "periods": periods,
        "review_anchor_by_reason": anchor_by_reason,
        "parameters": {
            "max_feature_lookback": MAX_FEATURE_LOOKBACK,
            "forward_label_horizon": FORWARD_HORIZON,
        },
        "policy": {
            "raw_prices_modified": False,
            "actual_engine_modified": False,
            "observations_deleted": False,
            "automatic_quarantine": False,
            "model_retrained": False,
            "corporate_action_causation_verified": False,
        },
        "limitations": [
            "Exposure is a conservative dependency estimate.",
            "Exposure does not prove a feature or label changed.",
            "The 200-row window is an upper-bound proxy.",
            "All three forward-label targets share the "
            "same five-row horizon.",
            "Some strong scale patterns may arise from "
            "legitimate corporate actions.",
            "Model impact requires a separate controlled "
            "experiment.",
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    result[
        [
            "ticker",
            "date",
            "sample_period",
            "invalid_ohlc",
            "extreme_return",
            "scale_pattern_class",
            "quality_review_anchor",
            "quality_review_reason",
            "potential_feature_exposure",
            "potential_label_exposure",
            "potential_any_exposure",
        ]
    ].to_csv(
        DETAIL_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    overall = summary["overall"]

    print(f"Price rows: {overall['rows']:,}")
    print(
        "Review anchors: "
        f"{overall['review_anchor_rows']:,}"
    )
    print(
        "Potential feature exposure: "
        f"{overall['potential_feature_exposure_rows']:,}"
    )
    print(
        "Potential label exposure: "
        f"{overall['potential_label_exposure_rows']:,}"
    )
    print(
        "Potential any exposure: "
        f"{overall['potential_any_exposure_rows']:,}"
    )

    print()
    print("STATUS: RESEARCH_DIAGNOSTIC_ONLY")


if __name__ == "__main__":
    main()