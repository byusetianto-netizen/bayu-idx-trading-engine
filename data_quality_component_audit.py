
"""
V1.3F-10 — Stage C: Component-Level Quality Audit

RESEARCH ONLY.

Checks OHLC consistency for Stage A review anchors.
Classifies observed violations without assigning blame
to any individual price component.

Does not modify prices, labels, features, or the model.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PRICES = ROOT / "data" / "idx_stock_prices.csv"

PROBE = ROOT / "data" / "execution_probe"

STAGE_A = PROBE / "data_quality_impact_detail.csv"

DETAIL_OUT = (
    PROBE / "data_quality_component_audit_detail.csv"
)

SUMMARY_OUT = (
    PROBE / "data_quality_component_audit_summary.json"
)

COMPONENTS = ["open", "high", "low", "close"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_data():
    require(PRICES.exists(), f"Missing: {PRICES}")
    require(STAGE_A.exists(), f"Missing: {STAGE_A}")

    prices = pd.read_csv(PRICES, parse_dates=["date"])

    required = {"ticker", "date", *COMPONENTS}

    require(
        required.issubset(prices.columns),
        "Price dataset missing OHLC columns.",
    )

    # Match actual_engine.py ordering and deduplication.
    prices = (
        prices.drop_duplicates(["ticker", "date"])
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    for column in COMPONENTS:
        prices[column] = pd.to_numeric(
            prices[column],
            errors="coerce",
        )

    audit = pd.read_csv(STAGE_A, parse_dates=["date"])

    required_audit = {
        "ticker",
        "date",
        "quality_review_anchor",
        "invalid_ohlc",
        "scale_pattern_class",
    }

    require(
        required_audit.issubset(audit.columns),
        "Stage A required fields missing.",
    )

    require(
        not audit.duplicated(["ticker", "date"]).any(),
        "Duplicate Stage A keys.",
    )

    for column in ["quality_review_anchor", "invalid_ohlc"]:
        values = (
            audit[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        require(
            values.isin(["true", "false"]).all(),
            f"Invalid boolean values: {column}",
        )

        audit[column] = values.eq("true")

    merged = prices.merge(
        audit[
            [
                "ticker",
                "date",
                "quality_review_anchor",
                "invalid_ohlc",
                "scale_pattern_class",
            ]
        ],
        on=["ticker", "date"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    require(
        merged["_merge"].eq("both").all(),
        "Stage A coverage does not match prices.",
    )

    require(
        len(merged) == len(audit),
        "Stage A and price row counts differ.",
    )

    return merged.drop(columns="_merge")


def classify_components(df):
    result = df.copy()

    o = result["open"]
    h = result["high"]
    l = result["low"]
    c = result["close"]

    # Use direct comparisons, without changing prices.
    result["missing_ohlc"] = (
        result[COMPONENTS].isna().any(axis=1)
    )

    result["nonpositive_ohlc"] = (
        result[COMPONENTS].le(0).any(axis=1)
    )

    result["low_above_high"] = l > h
    result["open_below_low"] = o < l
    result["open_above_high"] = o > h
    result["close_below_low"] = c < l
    result["close_above_high"] = c > h

    violation_columns = [
        "missing_ohlc",
        "nonpositive_ohlc",
        "low_above_high",
        "open_below_low",
        "open_above_high",
        "close_below_low",
        "close_above_high",
    ]

    result["computed_ohlc_violation"] = (
        result[violation_columns].any(axis=1)
    )

    # This is a descriptive classification only.
    # It does NOT identify which price is incorrect.
    result["observed_violation_types"] = (
        result[violation_columns]
        .apply(
            lambda row: "|".join(
                name
                for name, flagged in row.items()
                if flagged
            ) or "NONE",
            axis=1,
        )
    )

    result["stage_a_invalid_vs_computed_match"] = (
        result["invalid_ohlc"]
        == result["computed_ohlc_violation"]
    )

    result["evidence_status"] = np.select(
        [
            result["computed_ohlc_violation"],
            result["scale_pattern_class"].eq(
                "STRONG_SCALE_PATTERN"
            ),
        ],
        [
            "OHLC_INCONSISTENCY_OBSERVED",
            "SCALE_PATTERN_REVIEW_REQUIRED",
        ],
        default="OTHER_REVIEW_REQUIRED",
    )

    # We cannot infer whether Open, High, Low, or Close
    # is wrong from an OHLC constraint violation alone.
    result["component_fault_confirmed"] = False
    result["correction_authorized"] = False

    return result


def main():
    print("V1.3F-10 — Stage C Component Quality Audit")
    print("-----------------------------------------")

    data = load_data()

    review = data[
        data["quality_review_anchor"]
    ].copy()

    require(
        len(review) > 0,
        "No review anchors in Stage A.",
    )

    result = classify_components(review)

    violation_columns = [
        "missing_ohlc",
        "nonpositive_ohlc",
        "low_above_high",
        "open_below_low",
        "open_above_high",
        "close_below_low",
        "close_above_high",
    ]

    violation_counts = {
        column: int(result[column].sum())
        for column in violation_columns
    }

    summary = {
        "version": "V1.3F-10",
        "stage": "C_COMPONENT_LEVEL_QUALITY_AUDIT",
        "status": "RESEARCH_DIAGNOSTIC_ONLY",
        "review_anchor_rows": int(len(result)),
        "stage_a_invalid_ohlc_rows": int(
            result["invalid_ohlc"].sum()
        ),
        "computed_ohlc_violation_rows": int(
            result["computed_ohlc_violation"].sum()
        ),
        "stage_a_classification_mismatch_rows": int(
            (
                ~result[
                    "stage_a_invalid_vs_computed_match"
                ]
            ).sum()
        ),
        "violation_counts": violation_counts,
        "evidence_status_counts": {
            str(k): int(v)
            for k, v in (
                result["evidence_status"]
                .value_counts()
                .items()
            )
        },
        "policy": {
            "original_prices_modified": False,
            "actual_engine_modified": False,
            "individual_fault_assigned": False,
            "automatic_price_correction": False,
            "model_retrained": False,
        },
        "limitations": [
            "OHLC inconsistency does not identify "
            "which price component is incorrect.",
            "A scale pattern alone does not prove "
            "a vendor adjustment error.",
            "This audit checks only Stage A review anchors.",
            "Stage A classification may use rules not "
            "identical to this script.",
            "No independent verified OHLC series is used.",
            "No numerical feature, label or AUC impact "
            "is measured.",
        ],
    }

    PROBE.mkdir(parents=True, exist_ok=True)

    result[
        [
            "ticker",
            "date",
            *COMPONENTS,
            "invalid_ohlc",
            "scale_pattern_class",
            *violation_columns,
            "computed_ohlc_violation",
            "observed_violation_types",
            "stage_a_invalid_vs_computed_match",
            "evidence_status",
            "component_fault_confirmed",
            "correction_authorized",
        ]
    ].to_csv(
        DETAIL_OUT,
        index=False,
        date_format="%Y-%m-%d",
    )

    with open(
        SUMMARY_OUT,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print(
        "Review anchors:",
        summary["review_anchor_rows"],
    )
    print(
        "Stage A invalid OHLC:",
        summary["stage_a_invalid_ohlc_rows"],
    )
    print(
        "Computed OHLC violations:",
        summary["computed_ohlc_violation_rows"],
    )
    print(
        "Classification mismatches:",
        summary["stage_a_classification_mismatch_rows"],
    )

    print()
    print("Violation types:")

    for name, count in violation_counts.items():
        print(f"  {name}: {count}")

    print()
    print("STATUS: RESEARCH_DIAGNOSTIC_ONLY")


if __name__ == "__main__":
    main()
