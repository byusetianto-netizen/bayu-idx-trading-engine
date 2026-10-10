"""
V1.3F-9 Price Scale Evidence Summary

Reproduce relative historical price-scale differences
using recorded external price evidence.

Research only. No price corrections or production changes.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "execution_probe"

INPUT_FILE = (
    OUT_DIR / "independent_price_verification_detail.csv"
)

OUTPUT_FILE = (
    OUT_DIR / "price_scale_evidence_summary.json"
)


# Dates and comparison groups selected from prior
# anomaly investigation, not optimized to fit a result.
CASES = {
    "NISP": {
        "low_dates": ["2018-04-18"],
        "high_dates": ["2018-04-23"],
        "reference_factor": 2.0,
    },
    "TOWR": {
        "low_dates": [
            "2018-05-02",
            "2018-05-04",
        ],
        "high_dates": [
            "2018-05-03",
            "2018-05-07",
        ],
        "reference_factor": 5.0,
    },
}


def fail(message):
    raise RuntimeError(message)


def calculate_case(df, ticker, config):
    dates = (
        config["low_dates"]
        + config["high_dates"]
    )

    subset = df.loc[
        df["ticker"].eq(ticker)
        & df["date"].isin(dates)
    ].copy()

    if len(subset) != len(dates):
        fail(
            f"{ticker}: expected {len(dates)} "
            f"observations, found {len(subset)}."
        )

    if subset["date"].duplicated().any():
        fail(f"{ticker}: duplicate evidence dates.")

    if not set(subset["date"]) == set(dates):
        fail(f"{ticker}: evidence dates do not match.")

    if (
        subset["external_close"].isna().any()
        or subset["internal_close"].isna().any()
        or (subset["external_close"] <= 0).any()
        or (subset["internal_close"] <= 0).any()
    ):
        fail(f"{ticker}: missing or invalid prices.")

    if not subset["price_type"].eq("UNKNOWN").all():
        fail(
            f"{ticker}: unexpected external price type. "
            "Review evidence before proceeding."
        )

    if (
        subset["source_name"].isna().any()
        or subset["source_url"].isna().any()
        or (
            subset["source_name"].str.strip() == ""
        ).any()
        or (
            subset["source_url"].str.strip() == ""
        ).any()
    ):
        fail(f"{ticker}: missing source metadata.")

    if not np.isfinite(subset["close_ratio"]).all():
        fail(f"{ticker}: invalid recorded price ratios.")

    recalculated = (
        subset["internal_close"]
        / subset["external_close"]
    )

    if not np.allclose(
        recalculated,
        subset["close_ratio"],
        rtol=1e-10,
        atol=1e-12,
    ):
        fail(
            f"{ticker}: recorded ratios do not "
            "match underlying prices."
        )

    low = subset.loc[
        subset["date"].isin(config["low_dates"]),
        "close_ratio",
    ]

    high = subset.loc[
        subset["date"].isin(config["high_dates"]),
        "close_ratio",
    ]

    low_mean = float(low.mean())
    high_mean = float(high.mean())

    factor = high_mean / low_mean
    reference = config["reference_factor"]

    deviation_pct = (
        factor / reference - 1
    ) * 100

    records = (
        subset[
            [
                "date",
                "internal_close",
                "external_close",
                "close_ratio",
                "price_type",
                "source_name",
                "source_url",
            ]
        ]
        .sort_values("date")
        .to_dict("records")
    )

    return {
        "ticker": ticker,
        "low_group_dates": config["low_dates"],
        "high_group_dates": config["high_dates"],
        "low_group_mean_ratio": low_mean,
        "high_group_mean_ratio": high_mean,
        "relative_scale_factor": float(factor),
        "reference_factor": reference,
        "deviation_from_reference_percent": float(
            deviation_pct
        ),
        "evidence_observations": int(len(subset)),
        "observations": records,
        "status": (
            "SCALE_FACTOR_DISCONTINUITY_SUPPORTED"
            "__ADJUSTMENT_BASIS_UNVERIFIED"
        ),
        "correction_authorized": False,
    }


def main():
    print("V1.3F-9 Price Scale Evidence Summary")
    print("-----------------------------------")

    if not INPUT_FILE.exists():
        fail(
            "Missing verification detail. Run "
            "independent_price_verification.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "ticker",
        "date",
        "internal_close",
        "external_close",
        "close_ratio",
        "price_type",
        "source_name",
        "source_url",
    }

    missing = required - set(df.columns)

    if missing:
        fail(f"Missing columns: {sorted(missing)}")

    for col in [
        "internal_close",
        "external_close",
        "close_ratio",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    results = {}

    for ticker, config in CASES.items():
        results[ticker] = calculate_case(
            df, ticker, config
        )

    summary = {
        "version": "V1.3F-9",
        "stage": "PRICE_SCALE_EVIDENCE_SUMMARY",
        "results": results,
        "policy": {
            "external_adjustment_basis_verified": False,
            "source_independence_authenticated": False,
            "corporate_action_causation_verified": False,
            "raw_prices_modified": False,
            "automatic_price_repair": False,
            "production_engine_modified": False,
        },
        "limitations": [
            "Date groups are preselected investigation cases.",
            "Reference factors are diagnostic hypotheses.",
            "A matching ratio does not establish causation.",
            "External historical price adjustment basis "
            "has not been verified.",
            "No price correction is authorized.",
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            allow_nan=False,
        )

    for ticker, result in results.items():
        print()
        print(f"{ticker}")
        print(
            "Relative scale factor: "
            f"{result['relative_scale_factor']:.9f}x"
        )
        print(
            "Reference factor: "
            f"{result['reference_factor']:.1f}x"
        )
        print(
            "Deviation (%): "
            f"{result['deviation_from_reference_percent']:.6f}"
        )

    print()
    print(
        "STATUS: SCALE_FACTOR_DISCONTINUITY_SUPPORTED"
        "__ADJUSTMENT_BASIS_UNVERIFIED"
    )


if __name__ == "__main__":
    main()