
"""
V1.3F-10 — Stage B: Exact Structural Dependency Audit

RESEARCH ONLY.

Identifies potential dependency paths from reviewed OHLC
observations to the 11 model features and 3 forward labels.

Does NOT:
- prove that an OHLC component is erroneous
- calculate actual changes in feature/label values
- correct prices or remove observations
- retrain or modify actual_engine.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "idx_stock_prices.csv"
STAGE_A = (
    ROOT / "data" / "execution_probe"
    / "data_quality_impact_detail.csv"
)
OUT = ROOT / "data" / "execution_probe"

DETAIL_OUT = OUT / "data_quality_exact_dependency_detail.csv"
SUMMARY_OUT = OUT / "data_quality_exact_dependency_summary.json"

FEATURE_OFFSETS = {
    # A close at position a influences a and a+n
    # for pct_change(n).
    "ret5": [0, 5],
    "ret10": [0, 10],
    "ret20": [0, 20],
    "ret60": [0, 60],

    # Rolling average includes positions a to a+n-1.
    # Close is also the numerator at position a.
    "ma20_dist": list(range(20)),
    "ma50_dist": list(range(50)),
    "ma200_dist": list(range(200)),

    # Volume ratio has no direct close/high/low input.
    "vol_ratio": [],

    # RSI uses 14 close differences.
    # Each difference depends on two close observations.
    "rsi": list(range(15)),

    # Rolling std of 4 ret5 observations.
    # Structural offsets from both ends of pct_change(5).
    "volatility20": sorted(
        set(range(4)) | set(range(5, 9))
    ),

    # Stock return ret20 minus independent IHSG return.
    "rs20": [0, 20],
}

LABELS = ["y3", "y5", "y8"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_inputs():
    require(DATA.exists(), f"Missing prices: {DATA}")
    require(STAGE_A.exists(), f"Missing Stage A: {STAGE_A}")

    prices = pd.read_csv(DATA, parse_dates=["date"])

    required_prices = {
        "ticker", "date", "open", "high",
        "low", "close", "volume",
    }
    require(
        required_prices.issubset(prices.columns),
        "Missing expected price columns.",
    )

    # Match baseline ordering/deduplication.
    prices = (
        prices.drop_duplicates(["ticker", "date"])
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    for column in ["open", "high", "low", "close", "volume"]:
        prices[column] = pd.to_numeric(
            prices[column], errors="coerce"
        )

    audit = pd.read_csv(STAGE_A, parse_dates=["date"])

    require(
        {
            "ticker", "date", "quality_review_anchor",
        }.issubset(audit.columns),
        "Stage A required columns missing.",
    )

    require(
        not audit.duplicated(["ticker", "date"]).any(),
        "Duplicate Stage A keys.",
    )

    anchor_values = (
        audit["quality_review_anchor"]
        .astype("string")
        .str.lower()
    )

    require(
        anchor_values.isin(["true", "false"]).all(),
        "Invalid Stage A anchor flags.",
    )

    audit["quality_review_anchor"] = (
        anchor_values.eq("true")
    )

    merged = prices.merge(
        audit[["ticker", "date", "quality_review_anchor"]],
        on=["ticker", "date"],
        how="left",
        validate="one_to_one",
    )

    require(
        merged["quality_review_anchor"].notna().all(),
        "Stage A does not cover every baseline price row.",
    )

    merged["quality_review_anchor"] = (
        merged["quality_review_anchor"].astype(bool)
    )

    return merged


def period_of(dates):
    return np.select(
        [
            dates < pd.Timestamp("2023-01-01"),
            dates < pd.Timestamp("2025-01-01"),
        ],
        [
            "TRAIN_PERIOD",
            "VALIDATION_PERIOD",
        ],
        default="POST_VALIDATION",
    )


def analyze(df):
    n = len(df)

    # Masks represent possible structural dependencies.
    # Every anchor is treated as an OHLC review point,
    # not as a proven incorrect Close, High or Low.
    feature_masks = {
        name: np.zeros(n, dtype=bool)
        for name in FEATURE_OFFSETS
    }

    close_label_mask = np.zeros(n, dtype=bool)
    future_hilo_label_mask = np.zeros(n, dtype=bool)

    # Work within each ticker to prevent cross-ticker leakage.
    for _, group in df.groupby("ticker", sort=False):
        positions = group.index.to_numpy()
        local_anchors = np.flatnonzero(
            group["quality_review_anchor"].to_numpy()
        )
        size = len(positions)

        for a in local_anchors:

            # Close dependencies for model features.
            for name, offsets in FEATURE_OFFSETS.items():
                for offset in offsets:
                    j = a + offset
                    if j < size:
                        feature_masks[name][positions[j]] = True

            # Label on current row references current Close.
            close_label_mask[positions[a]] = True

            # Prior five labels reference future High/Low.
            for lag in range(1, 6):
                j = a - lag
                if j >= 0:
                    future_hilo_label_mask[positions[j]] = True

    result = df[["ticker", "date", "quality_review_anchor"]].copy()

    for name, mask in feature_masks.items():
        result[f"dep_{name}"] = mask

    result["dep_label_close"] = close_label_mask
    result["dep_label_future_hilo"] = (
        future_hilo_label_mask
    )

    feature_columns = [
        f"dep_{name}" for name in FEATURE_OFFSETS
    ]

    result["feature_dependency_count"] = (
        result[feature_columns].sum(axis=1)
    )

    result["any_feature_dependency"] = (
        result[feature_columns].any(axis=1)
    )

    result["any_label_dependency"] = (
        result["dep_label_close"]
        | result["dep_label_future_hilo"]
    )

    result["any_dependency"] = (
        result["any_feature_dependency"]
        | result["any_label_dependency"]
    )

    result["period"] = period_of(result["date"])

    # Formula requires all five next High and Low values.
    g = df.groupby("ticker", sort=False)
    high_future = pd.concat(
        [g["high"].shift(-i) for i in range(1, 6)],
        axis=1,
    )
    low_future = pd.concat(
        [g["low"].shift(-i) for i in range(1, 6)],
        axis=1,
    )

    result["complete_label_horizon"] = (
        high_future.notna().all(axis=1)
        & low_future.notna().all(axis=1)
    )

    result["eligible_label_dependency"] = (
        result["any_label_dependency"]
        & result["complete_label_horizon"]
        & df["close"].notna()
    )

    return result


def summarize(result):
    def metrics(part):
        values = {
            "rows": int(len(part)),
            "review_anchors": int(
                part["quality_review_anchor"].sum()
            ),
            "any_feature_dependency_rows": int(
                part["any_feature_dependency"].sum()
            ),
            "any_label_dependency_rows": int(
                part["any_label_dependency"].sum()
            ),
            "eligible_label_dependency_rows": int(
                part["eligible_label_dependency"].sum()
            ),
            "any_dependency_rows": int(
                part["any_dependency"].sum()
            ),
        }

        values["feature_rows"] = {
            name: int(part[f"dep_{name}"].sum())
            for name in FEATURE_OFFSETS
        }

        values["label_source_rows"] = {
            "close": int(
                part["dep_label_close"].sum()
            ),
            "future_high_or_low": int(
                part["dep_label_future_hilo"].sum()
            ),
        }

        return values

    periods = {
        str(period): metrics(part)
        for period, part in result.groupby("period")
    }

    validation = result[
        (result["period"] == "VALIDATION_PERIOD")
        & result["any_dependency"]
    ]

    validation_tickers = {
        str(ticker): {
            "rows": int(len(part)),
            "first_date": str(part["date"].min().date()),
            "last_date": str(part["date"].max().date()),
        }
        for ticker, part in validation.groupby("ticker")
    }

    return {
        "version": "V1.3F-10",
        "stage": "B_EXACT_STRUCTURAL_DEPENDENCY",
        "status": "RESEARCH_DIAGNOSTIC_ONLY",
        "overall": metrics(result),
        "periods": periods,
        "validation_tickers": validation_tickers,
        "label_targets": LABELS,
        "policy": {
            "raw_prices_modified": False,
            "actual_engine_modified": False,
            "prices_corrected": False,
            "model_retrained": False,
            "numerical_impact_confirmed": False,
        },
        "limitations": [
            "An OHLC violation does not identify which component is wrong.",
            "Dependencies are structural possibilities, not numeric changes.",
            "Feature calculation validity and missing values are not "
            "fully evaluated by this audit.",
            "All three labels share the same five-session dependency window.",
            "No alternative verified price series is used.",
            "This is not a backtest or an AUC experiment.",
        ],
    }


def main():
    print("V1.3F-10 — Stage B Exact Dependency Audit")
    print("----------------------------------------")

    prices = load_inputs()
    result = analyze(prices)
    summary = summarize(result)

    OUT.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        DETAIL_OUT,
        index=False,
        date_format="%Y-%m-%d",
    )

    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    overall = summary["overall"]

    print(f"Rows: {overall['rows']:,}")
    print(f"Review anchors: {overall['review_anchors']:,}")
    print(
        "Any feature dependency: "
        f"{overall['any_feature_dependency_rows']:,}"
    )
    print(
        "Any label dependency: "
        f"{overall['any_label_dependency_rows']:,}"
    )
    print(
        "Any dependency: "
        f"{overall['any_dependency_rows']:,}"
    )

    print()
    print("Validation dependencies by ticker:")

    for ticker, info in summary["validation_tickers"].items():
        print(
            f"{ticker}: {info['rows']:,} rows "
            f"({info['first_date']} to {info['last_date']})"
        )

    print()
    print("STATUS: RESEARCH_DIAGNOSTIC_ONLY")


if __name__ == "__main__":
    main()
