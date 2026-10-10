
"""
V1.3F-10 Stage E — Numerical Masking Sensitivity Test

RESEARCH ONLY.

This experiment compares baseline model inputs against a
component-masking scenario.

Masking is NOT price correction and does NOT identify
which OHLC component is actually wrong.

No source files or production engine are modified.
No model retraining or AUC measurement.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PRICE_FILE = ROOT / "data" / "idx_stock_prices.csv"
IHSG_FILE = ROOT / "data" / "idx_ihsg_index.csv"

PROBE = ROOT / "data" / "execution_probe"
STAGE_D = PROBE / "data_quality_model_relevance_detail.csv"

DETAIL_OUT = PROBE / "data_quality_numerical_impact_detail.csv"
SUMMARY_OUT = PROBE / "data_quality_numerical_impact_summary.json"

FEATURES = [
    "ret5", "ret10", "ret20", "ret60",
    "ma20_dist", "ma50_dist", "ma200_dist",
    "vol_ratio", "rsi", "volatility20", "rs20",
]

LABELS = ["y3", "y5", "y8"]

MASK_RULES = {
    "OPEN_RANGE_VIOLATION": ["open"],
    "CLOSE_RANGE_VIOLATION": ["close"],
    "STRONG_SCALE_PATTERN_ONLY": [
        "close", "high", "low"
    ],
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_prices():
    require(
        PRICE_FILE.exists(),
        f"Missing price file: {PRICE_FILE}"
    )

    df = pd.read_csv(PRICE_FILE, parse_dates=["date"])

    required = {
        "date", "ticker", "open", "high",
        "low", "close", "volume"
    }

    require(
        required.issubset(df.columns),
        "Missing required price columns"
    )

    # Reproduce actual_engine.py ordering and deduplication.
    df = (
        df.drop_duplicates(["ticker", "date"])
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    require(
        not df[["ticker", "date"]].isna().any().any(),
        "Null ticker/date in prices"
    )

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_ihsg():
    require(
        IHSG_FILE.exists(),
        f"Missing IHSG file: {IHSG_FILE}"
    )

    idx = pd.read_csv(IHSG_FILE)

    if "Price" in idx.columns:
        idx = idx[
            pd.to_datetime(
                idx["Price"], errors="coerce"
            ).notna()
        ].copy()

        idx = idx.rename(columns={
            "Price": "date",
            "Close": "close",
        })

    elif "date" in idx.columns:
        idx = idx[
            pd.to_datetime(
                idx["date"], errors="coerce"
            ).notna()
        ].copy()

    else:
        raise RuntimeError("IHSG date column not recognized")

    rename_lower = {}

    for col in idx.columns:
        low = str(col).lower()
        if low in {"date", "close"} and col != low:
            rename_lower[col] = low

    idx = idx.rename(columns=rename_lower)

    require(
        {"date", "close"}.issubset(idx.columns),
        "Missing IHSG date/close"
    )

    idx["date"] = pd.to_datetime(
        idx["date"], errors="coerce"
    )
    idx["close"] = pd.to_numeric(
        idx["close"], errors="coerce"
    )

    idx = (
        idx[["date", "close"]]
        .dropna()
        .drop_duplicates("date")
        .sort_values("date")
    )

    return idx.set_index("date")["close"]


def load_review():
    require(
        STAGE_D.exists(),
        f"Missing Stage D detail: {STAGE_D}"
    )

    review = pd.read_csv(STAGE_D, parse_dates=["date"])

    required = {"ticker", "date", "review_class"}

    require(
        required.issubset(review.columns),
        "Stage D missing required fields"
    )

    require(
        not review.duplicated(["ticker", "date"]).any(),
        "Duplicate Stage D review keys"
    )

    require(
        review["review_class"].isin(MASK_RULES).all(),
        "Unexpected review classification"
    )

    counts = review["review_class"].value_counts()

    expected = {
        "OPEN_RANGE_VIOLATION": 78,
        "CLOSE_RANGE_VIOLATION": 3,
        "STRONG_SCALE_PATTERN_ONLY": 4,
    }

    require(
        counts.to_dict() == expected,
        "Stage D anchor counts differ from reviewed checkpoint"
    )

    return review


def make_scenario(prices, review):
    scenario = prices.copy()

    joined = scenario[["ticker", "date"]].merge(
        review[["ticker", "date", "review_class"]],
        on=["ticker", "date"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    require(
        joined["_merge"].eq("both").sum() == len(review),
        "Not all review anchors exist in price data"
    )

    scenario["mask_class"] = joined["review_class"]

    mask_counts = {}

    for category, columns in MASK_RULES.items():
        rows = scenario["mask_class"].eq(category)

        for col in columns:
            scenario.loc[rows, col] = np.nan

        mask_counts[category] = int(rows.sum())

    return scenario, mask_counts


def calculate_model_inputs(source, ihsg, mask_missing_close_label=False):
    df = source.copy()

    g = df.groupby("ticker", group_keys=False)

    for n in [5, 10, 20, 60]:
        df[f"ret{n}"] = g["close"].pct_change(n, fill_method=None)

    for n in [20, 50, 200]:
        ma = g["close"].transform(
            lambda s: s.rolling(
                n, min_periods=n
            ).mean()
        )
        df[f"ma{n}_dist"] = df["close"] / ma - 1

    df["vol20"] = g["volume"].transform(
        lambda s: s.rolling(
            20, min_periods=20
        ).mean()
    )

    df["vol_ratio"] = df["volume"] / df["vol20"]

    df["volatility20"] = g["ret5"].transform(
        lambda s: s.rolling(
            4, min_periods=4
        ).std()
    )

    delta = g["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.groupby(df["ticker"]).transform(
        lambda s: s.rolling(
            14, min_periods=14
        ).mean()
    )

    avg_loss = loss.groupby(df["ticker"]).transform(
        lambda s: s.rolling(
            14, min_periods=14
        ).mean()
    )

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (100 / (1 + rs))

    ihsg_ret20 = ihsg.pct_change(20, fill_method=None)

    df["ihsg_ret20"] = df["date"].map(ihsg_ret20)
    df["rs20"] = df["ret20"] - df["ihsg_ret20"]

    future_highs = pd.concat(
        [g["high"].shift(-i) for i in range(1, 6)],
        axis=1,
    )

    future_lows = pd.concat(
        [g["low"].shift(-i) for i in range(1, 6)],
        axis=1,
    )

    complete_horizon = (
        future_highs.notna().all(axis=1)
        & future_lows.notna().all(axis=1)
    )

    hi = future_highs.max(axis=1, skipna=False)
    lo = future_lows.min(axis=1, skipna=False)

    for target in [0.03, 0.05, 0.08]:
        label = (
            (hi >= df["close"] * (1 + target))
            & (lo > df["close"] * 0.97)
        ).astype(float)

        # Follow actual_engine.py: incomplete future
        # horizons produce unknown labels.
        label = label.where(
            complete_horizon, np.nan
        )

        # Additional scenario safety:
        # a masked current Close must not silently
        # become a negative label.
        if mask_missing_close_label:
            label = label.where(
                df["close"].notna(), np.nan
            )

        df[f"y{int(target * 100)}"] = label

    return df


def compare_column(base, scenario):
    a = pd.to_numeric(base, errors="coerce")
    b = pd.to_numeric(scenario, errors="coerce")

    a_values = a.to_numpy(dtype=float)
    b_values = b.to_numpy(dtype=float)

    both_nan = np.isnan(a_values) & np.isnan(b_values)

    # Treat non-finite results as unavailable;
    # do not turn infinity into a numeric delta.
    both_finite = (
        np.isfinite(a_values)
        & np.isfinite(b_values)
    )

    changed_numeric = (
        both_finite
        & ~np.isclose(
            a_values,
            b_values,
            rtol=1e-10,
            atol=1e-12,
        )
    )

    availability_changed = (
        ~(both_nan | both_finite)
    )

    return changed_numeric, availability_changed



def classify_numeric_precision(
    baseline,
    scenario,
    numeric_changed,
):
    """
    Separate numerically detected differences into
    near-precision differences and material candidates.

    Thresholds are research conventions, not proof
    of economic materiality.
    """
    a = pd.to_numeric(
        baseline, errors="coerce"
    ).to_numpy(dtype=float)

    b = pd.to_numeric(
        scenario, errors="coerce"
    ).to_numpy(dtype=float)

    near_precision = (
        numeric_changed
        & np.isclose(
            a,
            b,
            rtol=1e-6,
            atol=1e-9,
        )
    )

    material_candidate = (
        numeric_changed & ~near_precision
    )

    return near_precision, material_candidate

def period_of(date):
    return np.select(
        [
            date < pd.Timestamp("2023-01-01"),
            date < pd.Timestamp("2025-01-01"),
        ],
        [
            "TRAIN_PERIOD",
            "VALIDATION_PERIOD",
        ],
        default="POST_VALIDATION",
    )


def main():
    print("V1.3F-10 — Stage E Numerical Sensitivity")
    print("---------------------------------------")

    prices = load_prices()
    ihsg = load_ihsg()
    review = load_review()

    scenario, mask_counts = make_scenario(
        prices, review
    )

    baseline = calculate_model_inputs(prices, ihsg)
    simulated = calculate_model_inputs(
        scenario, ihsg, mask_missing_close_label=True
    )

    require(
        baseline[["ticker", "date"]].equals(
            simulated[["ticker", "date"]]
        ),
        "Baseline and scenario row alignment failed"
    )

    detail = baseline[["ticker", "date"]].copy()
    detail["period"] = period_of(detail["date"])

    column_stats = {}

    for col in FEATURES + LABELS:
        numeric, availability = compare_column(
           baseline[col], simulated[col]
        )
    
        near_precision, material_candidate = classify_numeric_precision(
           baseline[col],
           simulated[col],
           numeric,
        )
    
        prefix = f"impact_{col}"
    
        detail[f"{prefix}_numeric"] = numeric
        detail[f"{prefix}_near_precision"] = near_precision
        detail[f"{prefix}_material_candidate"] = material_candidate
        detail[f"{prefix}_availability"] = availability

        column_stats[col] = {
            "near_precision_rows": int(near_precision.sum()),
            "material_candidate_rows": int(material_candidate.sum()),
            "numeric_changed_rows": int(numeric.sum()),
            "availability_changed_rows": int(
                availability.sum()
            ),
            "validation_numeric_changed_rows": int(
                numeric[
                    detail["period"].eq(
                        "VALIDATION_PERIOD"
                    ).to_numpy()
                ].sum()
            ),
            "validation_availability_changed_rows": int(
                availability[
                    detail["period"].eq(
                        "VALIDATION_PERIOD"
                    ).to_numpy()
                ].sum()
            ),
        }

    numeric_columns = [
        f"impact_{col}_numeric"
        for col in FEATURES + LABELS
    ]

    availability_columns = [
        f"impact_{col}_availability"
        for col in FEATURES + LABELS
    ]

    detail["any_numeric_change"] = (
        detail[numeric_columns].any(axis=1)
    )

    detail["any_availability_change"] = (
        detail[availability_columns].any(axis=1)
    )

    detail["any_scenario_impact"] = (
        detail["any_numeric_change"]
        | detail["any_availability_change"]
    )

    periods = {}

    for period, part in detail.groupby("period"):
        periods[str(period)] = {
            "rows": int(len(part)),
            "numeric_change_rows": int(
                part["any_numeric_change"].sum()
            ),
            "availability_change_rows": int(
                part["any_availability_change"].sum()
            ),
            "any_scenario_impact_rows": int(
                part["any_scenario_impact"].sum()
            ),
        }

    summary = {
        "version": "V1.3F-10",
        "stage": "E_NUMERICAL_MASKING_SENSITIVITY",
        "status": "RESEARCH_DIAGNOSTIC_ONLY",
        "total_rows": int(len(detail)),
        "review_anchors": int(len(review)),
        "mask_counts": mask_counts,
        "column_statistics": column_stats,
        "periods": periods,
        "policy": {
            "raw_prices_modified": False,
            "actual_engine_modified": False,
            "verified_price_correction": False,
            "model_retrained": False,
            "auc_measured": False,
        },
        "limitations": [
            "Masked components are scenario assumptions, "
            "not verified faulty prices.",
            "A numeric change measures masking sensitivity, "
            "not an observed historical pricing error.",
            "Availability changes include new missing values.",
            "Labels are not model predictions.",
            "Baseline does not include the complete "
            "model-training eligibility and imputation pipeline.",
            "Masked current Close is explicitly treated "
            "as unknown for label safety.",
            "No price replacement or vendor-source "
            "reconciliation is performed.",
        ],
    }

    PROBE.mkdir(parents=True, exist_ok=True)

    detail.to_csv(
        DETAIL_OUT,
        index=False,
        date_format="%Y-%m-%d",
    )

    with open(
        SUMMARY_OUT, "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print(f"Rows: {len(detail):,}")
    print(f"Review anchors: {len(review)}")

    print()
    print("Masking scenarios:")
    for key, value in mask_counts.items():
        print(f"  {key}: {value}")

    print()
    print("Scenario impact by period:")

    for period, stats in periods.items():
        print(
            f"  {period}: "
            f"{stats['any_scenario_impact_rows']:,} "
            "rows"
        )

    print()
    print("Per-column impact:")

    for name, stats in column_stats.items():
        print(
            f"  {name}: "
            f"numeric={stats['numeric_changed_rows']:,}, "
            f"availability="
            f"{stats['availability_changed_rows']:,}"
        )

    print()
    print("STATUS: RESEARCH_DIAGNOSTIC_ONLY")


if __name__ == "__main__":
    main()
