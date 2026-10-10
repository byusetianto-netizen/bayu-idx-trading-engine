"""
V1.3F-9 Adjusted-Price Discontinuity Audit

Research-only diagnostic.
No raw data modifications.
No automatic corporate-action attribution.
No production engine changes.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PRICE_FILE = ROOT / "data" / "idx_stock_prices.csv"

ELIGIBILITY_FILE = (
    ROOT
    / "data"
    / "execution_probe"
    / "research_session_eligibility.csv"
)

OUT_DIR = ROOT / "data" / "execution_probe"

DETAIL_FILE = OUT_DIR / "adjusted_price_discontinuity_detail.csv"
SUMMARY_FILE = OUT_DIR / "adjusted_price_discontinuity_summary.json"

EXTREME_RETURN = 0.50
REVERSAL_RETURN = 0.40
ROUNDTRIP_TOLERANCE = 0.15


def fail(message):
    raise RuntimeError(message)


def load_prices():
    if not PRICE_FILE.exists():
        fail(f"Missing price file: {PRICE_FILE}")

    df = pd.read_csv(PRICE_FILE)

    required = {
        "date", "ticker", "open", "high",
        "low", "close", "volume"
    }

    missing = required - set(df.columns)

    if missing:
        fail(f"Missing price columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    )

    df["ticker"] = (
        df["ticker"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if df["date"].isna().any():
        fail("Invalid price dates.")

    if df["ticker"].isna().any() or (df["ticker"] == "").any():
        fail("Invalid tickers.")

    if df.duplicated(["ticker", "date"]).any():
        fail("Duplicate ticker/date observations.")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    return (
        df.sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )


def load_eligibility():
    if not ELIGIBILITY_FILE.exists():
        fail(
            "Missing eligibility artifact. Run "
            "research_session_eligibility.py first."
        )

    e = pd.read_csv(ELIGIBILITY_FILE)

    required = {
        "date", "ticker", "source_universe",
        "placeholder_candidate"
    }

    missing = required - set(e.columns)

    if missing:
        fail(
            f"Missing eligibility columns: {sorted(missing)}"
        )

    e = e.loc[
        e["source_universe"] == "BASE_95"
    ].copy()

    e["date"] = pd.to_datetime(
        e["date"], errors="coerce"
    )

    e["ticker"] = (
        e["ticker"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if e["date"].isna().any():
        fail("Invalid eligibility dates.")

    if e.duplicated(["ticker", "date"]).any():
        fail("Duplicate eligibility ticker/date keys.")

    values = (
        e["placeholder_candidate"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    if not values.isin(["true", "false", "1", "0"]).all():
        fail("Invalid placeholder_candidate values.")

    e["placeholder_candidate"] = values.isin(
        ["true", "1"]
    )

    return e[
        ["ticker", "date", "placeholder_candidate"]
    ]


def run_audit(df):
    x = df.copy()

    g = x.groupby("ticker", group_keys=False)

    x["previous_date"] = g["date"].shift(1)
    x["next_date"] = g["date"].shift(-1)

    x["previous_close"] = g["close"].shift(1)
    x["next_close"] = g["close"].shift(-1)

    x["previous_volume"] = g["volume"].shift(1)
    x["next_volume"] = g["volume"].shift(-1)

    prev = x["previous_close"]
    curr = x["close"]
    nxt = x["next_close"]

    # Returns are meaningful only when both prices are positive.
    valid_prev = (
        np.isfinite(prev)
        & np.isfinite(curr)
        & (prev > 0)
        & (curr > 0)
    )

    valid_next = (
        np.isfinite(curr)
        & np.isfinite(nxt)
        & (curr > 0)
        & (nxt > 0)
    )

    x["return_from_previous"] = (
        (curr / prev - 1).where(valid_prev)
    )

    x["return_to_next"] = (
        (nxt / curr - 1).where(valid_next)
    )

    valid_triplet = (
        valid_prev
        & valid_next
        & np.isfinite(nxt)
        & (nxt > 0)
    )

    x["roundtrip_deviation"] = (
        (nxt / prev - 1).abs().where(valid_triplet)
    )

    x["invalid_close"] = (
        ~np.isfinite(curr) | (curr <= 0)
    )

    x["extreme_return"] = (
        x["return_from_previous"].abs()
        >= EXTREME_RETURN
    ).fillna(False)

    r1 = x["return_from_previous"]
    r2 = x["return_to_next"]

    opposite_direction = (
        ((r1 > 0) & (r2 < 0))
        | ((r1 < 0) & (r2 > 0))
    )

    x["sharp_reversal"] = (
        opposite_direction
        & (r1.abs() >= EXTREME_RETURN)
        & (r2.abs() >= REVERSAL_RETURN)
    ).fillna(False)

    x["near_roundtrip"] = (
        x["sharp_reversal"]
        & (
            x["roundtrip_deviation"]
            <= ROUNDTRIP_TOLERANCE
        )
    ).fillna(False)

    # Positive and finite OHLC required for this comparison.
    ohlc = x[["open", "high", "low", "close"]]

    x["invalid_ohlc"] = (
        ~np.isfinite(ohlc).all(axis=1)
        | (ohlc <= 0).any(axis=1)
        | (x["high"] < x["low"])
        | (x["high"] < x["open"])
        | (x["high"] < x["close"])
        | (x["low"] > x["open"])
        | (x["low"] > x["close"])
    )

    x["zero_volume"] = (
        x["volume"].notna()
        & (x["volume"] == 0)
    )

    x["audit_flag"] = (
        x["invalid_close"]
        | x["invalid_ohlc"]
        | x["extreme_return"]
        | x["sharp_reversal"]
    )

    x["severity"] = np.select(
        [
            x["invalid_close"] | x["invalid_ohlc"],
            x["near_roundtrip"],
            x["sharp_reversal"],
            x["extreme_return"],
        ],
        [
            "INVALID_PRICE_OR_OHLC",
            "EXTREME_NEAR_ROUNDTRIP",
            "SHARP_REVERSAL",
            "EXTREME_SINGLE_RETURN",
        ],
        default="NOT_FLAGGED",
    )

    return x


def detect_multi_session_scale_reversal(df, max_lookahead=5):
    """
    Detect extreme price-scale changes followed by recovery
    within the next max_lookahead observations.

    Research diagnostic only.
    No automatic correction or corporate-action attribution.
    """

    x = df.copy()

    x["scale_reversal_candidate"] = False
    x["scale_reversal_lag"] = np.nan
    x["scale_reversal_ratio"] = np.nan
    x["scale_reversal_recovery_deviation"] = np.nan
    x["scale_reversal_zero_volume_between"] = np.nan

    close = x["close"]
    ticker = x["ticker"]

    start = x["extreme_return"] & close.gt(0)

    for lag in range(1, max_lookahead + 1):
        future_close = close.groupby(ticker).shift(-lag)
        future_ticker = ticker.shift(-lag)

        valid = (
            start
            & ticker.eq(future_ticker)
            & x["previous_close"].gt(0)
            & future_close.gt(0)
            & ~x["scale_reversal_candidate"]
        )

        ratio = close / x["previous_close"]

        recovery_deviation = (
            future_close / x["previous_close"] - 1
        ).abs()

        candidate = (
            valid
            & recovery_deviation.le(ROUNDTRIP_TOLERANCE)
        )

        if not candidate.any():
            continue

        x.loc[candidate, "scale_reversal_candidate"] = True
        x.loc[candidate, "scale_reversal_lag"] = lag

        x.loc[candidate, "scale_reversal_ratio"] = (
            ratio.loc[candidate]
        )

        x.loc[
            candidate,
            "scale_reversal_recovery_deviation"
        ] = recovery_deviation.loc[candidate]

        # Count zero-volume observations strictly between
        # the initiating event and the recovery observation.
        zero_between = pd.Series(
            0,
            index=x.index,
            dtype=int,
        )

        for offset in range(1, lag):
            future_zero = (
                x["volume"]
                .groupby(ticker)
                .shift(-offset)
                .eq(0)
            )

            zero_between += future_zero.astype(int)

        x.loc[
            candidate,
            "scale_reversal_zero_volume_between"
        ] = zero_between.loc[candidate]

    return x


def classify_scale_reversal(df):
    """
    Classify detected scale-reversal patterns.

    STRONG_SCALE_PATTERN:
      Ratio near 2x, 3x, 4x, or 5x (or reciprocals),
      with recovery deviation <= 7%.

    POSSIBLE_REVERSAL:
      Recovery detected, but strong-pattern criteria not met.

    These are diagnostic classifications, not probabilities
    or confirmed corporate actions.
    """
    x = df.copy()

    x["scale_pattern_class"] = "NOT_DETECTED"
    x["nearest_scale_factor"] = np.nan
    x["scale_factor_deviation"] = np.nan

    factors = np.array([
        2.0, 3.0, 4.0, 5.0,
        1 / 2, 1 / 3, 1 / 4, 1 / 5,
    ])

    candidate = x["scale_reversal_candidate"].fillna(False)

    if not candidate.any():
        return x

    ratio = x.loc[candidate, "scale_reversal_ratio"].to_numpy(
        dtype=float
    )

    # Relative distance between observed ratio and candidate factors.
    deviations = np.abs(
        ratio[:, None] / factors[None, :] - 1
    )

    nearest_index = deviations.argmin(axis=1)

    x.loc[candidate, "nearest_scale_factor"] = factors[
        nearest_index
    ]

    x.loc[candidate, "scale_factor_deviation"] = deviations[
        np.arange(len(ratio)),
        nearest_index,
    ]

    x.loc[candidate, "scale_pattern_class"] = "POSSIBLE_REVERSAL"

    strong = (
        candidate
        & x["scale_factor_deviation"].le(0.05)
        & x["scale_reversal_recovery_deviation"].le(0.07)
    )

    x.loc[strong, "scale_pattern_class"] = "STRONG_SCALE_PATTERN"

    return x


def main():
    print("V1.3F-9 Adjusted-Price Discontinuity Audit")
    print("-----------------------------------------")

    prices = load_prices()
    eligibility = load_eligibility()

    joined = prices.merge(
        eligibility,
        on=["ticker", "date"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = joined["_merge"] != "both"

    if unmatched.any():
        examples = (
            joined.loc[unmatched, ["ticker", "date"]]
            .head(5)
            .astype(str)
            .to_dict("records")
        )
        fail(
            "Missing BASE_95 eligibility rows. "
            f"Examples: {examples}"
        )

    joined = joined.drop(columns="_merge")

    audited = run_audit(joined)

    audited = detect_multi_session_scale_reversal(
        audited,
        max_lookahead=5,
    )
    audited = classify_scale_reversal(audited)

    flagged = audited.loc[
        audited["audit_flag"]
    ].copy()

    severity_counts = (
        flagged["severity"]
        .value_counts()
        .to_dict()
    )

    ticker_counts = (
        flagged.groupby("ticker")
        .size()
        .sort_values(ascending=False)
    )
    # Summarize TOWR separately as a known investigation case.
    towr = audited.loc[
        audited["ticker"] == "TOWR"
    ].copy()

    towr_flagged = towr.loc[
        towr["audit_flag"]
    ].copy()

    summary = {
        "version": "V1.3F-9",
        "status": "DIAGNOSTIC_COMPLETE_NO_PRODUCTION_CHANGE",
        "policy": {
            "raw_data_modified": False,
            "automatic_price_repair": False,
            "actual_engine_modified": False,
            "corporate_action_verified": False,
            "official_trading_calendar_verified": False,
        },
        "thresholds": {
            "extreme_absolute_return": EXTREME_RETURN,
            "sharp_reversal_absolute_return": REVERSAL_RETURN,
            "near_roundtrip_tolerance": ROUNDTRIP_TOLERANCE,
        },
        "dataset": {
            "rows": int(len(audited)),
            "unique_tickers": int(audited["ticker"].nunique()),
            "unique_dates": int(audited["date"].nunique()),
            "placeholder_candidate_rows": int(
                audited["placeholder_candidate"].sum()
            ),
        },
        "audit": {
            "flagged_rows": int(len(flagged)),
            "flagged_tickers": int(
                flagged["ticker"].nunique()
            ),
            "invalid_close_rows": int(
                audited["invalid_close"].sum()
            ),
            "invalid_ohlc_rows": int(
                audited["invalid_ohlc"].sum()
            ),
            "extreme_return_rows": int(
                audited["extreme_return"].sum()
            ),
            "sharp_reversal_rows": int(
                audited["sharp_reversal"].sum()
            ),
            "near_roundtrip_rows": int(
                audited["near_roundtrip"].sum()
            ),
	    "multi_session_scale_reversal_rows": int(
                audited["scale_reversal_candidate"].sum()
            ),
            "multi_session_scale_reversal_tickers": int(
                audited.loc[
                    audited["scale_reversal_candidate"],
                    "ticker",
                ].nunique()
            ),
            "scale_pattern_class_counts": {
                str(k): int(v)
                for k, v in (
                    audited.loc[
                        audited["scale_reversal_candidate"],
                        "scale_pattern_class",
                    ]
                    .value_counts()
                    .items()
                )
            },
            "flagged_placeholder_rows": int(
                flagged["placeholder_candidate"].sum()
            ),
            "flagged_zero_volume_rows": int(
                flagged["zero_volume"].sum()
            ),
            "severity_counts": {
                str(k): int(v)
                for k, v in severity_counts.items()
            },
        },
        "top_flagged_tickers": [
            {
                "ticker": str(ticker),
                "flagged_rows": int(count),
            }
            for ticker, count in ticker_counts.head(20).items()
        ],
        "t o w r": {
            "observations": int(len(towr)),
            "flagged_rows": int(len(towr_flagged)),
            "extreme_return_rows": int(
                towr["extreme_return"].sum()
            ),
            "sharp_reversal_rows": int(
                towr["sharp_reversal"].sum()
            ),
            "near_roundtrip_rows": int(
                towr["near_roundtrip"].sum()
            ),
        },
        "limitations": [
            "An extreme return is not proof of erroneous data.",
            "Corporate actions are not independently verified.",
            "Placeholder candidates are not proven holidays.",
            "Adjusted prices are not official execution prices.",
            "No price rows are deleted or automatically repaired.",
        ],
    }

    # Use an ordinary JSON key for the TOWR investigation.
    summary["TOWR"] = summary.pop("t o w r")

    detail_columns = [
        "ticker",
        "date",
        "previous_date",
        "next_date",
        "previous_close",
        "open",
        "high",
        "low",
        "close",
        "next_close",
        "volume",
        "previous_volume",
        "next_volume",
        "return_from_previous",
        "return_to_next",
        "roundtrip_deviation",
        "placeholder_candidate",
        "zero_volume",
        "invalid_close",
        "invalid_ohlc",
        "extreme_return",
        "sharp_reversal",
        "near_roundtrip",
        "scale_reversal_candidate",
        "scale_reversal_lag",
        "scale_reversal_ratio",
        "scale_reversal_recovery_deviation",
        "scale_reversal_zero_volume_between",
        "scale_pattern_class",
        "nearest_scale_factor",
        "scale_factor_deviation",
        "severity",
    ]

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    flagged[detail_columns].to_csv(
        DETAIL_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(
            summary,
            fh,
            indent=2,
        )

    print(f"Audited rows: {len(audited):,}")
    print(f"Flagged rows: {len(flagged):,}")
    print(
        "Flagged tickers: "
        f"{flagged['ticker'].nunique():,}"
    )
    print(
        "Extreme returns: "
        f"{int(audited['extreme_return'].sum()):,}"
    )
    print(
        "Sharp reversals: "
        f"{int(audited['sharp_reversal'].sum()):,}"
    )
    print(
        "Near roundtrips: "
        f"{int(audited['near_roundtrip'].sum()):,}"
    )

    print()
    print("Top flagged tickers")
    print("-------------------")
    print(ticker_counts.head(20).to_string())

    print()
    print("TOWR investigation")
    print("------------------")
    print(
        f"TOWR flagged rows: {len(towr_flagged):,}"
    )

    print()
    print("STATUS: DIAGNOSTIC_COMPLETE_NO_PRODUCTION_CHANGE")


if __name__ == "__main__":
    main()
    