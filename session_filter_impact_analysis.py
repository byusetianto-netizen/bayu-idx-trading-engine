"""
V1.3F-6 Session-Filter Impact Analysis

Compare the current base-price session semantics against a research
view that excludes V1.3F-4 PLACEHOLDER_CANDIDATE observations before
ordered-session feature and forward-label calculations.

This is diagnostic only:
- Does NOT modify raw price data.
- Does NOT modify actual_engine.py.
- Does NOT retrain any model.
- Does NOT claim placeholder candidates are proven IDX non-trading days.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PROBE = DATA / "execution_probe"

PRICE = DATA / "idx_stock_prices.csv"
ELIGIBILITY = PROBE / "research_session_eligibility.csv"

OUT_DETAIL = PROBE / "session_filter_impact_detail.csv"
OUT_SUMMARY = PROBE / "session_filter_impact_summary.json"

KEY = ["ticker", "date"]

PRICE_REQUIRED = {
    "date", "ticker", "open", "high", "low", "close", "volume"
}

ELIG_REQUIRED = {
    "date",
    "ticker",
    "source_universe",
    "placeholder_candidate",
    "research_session_eligible",
}


def fail(message):
    raise RuntimeError(message)


def parse_bool(series, source):
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    normalized = series.astype("string").str.strip().str.lower()
    parsed = normalized.map(mapping)

    if parsed.isna().any():
        bad = sorted(
            normalized.loc[parsed.isna()].dropna().unique().tolist()
        )
        fail(f"{source}: invalid boolean values: {bad[:10]}")

    return parsed.astype(bool)


def load_price():
    if not PRICE.exists():
        fail(f"Missing price file: {PRICE}")

    df = pd.read_csv(PRICE)

    missing = PRICE_REQUIRED - set(df.columns)
    if missing:
        fail(f"Price dataset missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = (
        df["ticker"].astype("string").str.strip().str.upper()
    )

    if df["date"].isna().any():
        fail("Price dataset contains invalid dates.")

    if df["ticker"].isna().any() or (df["ticker"] == "").any():
        fail("Price dataset contains invalid tickers.")

    if df.duplicated(KEY).any():
        fail("Price dataset contains duplicate ticker+date keys.")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(KEY).reset_index(drop=True)


def load_base_eligibility():
    if not ELIGIBILITY.exists():
        fail(
            "Missing V1.3F-5 local eligibility artifact: "
            f"{ELIGIBILITY}"
        )

    e = pd.read_csv(ELIGIBILITY)

    missing = ELIG_REQUIRED - set(e.columns)
    if missing:
        fail(
            "Eligibility artifact missing columns: "
            f"{sorted(missing)}"
        )

    e["date"] = pd.to_datetime(e["date"], errors="coerce")
    e["ticker"] = (
        e["ticker"].astype("string").str.strip().str.upper()
    )

    if e["date"].isna().any():
        fail("Eligibility artifact contains invalid dates.")

    e = e.loc[e["source_universe"] == "BASE_95"].copy()

    if e.empty:
        fail("Eligibility artifact has no BASE_95 rows.")

    e["placeholder_candidate"] = parse_bool(
        e["placeholder_candidate"],
        "placeholder_candidate",
    )
    e["research_session_eligible"] = parse_bool(
        e["research_session_eligible"],
        "research_session_eligible",
    )

    if e.duplicated(KEY).any():
        fail("BASE_95 eligibility contains duplicate ticker+date keys.")

    contradiction = (
        e["placeholder_candidate"]
        & e["research_session_eligible"]
    )
    if contradiction.any():
        fail(
            "Eligibility contradiction: placeholder candidate marked "
            "research-session eligible."
        )

    return e[
        KEY
        + [
            "placeholder_candidate",
            "research_session_eligible",
        ]
    ].copy()


def attach_eligibility(price, eligibility):
    merged = price.merge(
        eligibility,
        on=KEY,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing = merged["_merge"] != "both"
    if missing.any():
        examples = (
            merged.loc[missing, KEY]
            .head(10)
            .astype(str)
            .to_dict("records")
        )
        fail(
            "Some base-price rows have no V1.3F-5 eligibility record. "
            f"Examples: {examples}"
        )

    merged = merged.drop(columns="_merge")

    return merged


def calculate_features_and_labels(df):
    x = df.copy()
    x = x.sort_values(KEY).reset_index(drop=True)

    g = x.groupby("ticker", group_keys=False)

    for n in [5, 10, 20, 60]:
        x[f"ret{n}"] = g["close"].pct_change(n)

    for n in [20, 50, 200]:
        ma = g["close"].transform(
            lambda s: s.rolling(n, min_periods=n).mean()
        )
        x[f"ma{n}_dist"] = x["close"] / ma - 1

    x["vol20"] = g["volume"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
    x["vol_ratio"] = x["volume"] / x["vol20"]

    x["volatility20"] = g["ret5"].transform(
        lambda s: s.rolling(4, min_periods=4).std()
    )

    delta = g["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.groupby(x["ticker"]).transform(
        lambda s: s.rolling(14, min_periods=14).mean()
    )
    avg_loss = loss.groupby(x["ticker"]).transform(
        lambda s: s.rolling(14, min_periods=14).mean()
    )

    rs = avg_gain / avg_loss.replace(0, np.nan)
    x["rsi"] = 100 - (100 / (1 + rs))

    dollar_value = x["close"] * x["volume"]
    x["liq20"] = dollar_value.groupby(x["ticker"]).transform(
        lambda s: s.rolling(20, min_periods=20).median()
    )

    # Same forward-label semantics as actual_engine.py:
    # next 5 ordered rows for the same ticker.
    g = x.groupby("ticker", group_keys=False)

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
            (hi >= x["close"] * (1 + target))
            & (lo > x["close"] * 0.97)
        ).astype(float)

        label = label.where(complete_horizon, np.nan)
        x[f"y{int(target * 100)}"] = label

    return x


def numeric_change_stats(comp, name):
    base = comp[f"{name}_baseline"]
    filt = comp[f"{name}_filtered"]

    comparable = base.notna() & filt.notna()

    if not comparable.any():
        return {
            "comparable_rows": 0,
            "changed_rows": 0,
            "changed_pct": None,
            "mean_abs_diff": None,
            "max_abs_diff": None,
        }

    diff = (base.loc[comparable] - filt.loc[comparable]).abs()

    # Tight numerical tolerance: identify semantic differences rather
    # than insignificant floating-point noise.
    changed = ~np.isclose(
        base.loc[comparable],
        filt.loc[comparable],
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )

    return {
        "comparable_rows": int(comparable.sum()),
        "changed_rows": int(changed.sum()),
        "changed_pct": float(changed.mean() * 100),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
    }


def label_change_stats(comp, name):
    base = comp[f"{name}_baseline"]
    filt = comp[f"{name}_filtered"]

    comparable = base.notna() & filt.notna()
    changed = comparable & (base != filt)

    base_known_filtered_unknown = base.notna() & filt.isna()
    base_unknown_filtered_known = base.isna() & filt.notna()

    return {
        "comparable_rows": int(comparable.sum()),
        "changed_rows": int(changed.sum()),
        "changed_pct": (
            float(changed.sum() / comparable.sum() * 100)
            if comparable.sum()
            else None
        ),
        "baseline_known_filtered_unknown": int(
            base_known_filtered_unknown.sum()
        ),
        "baseline_unknown_filtered_known": int(
            base_unknown_filtered_known.sum()
        ),
    }


def main():
    price = load_price()
    eligibility = load_base_eligibility()
    joined = attach_eligibility(price, eligibility)

    baseline = calculate_features_and_labels(joined)

    filtered_input = joined.loc[
        joined["research_session_eligible"]
    ].copy()

    filtered = calculate_features_and_labels(filtered_input)

    numeric_features = [
        "ret5",
        "ret10",
        "ret20",
        "ret60",
        "ma20_dist",
        "ma50_dist",
        "ma200_dist",
        "vol_ratio",
        "volatility20",
        "rsi",
        "liq20",
    ]

    labels = ["y3", "y5", "y8"]

    keep = KEY + numeric_features + labels

    b = baseline[keep].copy()
    f = filtered[keep].copy()

    b = b.rename(
        columns={
            c: f"{c}_baseline"
            for c in numeric_features + labels
        }
    )

    f = f.rename(
        columns={
            c: f"{c}_filtered"
            for c in numeric_features + labels
        }
    )

    # Compare only rows retained by the filtered research view.
    comp = b.merge(
        f,
        on=KEY,
        how="inner",
        validate="one_to_one",
    )

    numeric_summary = {
        name: numeric_change_stats(comp, name)
        for name in numeric_features
    }

    label_summary = {
        name: label_change_stats(comp, name)
        for name in labels
    }

    any_numeric_changed = np.zeros(len(comp), dtype=bool)

    for name in numeric_features:
        a = comp[f"{name}_baseline"]
        b2 = comp[f"{name}_filtered"]

        comparable = a.notna() & b2.notna()

        changed = np.zeros(len(comp), dtype=bool)

        if comparable.any():
            changed[comparable.to_numpy()] = ~np.isclose(
                a.loc[comparable],
                b2.loc[comparable],
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            )

        any_numeric_changed |= changed

    any_label_changed = np.zeros(len(comp), dtype=bool)

    for name in labels:
        a = comp[f"{name}_baseline"]
        b2 = comp[f"{name}_filtered"]

        changed = (
            (a.notna() & b2.notna() & (a != b2))
            | (a.notna() & b2.isna())
            | (a.isna() & b2.notna())
        )

        any_label_changed |= changed.to_numpy()

    comp["any_numeric_feature_changed"] = any_numeric_changed
    comp["any_forward_label_changed"] = any_label_changed

    affected = comp.loc[
        comp["any_numeric_feature_changed"]
        | comp["any_forward_label_changed"]
    ].copy()

    summary = {
        "version": "V1.3F-6",
        "status": "SESSION_FILTER_IMPACT_ANALYSIS_COMPLETE",
        "policy": {
            "raw_price_datasets_modified": False,
            "actual_engine_modified": False,
            "model_retrained": False,
            "official_idx_calendar_established": False,
            "placeholder_is_proven_non_trading_session": False,
        },
        "input": {
            "base_rows": int(len(joined)),
            "placeholder_candidate_rows": int(
                joined["placeholder_candidate"].sum()
            ),
            "filtered_rows": int(len(filtered_input)),
            "unique_tickers": int(joined["ticker"].nunique()),
        },
        "comparison": {
            "common_retained_rows": int(len(comp)),
            "rows_with_any_numeric_feature_change": int(
                comp["any_numeric_feature_changed"].sum()
            ),
            "rows_with_any_forward_label_change": int(
                comp["any_forward_label_changed"].sum()
            ),
            "rows_with_any_feature_or_label_change": int(
                (
                    comp["any_numeric_feature_changed"]
                    | comp["any_forward_label_changed"]
                ).sum()
            ),
        },
        "numeric_features": numeric_summary,
        "forward_labels": label_summary,
        "notes": [
            (
                "Baseline reproduces ordered-row session semantics "
                "used by actual_engine.py for the analyzed fields."
            ),
            (
                "Filtered calculations exclude V1.3F-4 "
                "PLACEHOLDER_CANDIDATE rows before rolling and "
                "forward-horizon calculations."
            ),
            (
                "This analysis does not determine whether the "
                "placeholder dates are official IDX non-trading days."
            ),
            (
                "No model probabilities, AUC values, opportunity "
                "scores, or trading decisions are recalculated here."
            ),
        ],
    }

    # Store only affected common rows, not the entire comparison table.
    # This keeps the generated artifact smaller and auditable.
    affected.to_csv(
        OUT_DETAIL,
        index=False,
        date_format="%Y-%m-%d",
    )

    with open(OUT_SUMMARY, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("V1.3F-6 Session-Filter Impact Analysis")
    print("--------------------------------------")
    print(f"Base rows: {len(joined):,}")
    print(
        "Placeholder candidates excluded: "
        f"{joined['placeholder_candidate'].sum():,}"
    )
    print(f"Filtered rows: {len(filtered_input):,}")
    print(f"Common retained rows: {len(comp):,}")
    print()
    print(
        "Rows with any numeric feature change: "
        f"{comp['any_numeric_feature_changed'].sum():,}"
    )
    print(
        "Rows with any forward label change: "
        f"{comp['any_forward_label_changed'].sum():,}"
    )
    print(
        "Rows with any feature or label change: "
        f"{summary['comparison']['rows_with_any_feature_or_label_change']:,}"
    )
    print()
    print("Forward-label changes:")
    for name, stats in label_summary.items():
        print(
            f"  {name}: changed={stats['changed_rows']:,} "
            f"of comparable={stats['comparable_rows']:,}"
        )
    print()
    print("STATUS: SESSION_FILTER_IMPACT_ANALYSIS_COMPLETE")


if __name__ == "__main__":
    main()