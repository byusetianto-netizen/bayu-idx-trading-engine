"""
V1.3F-7 Filtered Model Experiment

Purpose
-------
Measure the model-level impact of excluding V1.3F-4
PLACEHOLDER_CANDIDATE observations before feature and forward-label
calculation.

Experimental rule:
    BASELINE and FILTERED must use the same:
    - source price dataset
    - IHSG context
    - feature definitions
    - forward-label definitions
    - train / validation split
    - model
    - hyperparameters
    - decision thresholds

The only intended experimental variable is session filtering.

This script:
- does NOT modify actual_engine.py
- does NOT modify raw datasets
- does NOT enable filtering in production
- does NOT claim placeholder candidates are official IDX non-trading days
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parent

PRICE_FILE = ROOT / "data" / "idx_stock_prices.csv"
IHSG_FILE = ROOT / "data" / "idx_ihsg_index.csv"

ELIGIBILITY_FILE = (
    ROOT
    / "data"
    / "execution_probe"
    / "research_session_eligibility.csv"
)

OUT_DIR = ROOT / "data" / "execution_probe"

SUMMARY_FILE = OUT_DIR / "filtered_model_experiment_summary.json"
METRICS_FILE = OUT_DIR / "filtered_model_experiment_metrics.csv"
LATEST_FILE = OUT_DIR / "filtered_model_experiment_latest.csv"


FEATURES = [
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
    "rs20",
    "liq20",
]

TARGETS = [3, 5, 8]

TRAIN_END = pd.Timestamp("2022-12-31")
VALID_START = pd.Timestamp("2023-01-01")
VALID_END = pd.Timestamp("2024-12-31")


def fail(message):
    raise RuntimeError(message)


def parse_bool(series, source):
    normalized = series.astype("string").str.strip().str.lower()

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    parsed = normalized.map(mapping)

    if parsed.isna().any():
        bad = sorted(
            normalized.loc[parsed.isna()]
            .dropna()
            .unique()
            .tolist()
        )
        fail(f"{source}: invalid boolean values: {bad[:10]}")

    return parsed.astype(bool)


def load_prices():
    if not PRICE_FILE.exists():
        fail(f"Missing price dataset: {PRICE_FILE}")

    df = pd.read_csv(PRICE_FILE)

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
        fail(
            "Price dataset missing required columns: "
            f"{sorted(missing)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = (
        df["ticker"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if df["date"].isna().any():
        fail("Price dataset contains invalid dates.")

    if df["ticker"].isna().any() or (df["ticker"] == "").any():
        fail("Price dataset contains invalid tickers.")

    if df.duplicated(["ticker", "date"]).any():
        fail("Price dataset contains duplicate ticker+date keys.")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df.sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )


def load_eligibility():
    if not ELIGIBILITY_FILE.exists():
        fail(
            "Missing V1.3F-5 eligibility artifact: "
            f"{ELIGIBILITY_FILE}\n"
            "Run research_session_eligibility.py first."
        )

    e = pd.read_csv(
        ELIGIBILITY_FILE,
        usecols=[
            "date",
            "ticker",
            "source_universe",
            "placeholder_candidate",
            "research_session_eligible",
        ],
    )

    e["date"] = pd.to_datetime(e["date"], errors="coerce")
    e["ticker"] = (
        e["ticker"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if e["date"].isna().any():
        fail("Eligibility artifact contains invalid dates.")

    e = e.loc[
        e["source_universe"] == "BASE_95"
    ].copy()

    if e.empty:
        fail("Eligibility artifact contains no BASE_95 rows.")

    e["placeholder_candidate"] = parse_bool(
        e["placeholder_candidate"],
        "placeholder_candidate",
    )

    e["research_session_eligible"] = parse_bool(
        e["research_session_eligible"],
        "research_session_eligible",
    )

    if e.duplicated(["ticker", "date"]).any():
        fail(
            "BASE_95 eligibility contains duplicate ticker+date keys."
        )

    contradiction = (
        e["placeholder_candidate"]
        & e["research_session_eligible"]
    )

    if contradiction.any():
        fail(
            "Eligibility contradiction: placeholder candidate "
            "marked research-session eligible."
        )

    return e[
        [
            "ticker",
            "date",
            "placeholder_candidate",
            "research_session_eligible",
        ]
    ].copy()


def load_ihsg():
    if not IHSG_FILE.exists():
        fail(f"Missing IHSG dataset: {IHSG_FILE}")

    ih = pd.read_csv(IHSG_FILE)

    if "date" not in ih.columns:
        if "Price" in ih.columns:
            ih = ih.rename(columns={"Price": "date"})
        else:
            fail("IHSG dataset has no date/Price column.")

    if "close" not in ih.columns:
        possible = [
            c
            for c in ih.columns
            if str(c).strip().lower() == "close"
        ]

        if not possible:
            fail("IHSG dataset has no close column.")

        ih = ih.rename(columns={possible[0]: "close"})

    ih["date"] = pd.to_datetime(
        ih["date"],
        errors="coerce",
    )

    ih["close"] = pd.to_numeric(
        ih["close"],
        errors="coerce",
    )

    ih = (
        ih.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    ih["ihsg_ret20"] = ih["close"].pct_change(20)

    return ih[["date", "ihsg_ret20"]]


def attach_eligibility(price, eligibility):
    merged = price.merge(
        eligibility,
        on=["ticker", "date"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing = merged["_merge"] != "both"

    if missing.any():
        examples = (
            merged.loc[
                missing,
                ["ticker", "date"],
            ]
            .head(10)
            .astype(str)
            .to_dict("records")
        )

        fail(
            "Base-price rows missing V1.3F-5 eligibility. "
            f"Examples: {examples}"
        )

    return merged.drop(columns="_merge")


def build_features(df, ihsg):
    x = (
        df.copy()
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    g = x.groupby("ticker", group_keys=False)

    for n in [5, 10, 20, 60]:
        x[f"ret{n}"] = g["close"].pct_change(n)

    for n in [20, 50, 200]:
        ma = g["close"].transform(
            lambda s: s.rolling(
                n,
                min_periods=n,
            ).mean()
        )

        x[f"ma{n}_dist"] = x["close"] / ma - 1

    x["vol20"] = g["volume"].transform(
        lambda s: s.rolling(
            20,
            min_periods=20,
        ).mean()
    )

    x["vol_ratio"] = x["volume"] / x["vol20"]

    x["volatility20"] = g["ret5"].transform(
        lambda s: s.rolling(
            4,
            min_periods=4,
        ).std()
    )

    delta = g["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.groupby(x["ticker"]).transform(
        lambda s: s.rolling(
            14,
            min_periods=14,
        ).mean()
    )

    avg_loss = loss.groupby(x["ticker"]).transform(
        lambda s: s.rolling(
            14,
            min_periods=14,
        ).mean()
    )

    rs = avg_gain / avg_loss.replace(0, np.nan)

    x["rsi"] = 100 - (100 / (1 + rs))

    dollar_value = x["close"] * x["volume"]

    x["liq20"] = dollar_value.groupby(
        x["ticker"]
    ).transform(
        lambda s: s.rolling(
            20,
            min_periods=20,
        ).median()
    )

    x = x.merge(
        ihsg,
        on="date",
        how="left",
        validate="many_to_one",
    )

    x["rs20"] = x["ret20"] - x["ihsg_ret20"]

    # Forward labels: exactly five ordered observations.
    g = x.groupby("ticker", group_keys=False)

    future_highs = pd.concat(
        [
            g["high"].shift(-i)
            for i in range(1, 6)
        ],
        axis=1,
    )

    future_lows = pd.concat(
        [
            g["low"].shift(-i)
            for i in range(1, 6)
        ],
        axis=1,
    )

    complete_horizon = (
        future_highs.notna().all(axis=1)
        & future_lows.notna().all(axis=1)
    )

    future_high = future_highs.max(
        axis=1,
        skipna=False,
    )

    future_low = future_lows.min(
        axis=1,
        skipna=False,
    )

    for target in TARGETS:
        threshold = target / 100

        y = (
            (
                future_high
                >= x["close"] * (1 + threshold)
            )
            & (
                future_low
                > x["close"] * 0.97
            )
        ).astype(float)

        x[f"y{target}"] = y.where(
            complete_horizon,
            np.nan,
        )

    return x


def make_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
    )


def experiment(name, frame):
    train = frame["date"] <= TRAIN_END

    valid = (
        (frame["date"] >= VALID_START)
        & (frame["date"] <= VALID_END)
    )

    latest_date = frame["date"].max()

    latest = frame["date"] == latest_date

    metrics = []
    latest_result = frame.loc[
        latest,
        ["ticker", "date", "close"],
    ].copy()

    for target in TARGETS:
        label = f"y{target}"

        train_mask = (
            train
            & frame[label].notna()
            & frame[FEATURES].notna().all(axis=1)
        )

        validation_mask = (
            valid
            & frame[label].notna()
            & frame[FEATURES].notna().all(axis=1)
        )

        y_train = (
            frame.loc[
                train_mask,
                label,
            ]
            .astype(int)
        )

        y_valid = (
            frame.loc[
                validation_mask,
                label,
            ]
            .astype(int)
        )

        if y_train.empty:
            fail(
                f"{name} target {target}: "
                "empty training sample."
            )

        if y_valid.empty:
            fail(
                f"{name} target {target}: "
                "empty validation sample."
            )

        if y_train.nunique() < 2:
            fail(
                f"{name} target {target}: "
                "training target has fewer than 2 classes."
            )

        model = make_model()

        model.fit(
            frame.loc[
                train_mask,
                FEATURES,
            ],
            y_train,
        )

        validation_probability = (
            model.predict_proba(
                frame.loc[
                    validation_mask,
                    FEATURES,
                ]
            )[:, 1]
        )

        auc = (
            roc_auc_score(
                y_valid,
                validation_probability,
            )
            if y_valid.nunique() > 1
            else np.nan
        )

        metrics.append(
            {
                "experiment": name,
                "target": target,
                "train_n": int(train_mask.sum()),
                "train_positive_n": int(y_train.sum()),
                "train_positive_rate": float(
                    y_train.mean()
                ),
                "validation_n": int(
                    validation_mask.sum()
                ),
                "validation_positive_n": int(
                    y_valid.sum()
                ),
                "validation_positive_rate": float(
                    y_valid.mean()
                ),
                "validation_auc": (
                    float(auc)
                    if pd.notna(auc)
                    else None
                ),
            }
        )

        latest_features = frame.loc[
            latest,
            FEATURES,
        ]

        latest_valid = (
            latest_features.notna().all(axis=1)
        )

        probability = np.full(
            len(latest_features),
            np.nan,
        )

        if latest_valid.any():
            probability[latest_valid.to_numpy()] = (
                model.predict_proba(
                    latest_features.loc[
                        latest_valid
                    ]
                )[:, 1]
            )

        latest_result[f"p{target}"] = probability

    latest_result["experiment"] = name

    return (
        pd.DataFrame(metrics),
        latest_result,
    )


def compare_latest(base_latest, filtered_latest):
    base = base_latest.rename(
        columns={
            "close": "close_baseline",
            "p3": "p3_baseline",
            "p5": "p5_baseline",
            "p8": "p8_baseline",
        }
    ).drop(columns=["experiment"])

    filt = filtered_latest.rename(
        columns={
            "close": "close_filtered",
            "p3": "p3_filtered",
            "p5": "p5_filtered",
            "p8": "p8_filtered",
        }
    ).drop(columns=["experiment"])

    comp = base.merge(
        filt,
        on=["ticker", "date"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    for target in TARGETS:
        b = comp[f"p{target}_baseline"]
        f = comp[f"p{target}_filtered"]

        comp[f"p{target}_delta"] = f - b

    return comp


def main():
    print("V1.3F-7 Filtered Model Experiment")
    print("--------------------------------")

    price = load_prices()
    eligibility = load_eligibility()
    ihsg = load_ihsg()

    joined = attach_eligibility(
        price,
        eligibility,
    )

    baseline_input = joined.copy()

    filtered_input = joined.loc[
        joined["research_session_eligible"]
    ].copy()

    print(f"Base rows: {len(baseline_input):,}")
    print(
        "Placeholder candidates excluded: "
        f"{joined['placeholder_candidate'].sum():,}"
    )
    print(
        f"Filtered rows: {len(filtered_input):,}"
    )

    baseline = build_features(
        baseline_input,
        ihsg,
    )

    filtered = build_features(
        filtered_input,
        ihsg,
    )

    baseline_metrics, baseline_latest = experiment(
        "BASELINE",
        baseline,
    )

    filtered_metrics, filtered_latest = experiment(
        "FILTERED",
        filtered,
    )

    metrics = pd.concat(
        [
            baseline_metrics,
            filtered_metrics,
        ],
        ignore_index=True,
    )

    latest_comparison = compare_latest(
        baseline_latest,
        filtered_latest,
    )

    metric_comparison = (
        baseline_metrics.merge(
            filtered_metrics,
            on="target",
            suffixes=(
                "_baseline",
                "_filtered",
            ),
            validate="one_to_one",
        )
    )

    metric_comparison["auc_delta"] = (
        metric_comparison["validation_auc_filtered"]
        - metric_comparison["validation_auc_baseline"]
    )

    metric_comparison["train_n_delta"] = (
        metric_comparison["train_n_filtered"]
        - metric_comparison["train_n_baseline"]
    )

    metric_comparison[
        "train_positive_rate_delta"
    ] = (
        metric_comparison[
            "train_positive_rate_filtered"
        ]
        - metric_comparison[
            "train_positive_rate_baseline"
        ]
    )

    metric_comparison[
        "validation_positive_rate_delta"
    ] = (
        metric_comparison[
            "validation_positive_rate_filtered"
        ]
        - metric_comparison[
            "validation_positive_rate_baseline"
        ]
    )

    probability_summary = {}

    for target in TARGETS:
        b = latest_comparison[
            f"p{target}_baseline"
        ]

        f = latest_comparison[
            f"p{target}_filtered"
        ]

        comparable = b.notna() & f.notna()

        delta = (
            f.loc[comparable]
            - b.loc[comparable]
        )

        probability_summary[f"p{target}"] = {
            "comparable_latest_tickers": int(
                comparable.sum()
            ),
            "mean_probability_delta": (
                float(delta.mean())
                if len(delta)
                else None
            ),
            "median_probability_delta": (
                float(delta.median())
                if len(delta)
                else None
            ),
            "mean_absolute_probability_delta": (
                float(delta.abs().mean())
                if len(delta)
                else None
            ),
            "max_absolute_probability_delta": (
                float(delta.abs().max())
                if len(delta)
                else None
            ),
        }

    summary = {
        "version": "V1.3F-7",
        "status": (
            "FILTERED_MODEL_EXPERIMENT_COMPLETE"
            "__NO_PRODUCTION_CHANGE"
        ),
        "policy": {
            "raw_price_datasets_modified": False,
            "actual_engine_modified": False,
            "production_filter_enabled": False,
            "official_idx_calendar_established": False,
            "placeholder_is_proven_non_trading_session": False,
            "experiment_variable": (
                "exclude V1.3F-4 PLACEHOLDER_CANDIDATE "
                "before ordered-session calculations"
            ),
        },
        "input": {
            "base_rows": int(len(baseline_input)),
            "filtered_rows": int(len(filtered_input)),
            "excluded_rows": int(
                joined["placeholder_candidate"].sum()
            ),
            "unique_tickers": int(
                joined["ticker"].nunique()
            ),
            "baseline_latest_date": str(
                baseline["date"].max().date()
            ),
            "filtered_latest_date": str(
                filtered["date"].max().date()
            ),
        },
        "split": {
            "train_end": str(TRAIN_END.date()),
            "validation_start": str(
                VALID_START.date()
            ),
            "validation_end": str(
                VALID_END.date()
            ),
        },
        "metric_comparison": (
            metric_comparison[
                [
                    "target",
                    "validation_auc_baseline",
                    "validation_auc_filtered",
                    "auc_delta",
                    "train_n_baseline",
                    "train_n_filtered",
                    "train_n_delta",
                    "train_positive_rate_baseline",
                    "train_positive_rate_filtered",
                    "train_positive_rate_delta",
                    "validation_n_baseline",
                    "validation_n_filtered",
                    "validation_positive_rate_baseline",
                    "validation_positive_rate_filtered",
                    "validation_positive_rate_delta",
                ]
            ]
            .to_dict("records")
        ),
        "latest_probability_comparison": (
            probability_summary
        ),
        "notes": [
            (
                "BASELINE and FILTERED use identical model "
                "configuration and date split."
            ),
            (
                "The intended experimental variable is only "
                "V1.3F-5 research-session filtering."
            ),
            (
                "Results are research diagnostics and do not "
                "authorize production integration."
            ),
            (
                "Adjusted-price integrity issues remain a "
                "separate unresolved data-quality dimension."
            ),
        ],
    }

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_csv(
        METRICS_FILE,
        index=False,
    )

    latest_comparison.to_csv(
        LATEST_FILE,
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

    print()
    print("Validation comparison")
    print("---------------------")

    display_cols = [
        "target",
        "validation_auc_baseline",
        "validation_auc_filtered",
        "auc_delta",
        "train_n_baseline",
        "train_n_filtered",
        "train_n_delta",
    ]

    print(
        metric_comparison[
            display_cols
        ].to_string(index=False)
    )

    print()
    print("Latest probability impact")
    print("-------------------------")

    for target in TARGETS:
        stats = probability_summary[
            f"p{target}"
        ]

        print(
            f"p{target}: "
            f"comparable="
            f"{stats['comparable_latest_tickers']:,}, "
            f"mean_abs_delta="
            f"{stats['mean_absolute_probability_delta']}, "
            f"max_abs_delta="
            f"{stats['max_absolute_probability_delta']}"
        )

    print()
    print(
        "STATUS: "
        "FILTERED_MODEL_EXPERIMENT_COMPLETE"
        "__NO_PRODUCTION_CHANGE"
    )


if __name__ == "__main__":
    main()