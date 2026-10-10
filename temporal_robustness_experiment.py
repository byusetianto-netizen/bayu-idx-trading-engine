"""
V1.3F-8 Temporal Robustness / Walk-Forward Comparison

Purpose
-------
Test whether the impact of V1.3F-5 research-session filtering is
temporally stable.

This experiment compares BASELINE vs FILTERED using identical:
- source data
- features
- labels
- model
- hyperparameters
- training rule

Only session filtering differs.

IMPORTANT:
This is a research diagnostic.
It does NOT modify actual_engine.py or enable production filtering.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import filtered_model_experiment as f7


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "execution_probe"

SUMMARY_FILE = OUT_DIR / "temporal_robustness_summary.json"
DETAIL_FILE = OUT_DIR / "temporal_robustness_metrics.csv"


# Expanding-window walk-forward design.
#
# Each validation year is trained only on observations strictly
# before that year. No future observations are used for training.
WINDOWS = [
    {
        "name": "VALID_2022",
        "train_end": "2021-12-31",
        "valid_start": "2022-01-01",
        "valid_end": "2022-12-31",
    },
    {
        "name": "VALID_2023",
        "train_end": "2022-12-31",
        "valid_start": "2023-01-01",
        "valid_end": "2023-12-31",
    },
    {
        "name": "VALID_2024",
        "train_end": "2023-12-31",
        "valid_start": "2024-01-01",
        "valid_end": "2024-12-31",
    },
    {
        "name": "VALID_2025",
        "train_end": "2024-12-31",
        "valid_start": "2025-01-01",
        "valid_end": "2025-12-31",
    },
]


def fail(message):
    raise RuntimeError(message)


def evaluate_window(frame, experiment_name, window):
    train_end = pd.Timestamp(window["train_end"])
    valid_start = pd.Timestamp(window["valid_start"])
    valid_end = pd.Timestamp(window["valid_end"])

    # Purge the final five observations per ticker before validation.
    # Forward labels require five subsequent observations.
    pre_validation = frame["date"] <= train_end

    session_position = (
        frame.groupby("ticker").cumcount()
    )

    last_train_position = (
        session_position.where(pre_validation)
        .groupby(frame["ticker"])
        .transform("max")
    )

    train = (
        pre_validation
        & (session_position <= last_train_position - 5)
    )

    valid = (
        (frame["date"] >= valid_start)
        & (frame["date"] <= valid_end)
    )

    rows = []

    for target in f7.TARGETS:
        label = f"y{target}"

        train_mask = (
            train
            & frame[label].notna()
            & frame[f7.FEATURES].notna().all(axis=1)
        )

        validation_mask = (
            valid
            & frame[label].notna()
            & frame[f7.FEATURES].notna().all(axis=1)
        )

        y_train = (
            frame.loc[train_mask, label]
            .astype(int)
        )

        y_valid = (
            frame.loc[validation_mask, label]
            .astype(int)
        )

        if y_train.empty:
            fail(
                f"{experiment_name} {window['name']} "
                f"target {target}: empty training sample."
            )

        if y_valid.empty:
            fail(
                f"{experiment_name} {window['name']} "
                f"target {target}: empty validation sample."
            )

        if y_train.nunique() < 2:
            fail(
                f"{experiment_name} {window['name']} "
                f"target {target}: training target "
                "has fewer than two classes."
            )

        model = f7.make_model()

        model.fit(
            frame.loc[train_mask, f7.FEATURES],
            y_train,
        )

        probability = model.predict_proba(
            frame.loc[
                validation_mask,
                f7.FEATURES,
            ]
        )[:, 1]

        auc = (
            roc_auc_score(
                y_valid,
                probability,
            )
            if y_valid.nunique() > 1
            else np.nan
        )

        rows.append(
            {
                "experiment": experiment_name,
                "window": window["name"],
                "train_end": window["train_end"],
                "valid_start": window["valid_start"],
                "valid_end": window["valid_end"],
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

    return pd.DataFrame(rows)


def main():
    print(
        "V1.3F-8 Temporal Robustness / "
        "Walk-Forward Comparison"
    )
    print("--------------------------------------------")

    price = f7.load_prices()
    eligibility = f7.load_eligibility()
    ihsg = f7.load_ihsg()

    joined = f7.attach_eligibility(
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
        f"{int(joined['placeholder_candidate'].sum()):,}"
    )
    print(
        f"Filtered rows: {len(filtered_input):,}"
    )

    baseline = f7.build_features(
        baseline_input,
        ihsg,
    )

    filtered = f7.build_features(
        filtered_input,
        ihsg,
    )

    result_parts = []

    for window in WINDOWS:
        print()
        print(f"Evaluating {window['name']}...")

        result_parts.append(
            evaluate_window(
                baseline,
                "BASELINE",
                window,
            )
        )

        result_parts.append(
            evaluate_window(
                filtered,
                "FILTERED",
                window,
            )
        )

    metrics = pd.concat(
        result_parts,
        ignore_index=True,
    )

    baseline_metrics = metrics.loc[
        metrics["experiment"] == "BASELINE"
    ].copy()

    filtered_metrics = metrics.loc[
        metrics["experiment"] == "FILTERED"
    ].copy()

    comparison = baseline_metrics.merge(
        filtered_metrics,
        on=["window", "target"],
        suffixes=("_baseline", "_filtered"),
        validate="one_to_one",
    )

    comparison["auc_delta"] = (
        comparison["validation_auc_filtered"]
        - comparison["validation_auc_baseline"]
    )

    comparison["train_n_delta"] = (
        comparison["train_n_filtered"]
        - comparison["train_n_baseline"]
    )

    comparison["validation_n_delta"] = (
        comparison["validation_n_filtered"]
        - comparison["validation_n_baseline"]
    )

    comparison["train_positive_rate_delta"] = (
        comparison["train_positive_rate_filtered"]
        - comparison["train_positive_rate_baseline"]
    )

    comparison["validation_positive_rate_delta"] = (
        comparison["validation_positive_rate_filtered"]
        - comparison["validation_positive_rate_baseline"]
    )

    target_summary = []

    for target in f7.TARGETS:
        subset = comparison.loc[
            comparison["target"] == target
        ].copy()

        delta = subset["auc_delta"].dropna()

        target_summary.append(
            {
                "target": target,
                "windows_total": int(len(subset)),
                "windows_with_auc": int(len(delta)),
                "positive_auc_delta_windows": int(
                    (delta > 0).sum()
                ),
                "negative_auc_delta_windows": int(
                    (delta < 0).sum()
                ),
                "zero_auc_delta_windows": int(
                    (delta == 0).sum()
                ),
                "mean_auc_delta": (
                    float(delta.mean())
                    if len(delta)
                    else None
                ),
                "median_auc_delta": (
                    float(delta.median())
                    if len(delta)
                    else None
                ),
                "min_auc_delta": (
                    float(delta.min())
                    if len(delta)
                    else None
                ),
                "max_auc_delta": (
                    float(delta.max())
                    if len(delta)
                    else None
                ),
            }
        )

    all_delta = comparison["auc_delta"].dropna()

    summary = {
        "version": "V1.3F-8",
        "status": (
            "TEMPORAL_ROBUSTNESS_EXPERIMENT_COMPLETE"
            "__NO_PRODUCTION_CHANGE"
        ),
        "design": {
            "method": "expanding_window_walk_forward",
            "windows": WINDOWS,
            "baseline_and_filtered_same_model": True,
            "baseline_and_filtered_same_features": True,
            "baseline_and_filtered_same_labels": True,
            "only_intended_variable": (
                "V1.3F-5 research-session filtering"
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
        },
        "aggregate": {
            "comparisons": int(len(comparison)),
            "positive_auc_delta": int(
                (all_delta > 0).sum()
            ),
            "negative_auc_delta": int(
                (all_delta < 0).sum()
            ),
            "zero_auc_delta": int(
                (all_delta == 0).sum()
            ),
            "mean_auc_delta": (
                float(all_delta.mean())
                if len(all_delta)
                else None
            ),
            "median_auc_delta": (
                float(all_delta.median())
                if len(all_delta)
                else None
            ),
        },
        "by_target": target_summary,
        "policy": {
            "actual_engine_modified": False,
            "production_filter_enabled": False,
            "raw_price_data_modified": False,
            "official_idx_calendar_established": False,
            "placeholder_proven_non_trading_day": False,
        },
        "notes": [
            (
                "Each validation year is trained only on "
                "observations before that year."
            ),
            (
                "AUC improvement is not assumed; positive and "
                "negative results are retained."
            ),
            (
                "Adjusted-price integrity remains a separate "
                "unresolved data-quality dimension."
            ),
            (
                "This experiment alone does not authorize "
                "production integration."
            ),
        ],
    }

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        DETAIL_FILE,
        index=False,
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
    print("Temporal AUC comparison")
    print("-----------------------")

    display = comparison[
        [
            "window",
            "target",
            "validation_auc_baseline",
            "validation_auc_filtered",
            "auc_delta",
            "train_n_baseline",
            "train_n_filtered",
            "validation_n_baseline",
            "validation_n_filtered",
        ]
    ]

    print(display.to_string(index=False))

    print()
    print("AUC delta summary by target")
    print("---------------------------")

    print(
        pd.DataFrame(
            target_summary
        ).to_string(index=False)
    )

    print()
    print(
        "STATUS: "
        "TEMPORAL_ROBUSTNESS_EXPERIMENT_COMPLETE"
        "__NO_PRODUCTION_CHANGE"
    )


if __name__ == "__main__":
    main()