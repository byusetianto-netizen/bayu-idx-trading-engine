
"""
Bayu IDX Trading Engine
V1.3F-11 Stage E1 - Weekly Calendar and Label Maturity Audit

RESEARCH ONLY.

- Weekly evaluation dates: first available common stock date
  in each calendar week during 2023-2024.
- Training labels must mature strictly before evaluation date.
- Maturity is calculated independently for each dataset.
- No machine learning, ranking, trades or price modifications.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v13f11_calendar_sensitivity import (
    FEATURES,
    LABELS,
    calculate_components,
    read_sources,
)


OUT = Path("data/execution_probe")

SUMMARY_PATH = OUT / "v13f11_weekly_maturity_summary.json"
DETAIL_PATH = OUT / "v13f11_weekly_maturity_detail.csv"

START = pd.Timestamp("2023-01-01")
END = pd.Timestamp("2025-01-01")

KEYS = ["ticker", "date"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def build_dataset(stock, ihsg, name):
    stock = stock.sort_values(KEYS).reset_index(drop=True)

    require(
        not stock.duplicated(KEYS).any(),
        f"{name}: duplicate ticker/date",
    )

    features = calculate_components(stock, ihsg)

    require(
        features[KEYS].equals(stock[KEYS]),
        f"{name}: feature alignment failed",
    )

    maturity = stock.groupby(
        "ticker", sort=False
    )["date"].shift(-5)

    features["label_maturity_date"] = maturity

    # One complete five-session horizon is required.
    has_labels = features[LABELS].notna().all(axis=1)

    require(
        not (
            has_labels
            & features["label_maturity_date"].isna()
        ).any(),
        f"{name}: labels exist without mature horizon",
    )

    require(
        not (
            features["label_maturity_date"].notna()
            & (
                features["label_maturity_date"]
                <= features["date"]
            )
        ).any(),
        f"{name}: invalid maturity chronology",
    )

    return features


def eligible_training(frame, decision_date):
    feature_values = frame[FEATURES].to_numpy(
        dtype=float
    )

    finite_features = np.isfinite(
        feature_values
    ).all(axis=1)

    complete_labels = frame[LABELS].notna().all(
        axis=1
    ).to_numpy()

    matured = (
        frame["label_maturity_date"] < decision_date
    ).to_numpy()

    historical = (
        frame["date"] < decision_date
    ).to_numpy()

    mask = (
        finite_features
        & complete_labels
        & matured
        & historical
    )

    require(
        not (
            mask
            & (
                frame["label_maturity_date"].to_numpy()
                >= np.datetime64(decision_date)
            )
        ).any(),
        "Future label entered training",
    )

    return mask


def weekly_decision_dates(base, scenario):
    base_dates = pd.DatetimeIndex(
        base.loc[
            (base["date"] >= START)
            & (base["date"] < END),
            "date",
        ].unique()
    )

    scenario_dates = pd.DatetimeIndex(
        scenario.loc[
            (scenario["date"] >= START)
            & (scenario["date"] < END),
            "date",
        ].unique()
    )

    common_dates = base_dates.intersection(
        scenario_dates
    ).sort_values()

    require(
        len(common_dates) > 0,
        "No common validation dates",
    )

    calendar = pd.DataFrame(
        {"decision_date": common_dates}
    )

    calendar["week"] = calendar[
        "decision_date"
    ].dt.to_period("W-SUN")

    return (
        calendar.groupby("week", sort=True)
        ["decision_date"]
        .min()
        .reset_index(drop=True)
    )


def main():
    print("V1.3F-11 Stage E1")
    print("Weekly Calendar and Label Maturity Audit")
    print("---------------------------------------")

    stock, ihsg, exclusions = read_sources()

    base = build_dataset(
        stock,
        ihsg,
        "baseline",
    )

    scenario_stock = stock.loc[
        ~stock["exclude"]
    ].copy()

    scenario = build_dataset(
        scenario_stock,
        ihsg,
        "scenario",
    )

    require(
        len(base) == len(stock),
        "Baseline row count mismatch",
    )

    require(
        len(scenario) == len(stock) - exclusions,
        "Scenario row count mismatch",
    )

    dates = weekly_decision_dates(
        base, scenario
    )

    records = []

    for decision_date in dates:
        base_train = eligible_training(
            base, decision_date
        )

        scenario_train = eligible_training(
            scenario, decision_date
        )

        # The decision date is included only in
        # the evaluation universe, never training.
        base_candidates = base.loc[
            base["date"].eq(decision_date)
        ]

        scenario_candidates = scenario.loc[
            scenario["date"].eq(decision_date)
        ]

        base_keys = set(
            zip(
                base_candidates["ticker"],
                base_candidates["date"],
            )
        )

        scenario_keys = set(
            zip(
                scenario_candidates["ticker"],
                scenario_candidates["date"],
            )
        )

        records.append(
            {
                "decision_date": decision_date,
                "baseline_train_n": int(
                    base_train.sum()
                ),
                "scenario_train_n": int(
                    scenario_train.sum()
                ),
                "baseline_candidates_n": len(
                    base_candidates
                ),
                "scenario_candidates_n": len(
                    scenario_candidates
                ),
                "common_candidates_n": len(
                    base_keys & scenario_keys
                ),
                "baseline_only_candidates_n": len(
                    base_keys - scenario_keys
                ),
                "scenario_only_candidates_n": len(
                    scenario_keys - base_keys
                ),
            }
        )

    detail = pd.DataFrame(records)

    require(
        not detail.empty,
        "No weekly decision dates",
    )

    require(
        detail["common_candidates_n"].gt(0).all(),
        "A weekly date has no common candidates",
    )

    require(
        detail["baseline_train_n"].gt(0).all(),
        "A weekly date has no eligible baseline training",
    )

    require(
        detail["scenario_train_n"].gt(0).all(),
        "A weekly date has no eligible scenario training",
    )

    summary = {
        "version": "V1.3F-11",
        "stage": "E1_WEEKLY_CALENDAR_LABEL_MATURITY",
        "status": "RESEARCH_AUDIT_PASS",
        "baseline_rows": len(base),
        "scenario_rows": len(scenario),
        "hypothetical_exclusions": int(exclusions),
        "weekly_decision_count": len(detail),
        "first_decision_date": str(
            detail["decision_date"].min().date()
        ),
        "last_decision_date": str(
            detail["decision_date"].max().date()
        ),
        "maturity_rule": (
            "Fifth subsequent ticker observation "
            "must occur strictly before decision date"
        ),
        "decision_schedule": (
            "First available common stock date "
            "of each calendar week"
        ),
        "training_policy": (
            "Historical rows with complete finite "
            "features and all three mature labels"
        ),
        "baseline_first_train_n": int(
            detail["baseline_train_n"].iloc[0]
        ),
        "scenario_first_train_n": int(
            detail["scenario_train_n"].iloc[0]
        ),
        "baseline_last_train_n": int(
            detail["baseline_train_n"].iloc[-1]
        ),
        "scenario_last_train_n": int(
            detail["scenario_train_n"].iloc[-1]
        ),
        "limitations": [
            "Calendar exclusions are hypothetical.",
            "IHSG dates are not an official IDX calendar.",
            "No model fitting performed.",
            "No Top-3 ranking performed.",
            "No execution backtest performed.",
            "Eligibility requires all three labels.",
            "Point-in-time feature publication and "
            "corporate-action availability are not "
            "independently verified.",
        ],
        "policy": {
            "raw_prices_modified": False,
            "actual_engine_modified": False,
            "production_model_replaced": False,
            "trading_readiness_established": False,
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)

    detail.to_csv(
        DETAIL_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("Weekly dates:", len(detail))
    print(
        "First decision:",
        summary["first_decision_date"],
    )
    print(
        "Last decision:",
        summary["last_decision_date"],
    )
    print(
        "First training baseline/scenario:",
        summary["baseline_first_train_n"],
        "/",
        summary["scenario_first_train_n"],
    )
    print("STATUS: RESEARCH_AUDIT_PASS")


if __name__ == "__main__":
    main()
