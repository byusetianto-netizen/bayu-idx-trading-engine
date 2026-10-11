
"""
Bayu IDX Trading Engine
V1.3F-11 Stage D — Controlled Calendar Model Impact

RESEARCH DIAGNOSTIC ONLY

D1: Fixed baseline model, comparing baseline and calendar-scenario
    validation inputs on identical ticker/date observations.

D2: Controlled retraining on the calendar-scenario dataset,
    evaluated on the same common validation observations.

Safety:
- The calendar exclusions are hypothetical, not verified errors.
- No changes to raw prices or actual_engine.py.
- No production signals or trades are generated.
- No walk-forward validation or trading backtest is performed.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from v13f11_calendar_sensitivity import (
    read_sources,
    calculate_components,
    FEATURES,
    LABELS,
)


VERSION = "V1.3F-11"
STAGE = "D_CONTROLLED_CALENDAR_MODEL_IMPACT"

OUT = Path("data/execution_probe")
SUMMARY_FILE = OUT / "v13f11_calendar_model_impact_summary.json"
DETAIL_FILE = OUT / "v13f11_calendar_model_impact_detail.csv"

C1_FILE = OUT / "v13f11_baseline_parity_summary.json"

TARGETS = [3, 5, 8]
PROBABILITY_CHANGE_TOLERANCE = 1e-10

TRAIN_END = pd.Timestamp("2023-01-01")
VALIDATION_END = pd.Timestamp("2025-01-01")

KEYS = ["ticker", "date"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def make_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=180,
            max_leaf_nodes=15,
            learning_rate=0.06,
            l2_regularization=1.0,
            random_state=42,
        ),
    )


def check_c1_gate():
    require(
        C1_FILE.is_file(),
        "Missing Stage C1 parity report. Run Stage C1 first.",
    )

    report = json.loads(
        C1_FILE.read_text(encoding="utf-8")
    )

    require(
        report.get("stage")
        == "C1_FEATURE_LABEL_BASELINE_PARITY",
        "Unexpected Stage C1 report.",
    )

    require(
        report.get("status") == "PASS",
        "Stage C1 parity has not passed.",
    )

    require(
        not report.get("failed_components"),
        "Stage C1 contains failed components.",
    )


def prepare(frame, name):
    require(
        set(KEYS + FEATURES + LABELS).issubset(frame.columns),
        f"{name}: missing required columns.",
    )

    result = frame.copy()

    result["date"] = pd.to_datetime(
        result["date"], errors="coerce"
    ).dt.normalize()

    result["ticker"] = (
        result["ticker"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    require(
        result[KEYS].notna().all().all(),
        f"{name}: invalid ticker/date.",
    )

    require(
        not result.duplicated(KEYS).any(),
        f"{name}: duplicate ticker/date.",
    )

    return result.set_index(KEYS).sort_index()


def eligible(frame, target):
    x = frame[FEATURES].to_numpy(dtype=float)
    y = frame[f"y{target}"]

    finite_features = pd.Series(
        np.isfinite(x).all(axis=1),
        index=frame.index,
    )

    valid_label = (
        y.notna()
        & np.isfinite(y.to_numpy(dtype=float))
    )

    return finite_features & valid_label


def fit_model(frame, target):
    dates = frame.index.get_level_values("date")
    mask = eligible(frame, target) & (dates < TRAIN_END)

    x = frame.loc[mask, FEATURES]
    y = frame.loc[mask, f"y{target}"].astype(int)

    require(
        len(x) > 0,
        f"No eligible training observations for y{target}.",
    )

    require(
        y.nunique() == 2,
        f"Training labels for y{target} contain only one class.",
    )

    model = make_model()
    model.fit(x, y)

    return model, mask


def calculate_auc(labels, probability):
    require(
        labels.nunique() == 2,
        "Common validation labels must contain both classes.",
    )

    return float(
        roc_auc_score(labels.astype(int), probability)
    )


def spearman_or_none(first, second):
    result = pd.Series(first).corr(
        pd.Series(second),
        method="spearman",
    )

    if pd.isna(result):
        return None

    return float(result)


def compare_target(base, scenario, target):
    label = f"y{target}"

    base_dates = base.index.get_level_values("date")
    scenario_dates = scenario.index.get_level_values("date")

    base_train_mask = eligible(base, target) & (
        base_dates < TRAIN_END
    )
    scenario_train_mask = eligible(scenario, target) & (
        scenario_dates < TRAIN_END
    )

    base_valid_mask = (
        eligible(base, target)
        & (base_dates >= TRAIN_END)
        & (base_dates < VALIDATION_END)
    )

    scenario_valid_mask = (
        eligible(scenario, target)
        & (scenario_dates >= TRAIN_END)
        & (scenario_dates < VALIDATION_END)
    )

    base_valid = base.loc[base_valid_mask]
    scenario_valid = scenario.loc[scenario_valid_mask]

    common_index = base_valid.index.intersection(
        scenario_valid.index
    ).sort_values()

    require(
        len(common_index) > 0,
        f"No common validation observations for y{target}.",
    )

    base_common = base.loc[common_index]
    scenario_common = scenario.loc[common_index]

    base_labels = base_common[label].astype(int)
    scenario_labels = scenario_common[label].astype(int)

    validation_label_changes = int(
        (
            base_labels.to_numpy()
            != scenario_labels.to_numpy()
        ).sum()
    )

    require(
        validation_label_changes == 0,
        (
            f"Validation labels differ for y{target}. "
            "AUC comparison is blocked."
        ),
    )

    # Measure label changes on matching eligible training observations.
    common_train_index = (
        base.loc[base_train_mask].index.intersection(
            scenario.loc[scenario_train_mask].index
        )
    )

    train_label_changes = int(
        (
            base.loc[common_train_index, label].to_numpy()
            != scenario.loc[common_train_index, label].to_numpy()
        ).sum()
    )

    # D1: Keep the baseline model fixed.
    baseline_model, _ = fit_model(base, target)

    p_base = baseline_model.predict_proba(
        base_common[FEATURES]
    )[:, 1]

    p_d1 = baseline_model.predict_proba(
        scenario_common[FEATURES]
    )[:, 1]

    # D2: Retrain the identical algorithm on the scenario.
    scenario_model, _ = fit_model(scenario, target)

    p_d2 = scenario_model.predict_proba(
        scenario_common[FEATURES]
    )[:, 1]

    d1_delta = np.abs(p_d1 - p_base)
    d2_delta = np.abs(p_d2 - p_base)

    metrics = {
        "target": int(target),
        "baseline_train_n": int(base_train_mask.sum()),
        "scenario_train_n": int(scenario_train_mask.sum()),
        "common_eligible_train_n": int(len(common_train_index)),
        "common_train_label_changes": train_label_changes,
        "baseline_validation_n": int(len(base_valid)),
        "scenario_validation_n": int(len(scenario_valid)),
        "common_validation_n": int(len(common_index)),
        "baseline_only_validation_n": int(
            len(base_valid) - len(common_index)
        ),
        "scenario_only_validation_n": int(
            len(scenario_valid) - len(common_index)
        ),
        "common_validation_label_changes": (
            validation_label_changes
        ),
        "baseline_auc_common": calculate_auc(
            base_labels, p_base
        ),
        "d1_fixed_model_auc_common": calculate_auc(
            base_labels, p_d1
        ),
        "d2_retrained_auc_common": calculate_auc(
            base_labels, p_d2
        ),
        "d1_changed_probabilities": int(
            (d1_delta > PROBABILITY_CHANGE_TOLERANCE).sum()
        ),
        "d2_changed_probabilities": int(
            (d2_delta > PROBABILITY_CHANGE_TOLERANCE).sum()
        ),
        "d1_mean_abs_probability_delta": float(
            d1_delta.mean()
        ),
        "d2_mean_abs_probability_delta": float(
            d2_delta.mean()
        ),
        "d1_max_abs_probability_delta": float(
            d1_delta.max()
        ),
        "d2_max_abs_probability_delta": float(
            d2_delta.max()
        ),
        "d1_spearman": spearman_or_none(p_base, p_d1),
        "d2_spearman": spearman_or_none(p_base, p_d2),
    }

    detail = common_index.to_frame(index=False)

    detail["target"] = int(target)
    detail["label"] = base_labels.to_numpy()
    detail["baseline_probability"] = p_base
    detail["fixed_calendar_probability"] = p_d1
    detail["retrained_calendar_probability"] = p_d2
    detail["d1_abs_probability_delta"] = d1_delta
    detail["d2_abs_probability_delta"] = d2_delta

    return metrics, detail


def main():
    print("=== V1.3F-11 STAGE D ===")

    check_c1_gate()

    stock, ihsg, exclusion_count = read_sources()

    base_raw = calculate_components(stock, ihsg)

    scenario_stock = stock.loc[
        ~stock["exclude"]
    ].copy()

    scenario_raw = calculate_components(
        scenario_stock,
        ihsg,
    )

    require(
        len(base_raw) == len(stock),
        "Baseline row count mismatch.",
    )

    require(
        len(scenario_raw) == len(stock) - exclusion_count,
        "Scenario row count mismatch.",
    )

    base = prepare(base_raw, "Baseline")
    scenario = prepare(scenario_raw, "Calendar scenario")

    require(
        scenario.index.isin(base.index).all(),
        "Scenario contains keys absent from baseline.",
    )

    results = []
    details = []

    for target in TARGETS:
        print(f"\nTesting target {target}% ...")

        metrics, detail = compare_target(
            base, scenario, target
        )

        results.append(metrics)
        details.append(detail)

        print(
            "Training:",
            f"{metrics['baseline_train_n']:,}",
            "->",
            f"{metrics['scenario_train_n']:,}",
        )

        print(
            "Changed training labels:",
            metrics["common_train_label_changes"],
        )

        print(
            "Common validation:",
            f"{metrics['common_validation_n']:,}",
        )

        print(
            "AUC baseline / D1 / D2:",
            metrics["baseline_auc_common"],
            "/",
            metrics["d1_fixed_model_auc_common"],
            "/",
            metrics["d2_retrained_auc_common"],
        )

        print(
            "Mean probability delta D1 / D2:",
            metrics["d1_mean_abs_probability_delta"],
            "/",
            metrics["d2_mean_abs_probability_delta"],
        )

    summary = {
        "version": VERSION,
        "stage": STAGE,
        "status": "RESEARCH_DIAGNOSTIC_ONLY",
        "baseline_rows": int(len(base)),
        "scenario_rows": int(len(scenario)),
        "candidate_exclusions": int(exclusion_count),
        "training_cutoff": "2023-01-01",
        "validation_start": "2023-01-01",
        "validation_end_exclusive": "2025-01-01",
        "model_parameters": {
            "algorithm": "HistGradientBoostingClassifier",
            "max_iter": 180,
            "max_leaf_nodes": 15,
            "learning_rate": 0.06,
            "l2_regularization": 1.0,
            "random_state": 42,
        },
        "results": results,
        "policy": {
            "raw_prices_modified": False,
            "actual_engine_modified": False,
            "official_idx_calendar_verified": False,
            "production_model_replaced": False,
            "trading_readiness_established": False,
        },
        "limitations": [
            "Calendar exclusions are hypothetical.",
            "IHSG dates are not an official IDX calendar.",
            "Training changes combine sample, feature and label effects.",
            "AUC is evaluated on common eligible validation rows.",
            "This is a fixed historical split, not walk-forward.",
            "Training labels near the cutoff have not been "
            "separately purged for point-in-time maturity.",
            "Model probabilities are not proven calibrated.",
            "Opportunity Score and WATCH decisions are not tested.",
            "No execution costs, slippage or trading backtest.",
            "No future profitability is established.",
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)

    pd.concat(
        details, ignore_index=True
    ).to_csv(
        DETAIL_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved: {SUMMARY_FILE}")
    print(f"Saved: {DETAIL_FILE}")
    print("STATUS: RESEARCH_DIAGNOSTIC_COMPLETED")


if __name__ == "__main__":
    main()
