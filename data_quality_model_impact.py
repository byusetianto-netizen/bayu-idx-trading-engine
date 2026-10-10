
"""
V1.3F-10 Stage F — Model Impact Assessment

RESEARCH ONLY.

F1: Fixed-model comparison on common eligible validation rows.
F2: Controlled retraining on masked data, evaluated on common rows.

This experiment measures masking sensitivity, NOT verified
historical pricing error or live trading performance.

Does not modify raw prices or actual_engine.py.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from data_quality_numerical_impact import (
    FEATURES,
    LABELS,
    load_prices,
    load_ihsg,
    load_review,
    make_scenario,
    calculate_model_inputs,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "execution_probe"

SUMMARY_OUT = OUT / "data_quality_model_impact_summary.json"
DETAIL_OUT = OUT / "data_quality_model_impact_detail.csv"

TARGETS = [3, 5, 8]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def build_pipeline():
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


def eligible(df, target):
    x = df[FEATURES]
    y = df[f"y{target}"]

    return (
        x.notna().all(axis=1)
        & np.isfinite(x.to_numpy(dtype=float)).all(axis=1)
        & y.notna()
    )


def auc_or_none(y, prediction):
    if len(y) == 0 or y.nunique() < 2:
        return None

    return float(
        roc_auc_score(
            y.astype(int),
            prediction,
        )
    )


def fit_model(df, target, train_period):
    mask = eligible(df, target) & train_period

    x = df.loc[mask, FEATURES]
    y = df.loc[mask, f"y{target}"].astype(int)

    require(
        len(x) > 0,
        f"No eligible training rows for y{target}"
    )

    require(
        y.nunique() == 2,
        f"Training target y{target} has one class"
    )

    pipe = build_pipeline()
    pipe.fit(x, y)

    return pipe, mask


def run_target(base, scenario, target):
    train = base["date"] < pd.Timestamp("2023-01-01")

    valid = (
        (base["date"] >= pd.Timestamp("2023-01-01"))
        & (base["date"] < pd.Timestamp("2025-01-01"))
    )

    base_eligible = eligible(base, target)
    scenario_eligible = eligible(scenario, target)

    base_train = base_eligible & train
    scenario_train = scenario_eligible & train

    base_valid = base_eligible & valid
    scenario_valid = scenario_eligible & valid

    common_valid = base_valid & scenario_valid

    require(
        common_valid.sum() > 0,
        f"No common validation rows for y{target}"
    )

    # F1: Model trained on baseline only.
    baseline_model, _ = fit_model(base, target, train)

    x_base = base.loc[common_valid, FEATURES]
    x_scenario = scenario.loc[common_valid, FEATURES]

    y_base = base.loc[
        common_valid, f"y{target}"
    ].astype(int)

    y_scenario = scenario.loc[
        common_valid, f"y{target}"
    ].astype(int)

    # Changing labels can make AUC incomparable.
    label_changed = (
        y_base.to_numpy() != y_scenario.to_numpy()
    )

    require(
        not label_changed.any(),
        (
            f"y{target} differs on common validation rows. "
            "Separate label-change analysis is required."
        )
    )

    p_base = baseline_model.predict_proba(x_base)[:, 1]
    p_masked_input = baseline_model.predict_proba(
        x_scenario
    )[:, 1]

    # F2: Retrain on the scenario.
    scenario_model, _ = fit_model(
        scenario, target, train
    )

    p_retrained = scenario_model.predict_proba(
        x_scenario
    )[:, 1]

    f1_delta = np.abs(p_masked_input - p_base)
    f2_delta = np.abs(p_retrained - p_base)

    # Both comparisons use the identical validation rows.
    auc_base = auc_or_none(y_base, p_base)
    auc_f1 = auc_or_none(y_base, p_masked_input)
    auc_f2 = auc_or_none(y_base, p_retrained)

    detail = base.loc[
        common_valid, ["ticker", "date"]
    ].copy()

    detail["target"] = target
    detail["label"] = y_base.to_numpy()
    detail["baseline_probability"] = p_base
    detail["masked_input_probability"] = p_masked_input
    detail["retrained_probability"] = p_retrained
    detail["f1_abs_probability_delta"] = f1_delta
    detail["f2_abs_probability_delta"] = f2_delta

    # Rank correlation measures stability of relative scores.
    rank_f1 = float(
        pd.Series(p_base).corr(
            pd.Series(p_masked_input),
            method="spearman",
        )
    )

    rank_f2 = float(
        pd.Series(p_base).corr(
            pd.Series(p_retrained),
            method="spearman",
        )
    )

    metrics = {
        "target": target,
        "baseline_train_n": int(base_train.sum()),
        "scenario_train_n": int(scenario_train.sum()),
        "baseline_validation_n": int(base_valid.sum()),
        "scenario_validation_n": int(
            scenario_valid.sum()
        ),
        "common_validation_n": int(common_valid.sum()),
        "baseline_only_validation_n": int(
            (base_valid & ~scenario_valid).sum()
        ),
        "scenario_only_validation_n": int(
            (scenario_valid & ~base_valid).sum()
        ),
        "common_validation_label_changes": int(
            label_changed.sum()
        ),
        "baseline_auc_common": auc_base,
        "f1_masked_input_auc_common": auc_f1,
        "f2_retrained_auc_common": auc_f2,
        "f1_changed_probabilities": int(
            np.count_nonzero(f1_delta > 1e-10)
        ),
        "f2_changed_probabilities": int(
            np.count_nonzero(f2_delta > 1e-10)
        ),
        "f1_mean_abs_probability_delta": float(
            f1_delta.mean()
        ),
        "f2_mean_abs_probability_delta": float(
            f2_delta.mean()
        ),
        "f1_max_abs_probability_delta": float(
            f1_delta.max()
        ),
        "f2_max_abs_probability_delta": float(
            f2_delta.max()
        ),
        "f1_spearman": rank_f1,
        "f2_spearman": rank_f2,
    }

    return metrics, detail


def main():
    print("V1.3F-10 — Stage F Model Impact Assessment")
    print("-----------------------------------------")

    prices = load_prices()
    ihsg = load_ihsg()
    review = load_review()

    scenario_prices, mask_counts = make_scenario(
        prices, review
    )

    base = calculate_model_inputs(prices, ihsg)

    scenario = calculate_model_inputs(
        scenario_prices,
        ihsg,
        mask_missing_close_label=True,
    )

    require(
        base[["ticker", "date"]].equals(
            scenario[["ticker", "date"]]
        ),
        "Baseline/scenario alignment failed"
    )

    results = []
    details = []

    for target in TARGETS:
        print(f"Testing target {target}% ...")

        metrics, detail = run_target(
            base, scenario, target
        )

        results.append(metrics)
        details.append(detail)

        print(
            f"  Train: "
            f"{metrics['baseline_train_n']:,} -> "
            f"{metrics['scenario_train_n']:,}"
        )

        print(
            f"  Common validation: "
            f"{metrics['common_validation_n']:,}"
        )

        print(
            "  AUC baseline/F1/F2: "
            f"{metrics['baseline_auc_common']} / "
            f"{metrics['f1_masked_input_auc_common']} / "
            f"{metrics['f2_retrained_auc_common']}"
        )

    OUT.mkdir(parents=True, exist_ok=True)

    pd.concat(
        details, ignore_index=True
    ).to_csv(
        DETAIL_OUT,
        index=False,
        date_format="%Y-%m-%d",
    )

    summary = {
        "version": "V1.3F-10",
        "stage": "F_MODEL_IMPACT_ASSESSMENT",
        "status": "RESEARCH_DIAGNOSTIC_ONLY",
        "baseline_rows": int(len(base)),
        "mask_counts": mask_counts,
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
            "verified_price_correction": False,
            "production_model_replaced": False,
            "trading_readiness_established": False,
        },
        "limitations": [
            "Masking is an experimental assumption, "
            "not verified price correction.",
            "The training eligibility criteria are "
            "matched to actual_engine.py.",
            "The scenario safety rule treats a masked "
            "current Close as an unknown label.",
            "Fixed-model and retraining results must "
            "be interpreted separately.",
            "AUC uses the intersection of eligible "
            "validation rows.",
            "Historical AUC does not establish "
            "future profitability.",
            "No costs, slippage or execution backtest "
            "are included.",
        ],
    }

    with open(
        SUMMARY_OUT, "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print()
    print("STATUS: RESEARCH_DIAGNOSTIC_ONLY")


if __name__ == "__main__":
    main()
