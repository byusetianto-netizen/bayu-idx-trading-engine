
"""
V1.3F-10 — Baseline Parity Gate

Research-only validation.

Recreates baseline model training using the Stage E
feature/label calculation and compares results with
actual_engine.py outputs from the same environment.

Run actual_engine.py BEFORE running this script.

No source data or production engine modification.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from data_quality_numerical_impact import (
    FEATURES,
    load_prices,
    load_ihsg,
    calculate_model_inputs,
)


ROOT = Path(__file__).resolve().parent

REFERENCE_FILE = (
    ROOT / "data" / "engine_output"
    / "model_validation_metrics.csv"
)

OUT_DIR = ROOT / "data" / "execution_probe"
OUT_FILE = OUT_DIR / "data_quality_baseline_parity.json"

TARGETS = [3, 5, 8]
AUC_TOLERANCE = 1e-8


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


def calculate_research_metrics(df):
    train = df["date"] < pd.Timestamp("2023-01-01")

    valid = (
        (df["date"] >= pd.Timestamp("2023-01-01"))
        & (df["date"] < pd.Timestamp("2025-01-01"))
    )

    results = []

    for target in TARGETS:
        label_name = f"y{target}"

        x_train = df.loc[train, FEATURES]
        y_train = df.loc[train, label_name]

        train_mask = (
            y_train.notna()
            & x_train.notna().all(axis=1)
        )

        x_valid = df.loc[valid, FEATURES]
        y_valid = df.loc[valid, label_name]

        valid_mask = (
            x_valid.notna().all(axis=1)
            & y_valid.notna()
        )

        require(
            train_mask.sum() > 0,
            f"No training samples for {label_name}",
        )

        y_fit = y_train.loc[train_mask].astype(int)

        require(
            y_fit.nunique() == 2,
            f"Training target has one class: {label_name}",
        )

        model = make_model()

        model.fit(
            x_train.loc[train_mask],
            y_fit,
        )

        y_eval = y_valid.loc[valid_mask].astype(int)

        require(
            len(y_eval) > 0,
            f"No validation samples for {label_name}",
        )

        prediction = model.predict_proba(
            x_valid.loc[valid_mask]
        )[:, 1]

        if y_eval.nunique() > 1:
            auc = float(
                roc_auc_score(y_eval, prediction)
            )
        else:
            auc = None

        results.append({
            "target": target,
            "train_n": int(train_mask.sum()),
            "validation_n": int(valid_mask.sum()),
            "validation_auc": auc,
        })

    return results


def load_reference():
    require(
        REFERENCE_FILE.exists(),
        (
            "Reference metrics missing. "
            "Run actual_engine.py first."
        ),
    )

    reference = pd.read_csv(REFERENCE_FILE)

    expected_columns = {
        "target",
        "train_n",
        "validation_auc",
    }

    require(
        expected_columns.issubset(reference.columns),
        "Actual engine metrics columns do not match.",
    )

    reference["target"] = pd.to_numeric(
        reference["target"], errors="raise"
    ).astype(int)

    require(
        reference["target"].is_unique,
        "Duplicate reference targets.",
    )

    require(
        set(reference["target"]) == set(TARGETS),
        "Unexpected reference targets.",
    )

    return reference.set_index("target")


def main():
    print("V1.3F-10 — Baseline Parity Gate")
    print("------------------------------")

    prices = load_prices()
    ihsg = load_ihsg()

    research = calculate_model_inputs(prices, ihsg)
    reference = load_reference()

    require(
        len(research) == len(prices),
        "Research model input row count differs.",
    )

    metrics = calculate_research_metrics(research)

    comparisons = []
    failures = []

    for row in metrics:
        target = row["target"]
        ref = reference.loc[target]

        expected_n = int(ref["train_n"])
        actual_n = row["train_n"]

        n_match = expected_n == actual_n

        ref_auc = (
            None
            if pd.isna(ref["validation_auc"])
            else float(ref["validation_auc"])
        )

        research_auc = row["validation_auc"]

        if ref_auc is None or research_auc is None:
            auc_match = (
                ref_auc is None
                and research_auc is None
            )
            auc_difference = None
        else:
            auc_difference = abs(
                ref_auc - research_auc
            )
            auc_match = (
                auc_difference <= AUC_TOLERANCE
            )

        passed = bool(n_match and auc_match)

        comparisons.append({
            "target": target,
            "reference_train_n": expected_n,
            "research_train_n": actual_n,
            "research_validation_n": row[
                "validation_n"
            ],
            "reference_auc": ref_auc,
            "research_auc": research_auc,
            "auc_absolute_difference": auc_difference,
            "train_n_match": bool(n_match),
            "auc_match": bool(auc_match),
            "passed": passed,
        })

        print(
            f"Target {target}%: "
            f"train_n {expected_n} vs {actual_n}, "
            f"AUC {ref_auc} vs {research_auc}"
        )

        if not passed:
            failures.append(target)

    summary = {
        "version": "V1.3F-10",
        "stage": "F_BASELINE_PARITY_GATE",
        "status": (
            "PASS" if not failures else "FAIL"
        ),
        "auc_tolerance": AUC_TOLERANCE,
        "source": "actual_engine.py metrics",
        "reference_file": str(
            REFERENCE_FILE.relative_to(ROOT)
        ),
        "results": comparisons,
        "limitations": [
            "Reference and research models must run "
            "in the same environment and on the same data.",
            "AUC parity and training count parity do not "
            "independently prove every feature cell is equal.",
            "This gate does not validate data provenance, "
            "point-in-time integrity or trading profitability.",
            "Validation sample count is reported from "
            "the research implementation.",
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        OUT_FILE, "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print()
    print("BASELINE PARITY:", summary["status"])

    if failures:
        raise RuntimeError(
            f"Baseline parity failed for targets: "
            f"{failures}"
        )


if __name__ == "__main__":
    main()
