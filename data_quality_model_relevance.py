
"""
V1.3F-10 — Stage D
Model-Relevant Component Exposure Audit

RESEARCH ONLY.

Uses Stage B dependency masks and Stage C OHLC classifications.
Separates observed violations from unverified component faults.

No price correction, row deletion, training, or engine modification.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROBE = ROOT / "data" / "execution_probe"

STAGE_B = PROBE / "data_quality_exact_dependency_detail.csv"
STAGE_C = PROBE / "data_quality_component_audit_detail.csv"

DETAIL_OUT = PROBE / "data_quality_model_relevance_detail.csv"
SUMMARY_OUT = PROBE / "data_quality_model_relevance_summary.json"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_boolean(series, name):
    values = series.astype("string").str.strip().str.lower()
    require(
        values.isin(["true", "false"]).all(),
        f"Unexpected boolean values in {name}",
    )
    return values.eq("true").to_numpy(dtype=bool)


def load_inputs():
    require(STAGE_B.exists(), f"Missing Stage B: {STAGE_B}")
    require(STAGE_C.exists(), f"Missing Stage C: {STAGE_C}")

    b = pd.read_csv(STAGE_B, parse_dates=["date"])
    c = pd.read_csv(STAGE_C, parse_dates=["date"])

    keys = ["ticker", "date"]

    for name, df in [("Stage B", b), ("Stage C", c)]:
        require(
            set(keys).issubset(df.columns),
            f"{name} missing ticker/date",
        )
        require(
            not df.duplicated(keys).any(),
            f"{name} contains duplicate ticker/date",
        )

    required_b = {
        "quality_review_anchor",
        "period",
        "any_feature_dependency",
        "any_label_dependency",
        "any_dependency",
    }

    required_c = {
        "invalid_ohlc",
        "scale_pattern_class",
        "observed_violation_types",
        "computed_ohlc_violation",
        "stage_a_invalid_vs_computed_match",
    }

    require(
        required_b.issubset(b.columns),
        "Stage B missing required columns",
    )
    require(
        required_c.issubset(c.columns),
        "Stage C missing required columns",
    )

    feature_cols = sorted(
        col for col in b.columns if col.startswith("dep_")
        and col not in {
            "dep_label_close",
            "dep_label_future_hilo",
        }
    )

    require(
        len(feature_cols) == 11,
        "Expected exactly 11 model-feature dependency columns",
    )

    for col in [
        "quality_review_anchor",
        "any_feature_dependency",
        "any_label_dependency",
        "any_dependency",
    ] + feature_cols:
        b[col] = read_boolean(b[col], col)

    for col in [
        "invalid_ohlc",
        "computed_ohlc_violation",
        "stage_a_invalid_vs_computed_match",
    ]:
        c[col] = read_boolean(c[col], col)

    require(
        b["quality_review_anchor"].sum() == len(c),
        "Stage C must contain exactly the Stage B anchor count",
    )

    anchors = b.loc[
        b["quality_review_anchor"],
        keys,
    ].sort_values(keys).reset_index(drop=True)

    ckeys = c[keys].sort_values(keys).reset_index(drop=True)

    require(
        anchors.equals(ckeys),
        "Stage B and Stage C anchor keys do not match",
    )

    require(
        c["stage_a_invalid_vs_computed_match"].all(),
        "Stage C contains Stage A classification mismatches",
    )

    return b, c, feature_cols


def classify_anchor(row):
    """
    Classification describes the observed violation.

    A violation cannot establish whether Open, High,
    Low or Close is the incorrect source component.
    """
    violation = str(row["observed_violation_types"])

    if row["computed_ohlc_violation"]:
        parts = set(violation.split("|"))

        open_parts = {
            "open_below_low",
            "open_above_high",
        }
        close_parts = {
            "close_below_low",
            "close_above_high",
        }

        if parts.issubset(open_parts):
            return "OPEN_RANGE_VIOLATION"

        if parts.issubset(close_parts):
            return "CLOSE_RANGE_VIOLATION"

        return "MIXED_OR_OTHER_OHLC_VIOLATION"

    if row["scale_pattern_class"] == "STRONG_SCALE_PATTERN":
        return "STRONG_SCALE_PATTERN_ONLY"

    return "OTHER_REVIEW_ANCHOR"


def build_anchor_detail(c):
    result = c.copy()
    result["review_class"] = result.apply(
        classify_anchor, axis=1
    )

    result["model_relevant_fault_possible"] = True
    result["model_relevant_fault_confirmed"] = False
    result["specific_faulty_component_verified"] = False
    result["automatic_correction_allowed"] = False

    result["reasoning"] = np.select(
        [
            result["review_class"].eq(
                "OPEN_RANGE_VIOLATION"
            ),
            result["review_class"].eq(
                "CLOSE_RANGE_VIOLATION"
            ),
            result["review_class"].eq(
                "STRONG_SCALE_PATTERN_ONLY"
            ),
        ],
        [
            "Open is outside High-Low; Open or boundary "
            "components may be inconsistent.",
            "Close is outside High-Low; Close or boundary "
            "components may be inconsistent.",
            "Relative price-scale pattern requires "
            "independent adjustment-basis verification.",
        ],
        default=(
            "Further component and source verification "
            "required."
        ),
    )

    return result


def count_flags(df, columns):
    return {
        col: int(df[col].sum())
        for col in columns
    }


def main():
    print("V1.3F-10 — Stage D Model Relevance Audit")
    print("---------------------------------------")

    b, c, feature_cols = load_inputs()
    detail = build_anchor_detail(c)

    classes = {
        str(k): int(v)
        for k, v in detail["review_class"]
        .value_counts()
        .items()
    }

    periods = {}

    for period, part in b.groupby("period"):
        periods[str(period)] = {
            "rows": int(len(part)),
            "any_feature_dependency": int(
                part["any_feature_dependency"].sum()
            ),
            "any_label_dependency": int(
                part["any_label_dependency"].sum()
            ),
            "any_dependency": int(
                part["any_dependency"].sum()
            ),
            "feature_dependencies": count_flags(
                part, feature_cols
            ),
        }

    validation = b[
        (b["period"] == "VALIDATION_PERIOD")
        & b["any_dependency"]
    ]

    validation_tickers = {
        str(ticker): int(len(part))
        for ticker, part in validation.groupby("ticker")
    }

    summary = {
        "version": "V1.3F-10",
        "stage": "D_MODEL_RELEVANT_COMPONENT_EXPOSURE",
        "status": "RESEARCH_DIAGNOSTIC_ONLY",
        "review_anchor_count": int(len(detail)),
        "anchor_classes": classes,
        "model_relevant_fault_confirmed": 0,
        "periods": periods,
        "validation_dependency_tickers": validation_tickers,
        "policy": {
            "price_correction_allowed": False,
            "observations_deleted": False,
            "actual_engine_modified": False,
            "model_retrained": False,
            "numeric_feature_impact_measured": False,
            "numeric_label_impact_measured": False,
        },
        "limitations": [
            "Open outside High-Low does not prove Open "
            "rather than High or Low is wrong.",
            "Close outside High-Low does not prove Close "
            "rather than High or Low is wrong.",
            "Scale-pattern evidence does not establish "
            "the correct adjustment basis.",
            "Stage B dependencies assume every review "
            "anchor could affect Close, High and Low.",
            "Validation dependency counts are inherited "
            "from Stage B, not causal attribution by class.",
            "No verified replacement prices are available.",
            "No numeric counterfactual or AUC test "
            "has been conducted.",
        ],
    }

    PROBE.mkdir(parents=True, exist_ok=True)

    detail[
        [
            "ticker",
            "date",
            "open",
            "high",
            "low",
            "close",
            "review_class",
            "observed_violation_types",
            "scale_pattern_class",
            "model_relevant_fault_possible",
            "model_relevant_fault_confirmed",
            "specific_faulty_component_verified",
            "automatic_correction_allowed",
            "reasoning",
        ]
    ].to_csv(
        DETAIL_OUT,
        index=False,
        date_format="%Y-%m-%d",
    )

    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Review anchors:", len(detail))
    print()
    print("Anchor classifications:")

    for key, value in sorted(classes.items()):
        print(f"  {key}: {value}")

    print()
    print("Validation dependency rows:")

    for ticker, count in sorted(validation_tickers.items()):
        print(f"  {ticker}: {count}")

    print()
    print("Confirmed faulty components: 0")
    print("STATUS: RESEARCH_DIAGNOSTIC_ONLY")


if __name__ == "__main__":
    main()
