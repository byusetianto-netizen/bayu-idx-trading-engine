
"""
Bayu IDX Trading Engine
V1.3F-11 Stage C1 — Feature & Label Baseline Parity

RESEARCH ONLY

Compare the Stage B baseline against the V1.3F-10 reference
implementation previously used for actual_engine.py model parity.

No raw price changes.
No model retraining.
No production model changes.
No sklearn dependency.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from data_quality_numerical_impact import (
    FEATURES,
    load_prices,
    load_ihsg,
    calculate_model_inputs,
)

from v13f11_calendar_sensitivity import (
    read_sources,
    calculate_components,
)


VERSION = "V1.3F-11"
STAGE = "C1_FEATURE_LABEL_BASELINE_PARITY"

OUTPUT = Path(
    "data/execution_probe/"
    "v13f11_baseline_parity_summary.json"
)

TARGETS = [3, 5, 8]
LABELS = [f"y{target}" for target in TARGETS]
COMPONENTS = list(FEATURES) + LABELS

RTOL = 1e-9
ATOL = 1e-10


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def prepare(frame, name):
    """Validate and normalize comparison keys."""
    require(
        {"ticker", "date"}.issubset(frame.columns),
        f"{name}: missing ticker/date",
    )

    data = frame.copy()
    data["ticker"] = (
        data["ticker"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    ).dt.normalize()

    require(
        data[["ticker", "date"]].notna().all().all(),
        f"{name}: invalid keys",
    )

    require(
        not data.duplicated(["ticker", "date"]).any(),
        f"{name}: duplicate ticker/date",
    )

    missing = set(COMPONENTS) - set(data.columns)

    require(
        not missing,
        f"{name}: missing components {sorted(missing)}",
    )

    return data[
        ["ticker", "date"] + COMPONENTS
    ].sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)


def compare_components(reference, candidate):
    require(
        len(reference) == len(candidate),
        "Reference and candidate row counts differ",
    )

    keys = ["ticker", "date"]

    require(
        reference[keys].equals(candidate[keys]),
        "Ticker/date row parity failed",
    )

    results = []
    failures = []

    for component in COMPONENTS:
        a = reference[component].to_numpy(dtype=float)
        b = candidate[component].to_numpy(dtype=float)

        a_nan = np.isnan(a)
        b_nan = np.isnan(b)

        missing_mismatch = a_nan ^ b_nan

        a_inf = np.isinf(a)
        b_inf = np.isinf(b)

        infinity_mismatch = (
            (a_inf ^ b_inf)
            | (
                a_inf
                & b_inf
                & (np.signbit(a) != np.signbit(b))
            )
        )

        both_finite = np.isfinite(a) & np.isfinite(b)

        numeric_mismatch = np.zeros(
            len(a), dtype=bool
        )

        if component in LABELS:
            numeric_mismatch[both_finite] = (
                a[both_finite] != b[both_finite]
            )
        else:
            numeric_mismatch[both_finite] = ~np.isclose(
                a[both_finite],
                b[both_finite],
                rtol=RTOL,
                atol=ATOL,
            )

        mismatched = (
            missing_mismatch
            | infinity_mismatch
            | numeric_mismatch
        )

        # A finite/infinite difference must also fail.
        mismatched |= (
            np.isfinite(a) ^ np.isfinite(b)
        )

        count = int(mismatched.sum())

        finite_delta = (
            np.abs(a[both_finite] - b[both_finite])
        )

        maximum_delta = (
            float(finite_delta.max())
            if len(finite_delta) else None
        )

        result = {
            "component": component,
            "compared_rows": int(len(a)),
            "missing_mismatch_rows": int(
                missing_mismatch.sum()
            ),
            "infinity_mismatch_rows": int(
                infinity_mismatch.sum()
            ),
            "numeric_mismatch_rows": int(
                numeric_mismatch.sum()
            ),
            "total_mismatch_rows": count,
            "max_abs_delta_on_finite_rows": maximum_delta,
            "passed": count == 0,
        }

        results.append(result)

        print(
            f"{component:15s} "
            f"mismatches={count:,} "
            f"max_abs_delta={maximum_delta}"
        )

        if count:
            failures.append(component)

    return results, failures


def main():
    print("=== V1.3F-11 STAGE C1 ===")

    # V1.3F-10 reference: already used in prior
    # model parity testing against actual_engine.py.
    reference_prices = load_prices()
    reference_ihsg = load_ihsg()

    reference = calculate_model_inputs(
        reference_prices,
        reference_ihsg,
        mask_missing_close_label=False,
    )

    # Stage B baseline: no candidate rows excluded.
    stage_b_stock, stage_b_ihsg, exclusion_count = (
        read_sources()
    )

    candidate = calculate_components(
        stage_b_stock,
        stage_b_ihsg,
    )

    ref = prepare(reference, "V1.3F-10 reference")
    test = prepare(candidate, "V1.3F-11 Stage B")

    require(
        len(ref) == len(reference_prices),
        "Reference row accounting failed",
    )

    require(
        len(test) == len(stage_b_stock),
        "Stage B row accounting failed",
    )

    results, failures = compare_components(
        ref,
        test,
    )

    summary = {
        "version": VERSION,
        "stage": STAGE,
        "status": "PASS" if not failures else "FAIL",
        "reference": (
            "V1.3F-10 calculate_model_inputs"
        ),
        "candidate": (
            "V1.3F-11 Stage B calculate_components"
        ),
        "reference_rows": int(len(ref)),
        "candidate_rows": int(len(test)),
        "scenario_exclusion_candidates": int(
            exclusion_count
        ),
        "feature_count": len(FEATURES),
        "label_count": len(LABELS),
        "feature_rtol": RTOL,
        "feature_atol": ATOL,
        "label_comparison": "EXACT",
        "failed_components": failures,
        "results": results,
        "policy": {
            "raw_prices_modified": False,
            "actual_engine_modified": False,
            "model_retrained": False,
            "auc_measured": False,
            "production_ready": False,
        },
        "limitations": [
            "This checks feature and label parity only.",
            "Model/AUC parity is a separate GitHub Actions gate.",
            "Passing does not verify official IDX calendar.",
            "Passing does not establish trading profitability.",
        ],
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print()
    print("Rows:", len(ref))
    print("Components:", len(COMPONENTS))
    print("Failed components:", failures)
    print("BASELINE FEATURE/LABEL PARITY:", summary["status"])
    print("Saved:", OUTPUT)

    if failures:
        raise RuntimeError(
            "Stage C1 baseline parity failed: "
            + ", ".join(failures)
        )


if __name__ == "__main__":
    main()
