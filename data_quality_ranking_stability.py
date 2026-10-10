# V1.3F-10 — Stage G: Ranking Stability Audit
# RESEARCH DIAGNOSTIC ONLY
# No modification to raw prices, actual_engine.py or production signals.

from pathlib import Path
import hashlib
import json
import pandas as pd

VERSION = "V1.3F-10"
STAGE = "G_RANKING_STABILITY_AUDIT"

INPUT_CANDIDATES = [
    Path("data_quality_model_impact_detail.csv"),
    Path("data/execution_probe/data_quality_model_impact_detail.csv"),
    Path("data/engine_output/data_quality_model_impact_detail.csv"),
]

OUTPUT_DIR = Path("data/execution_probe")
OUTPUT_SUMMARY = OUTPUT_DIR / "data_quality_ranking_stability_summary.json"
OUTPUT_DETAIL = OUTPUT_DIR / "data_quality_ranking_stability_detail.csv"

REQUIRED_COLUMNS = [
    "ticker",
    "date",
    "target",
    "baseline_probability",
    "masked_input_probability",
    "retrained_probability",
]


def find_input():
    for path in INPUT_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Stage F detail CSV not found. Place "
        "data_quality_model_impact_detail.csv in the project root."
    )


def top_three(group, probability_column):
    ranked = group.sort_values(
        [probability_column, "ticker"],
        ascending=[False, True],
        kind="mergesort",
    )
    return ranked["ticker"].head(3).tolist()


def compare_rankings(group, date, target, comparison_name, column):
    baseline = top_three(group, "baseline_probability")
    comparison = top_three(group, column)

    baseline_set = set(baseline)
    comparison_set = set(comparison)

    overlap = len(baseline_set & comparison_set)

    return {
        "date": str(date),
        "target": int(target),
        "comparison": comparison_name,
        "eligible_stock_count": int(len(group)),
        "baseline_top3": "|".join(baseline),
        "comparison_top3": "|".join(comparison),
        "top3_overlap_count": overlap,
        "top3_membership_identical": baseline_set == comparison_set,
        "top3_order_identical": baseline == comparison,
        "top1_identical": baseline[0] == comparison[0],
        "stocks_removed": "|".join(sorted(baseline_set - comparison_set)),
        "stocks_added": "|".join(sorted(comparison_set - baseline_set)),
    }


def summarize(detail):
    results = []

    for (target, comparison), group in detail.groupby(
        ["target", "comparison"], sort=True
    ):
        days = len(group)
        identical = int(group["top3_membership_identical"].sum())
        order_identical = int(group["top3_order_identical"].sum())
        same_top1 = int(group["top1_identical"].sum())

        results.append({
            "target": int(target),
            "comparison": str(comparison),
            "validation_dates": days,
            "top3_identical_dates": identical,
            "top3_changed_dates": days - identical,
            "top3_identical_pct": 100 * identical / days,
            "top3_changed_pct": 100 * (days - identical) / days,
            "top3_order_identical_dates": order_identical,
            "top3_order_identical_pct": 100 * order_identical / days,
            "top1_identical_dates": same_top1,
            "top1_identical_pct": 100 * same_top1 / days,
            "mean_top3_overlap": float(
                group["top3_overlap_count"].mean()
            ),
        })

    return results


def main():
    input_path = find_input()
    print(f"Reading: {input_path}")

    data = pd.read_csv(input_path, dtype={"ticker": str, "date": str})

    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if data.empty:
        raise ValueError("Input file is empty.")

    if data[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Input contains missing required values.")

    data["target"] = pd.to_numeric(data["target"], errors="raise")

    if set(data["target"].unique()) != {3, 5, 8}:
        raise ValueError("Expected exactly targets 3, 5 and 8.")

    probability_columns = [
        "baseline_probability",
        "masked_input_probability",
        "retrained_probability",
    ]

    for col in probability_columns:
        data[col] = pd.to_numeric(data[col], errors="raise")
        if not data[col].between(0, 1).all():
            raise ValueError(f"Invalid probability values in {col}")

    if data.duplicated(["date", "target", "ticker"]).any():
        raise ValueError("Duplicate date-target-ticker combinations found.")

    records = []

    for (target, date), group in data.groupby(
        ["target", "date"], sort=True
    ):
        if len(group) < 3:
            raise ValueError(
                f"Fewer than 3 eligible stocks: {date}, target {target}"
            )

        records.append(compare_rankings(
            group, date, target, "F1_FIXED_MODEL",
            "masked_input_probability",
        ))

        records.append(compare_rankings(
            group, date, target, "F2_RETRAINED",
            "retrained_probability",
        ))

    detail = pd.DataFrame(records).sort_values(
        ["target", "date", "comparison"]
    )

    summary_results = summarize(detail)

    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()

    summary = {
        "version": VERSION,
        "stage": STAGE,
        "status": "RESEARCH_DIAGNOSTIC_ONLY",
        "input_file": input_path.name,
        "input_sha256": input_sha256,
        "input_rows": int(len(data)),
        "method": {
            "ranking": "Probability descending, ticker ascending for ties",
            "top_n": 3,
            "population": "Stage F common validation sample",
            "comparisons": [
                "F1_FIXED_MODEL",
                "F2_RETRAINED",
            ],
        },
        "results": summary_results,
        "limitations": [
            "Top-3 uses model probability, not production opportunity_score.",
            "Only common eligible validation observations are evaluated.",
            "No transaction costs or execution constraints.",
            "No trading performance or future profitability is established.",
            "Price masking is hypothetical, not a verified correction.",
        ],
        "policy": {
            "raw_prices_modified": False,
            "actual_engine_modified": False,
            "production_model_replaced": False,
            "trading_readiness_established": False,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detail.to_csv(OUTPUT_DETAIL, index=False)

    with OUTPUT_SUMMARY.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)

    print("\n=== V1.3F-10 STAGE G ===")
    print(f"Input rows: {len(data):,}")
    print(f"Dates: {data['date'].nunique():,}")

    for result in summary_results:
        print(
            f"Target {result['target']}% "
            f"| {result['comparison']} "
            f"| Top-3 changed: {result['top3_changed_pct']:.2f}% "
            f"| Top-1 stable: {result['top1_identical_pct']:.2f}% "
            f"| Mean overlap: {result['mean_top3_overlap']:.4f}"
        )

    print(f"\nSaved: {OUTPUT_SUMMARY}")
    print(f"Saved: {OUTPUT_DETAIL}")
    print("STATUS: RESEARCH DIAGNOSTIC COMPLETED")


if __name__ == "__main__":
    main()