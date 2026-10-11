
"""
Bayu IDX Trading Engine
V1.3F-11 Stage F4 - Out-of-Sample Confirmation

RESEARCH ONLY.

Confirm the pre-specified 8% Top-3 predictive-lift
hypothesis on weekly decisions from 2025 onward.

Reuse Stage E2 training, scoring and eligibility rules.
No production changes, trading orders or price repair.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v13f11_calendar_sensitivity import read_sources

from v13f11_weekly_retraining import (
    TARGETS,
    KEYS,
    require,
    prepare,
    common_candidates,
    candidate_mask,
    fit_target,
    score_week,
    market_regime,
)


OUT = Path("data/execution_probe")

SUMMARY = OUT / "v13f11_oos_confirmation_summary.json"
DETAIL = OUT / "v13f11_oos_confirmation_detail.csv"
WEEKLY = OUT / "v13f11_oos_confirmation_weekly.csv"

OOS_START = pd.Timestamp("2025-01-01")
PRIMARY_TARGET = 8

EXPECTED_EXCLUSIONS = 5239


def select_oos_dates(base, scenario):
    """
    First common available date in each Monday-Sunday week.

    Require at least five later common-market dates.
    Individual ticker horizons are checked separately.
    """
    base_dates = pd.DatetimeIndex(
        base["date"].dropna().unique()
    )

    scenario_dates = pd.DatetimeIndex(
        scenario["date"].dropna().unique()
    )

    common_dates = (
        base_dates.intersection(scenario_dates)
        .sort_values()
    )

    require(
        len(common_dates) > 5,
        "Insufficient common historical dates",
    )

    # The last five common dates cannot be decision dates.
    safe_dates = common_dates[:-5]

    safe_dates = safe_dates[
        safe_dates >= OOS_START
    ]

    require(
        len(safe_dates) > 0,
        "No eligible OOS dates",
    )

    calendar = pd.DataFrame({
        "decision_date": safe_dates
    })

    calendar["week"] = (
        calendar["decision_date"]
        .dt.to_period("W-SUN")
    )

    dates = (
        calendar.groupby("week", sort=True)
        ["decision_date"]
        .min()
        .reset_index(drop=True)
    )

    require(
        not dates.empty,
        "No OOS weekly decisions",
    )

    require(
        dates.ge(OOS_START).all(),
        "OOS dates include pre-2025 observations",
    )

    return dates


def attach_historical_labels(scored, prepared, mode):
    """
    Attach outcomes only after scoring.

    The five-session future labels are used for
    evaluation, never for deciding the Top-3.
    """
    columns = ["ticker", "date"] + [
        f"y{target}" for target in TARGETS
    ]

    labels = prepared[columns].rename(
        columns={"date": "decision_date"}
    )

    require(
        not labels.duplicated(
            ["decision_date", "ticker"]
        ).any(),
        f"{mode}: duplicate historical labels",
    )

    result = scored.merge(
        labels,
        on=["decision_date", "ticker"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    require(
        result["_merge"].eq("both").all(),
        f"{mode}: missing historical price rows",
    )

    return result.drop(columns="_merge")


def score_decision(base, scenario, ihsg, date):
    b, s = common_candidates(
        base, scenario, date
    )

    keep = (
        candidate_mask(b)
        & candidate_mask(s)
    )

    b = b.loc[keep].reset_index(drop=True)
    s = s.loc[keep].reset_index(drop=True)

    require(
        len(b) >= 3,
        f"{date}: fewer than three candidates",
    )

    require(
        b[KEYS].equals(s[KEYS]),
        f"{date}: candidate misalignment",
    )

    b_models = {}
    s_models = {}

    for target in TARGETS:
        b_models[target], _ = fit_target(
            base, target, date
        )

        s_models[target], _ = fit_target(
            scenario, target, date
        )

    b_score = score_week(b, b_models)
    s_score = score_week(s, s_models)

    require(
        len(b_score) == len(s_score),
        "Scored candidate count differs",
    )

    b_rank = {
        ticker: rank
        for rank, ticker in enumerate(
            b_score["ticker"], start=1
        )
    }

    s_rank = {
        ticker: rank
        for rank, ticker in enumerate(
            s_score["ticker"], start=1
        )
    }

    require(
        set(b_rank) == set(s_rank),
        "Scored ticker universe differs",
    )

    b_index = b_score.set_index("ticker")
    s_index = s_score.set_index("ticker")

    regime = market_regime(ihsg, date)
    records = []

    for ticker in sorted(b_rank):
        br = b_index.loc[ticker]
        sr = s_index.loc[ticker]

        records.append({
            "decision_date": date,
            "ticker": ticker,
            "market_regime": regime,
            "baseline_rank": b_rank[ticker],
            "scenario_rank": s_rank[ticker],
            "baseline_score": float(
                br["opportunity_score"]
            ),
            "scenario_score": float(
                sr["opportunity_score"]
            ),
            "baseline_top3": b_rank[ticker] <= 3,
            "scenario_top3": s_rank[ticker] <= 3,
            "baseline_decision": br["decision"],
            "scenario_decision": sr["decision"],
            "baseline_p3": float(br["p3"]),
            "scenario_p3": float(sr["p3"]),
            "baseline_p5": float(br["p5"]),
            "scenario_p5": float(sr["p5"]),
            "baseline_p8": float(br["p8"]),
            "scenario_p8": float(sr["p8"]),
        })

    return pd.DataFrame(records)


def evaluate_weekly(detail):
    records = []

    for date, group in detail.groupby(
        "decision_date", sort=True
    ):
        record = {
            "decision_date": date,
            "year": int(date.year),
            "market_regime": group[
                "market_regime"
            ].iloc[0],
            "candidate_count": int(len(group)),
        }

        for mode in ["baseline", "scenario"]:
            top = group[f"{mode}_top3"]

            require(
                int(top.sum()) == 3,
                f"{date}: invalid {mode} Top-3",
            )

            for target in TARGETS:
                column = f"{mode}_y{target}"
                key = f"{mode}_target{target}"

                top_values = (
                    group.loc[top, column].dropna()
                )

                other_values = (
                    group.loc[~top, column].dropna()
                )

                top_n = int(len(top_values))
                other_n = int(len(other_values))
                record[f"{key}_top_unknown"] = (
                    int(top.sum()) - top_n
                )
                record[f"{key}_other_unknown"] = (
                    int((~top).sum()) - other_n
                )
                record[f"{key}_complete_week"] = (
                    top_n == 3
                    and other_n == int((~top).sum())
                )

                top_success = int(
                    top_values.eq(1).sum()
                )

                other_success = int(
                    other_values.eq(1).sum()
                )

                top_rate = (
                    top_success / top_n
                    if top_n else np.nan
                )

                other_rate = (
                    other_success / other_n
                    if other_n else np.nan
                )

                record[f"{key}_top_known"] = top_n
                record[f"{key}_other_known"] = other_n

                record[f"{key}_top_success"] = (
                    top_success
                )

                record[f"{key}_other_success"] = (
                    other_success
                )

                record[f"{key}_top_rate"] = top_rate
                record[f"{key}_other_rate"] = other_rate

                record[f"{key}_lift_pp"] = (
                    (top_rate - other_rate) * 100
                    if np.isfinite(top_rate)
                    and np.isfinite(other_rate)
                    else np.nan
                )

        records.append(record)

    return pd.DataFrame(records)


def summarize(weekly, detail, mode, target):
    key = f"{mode}_target{target}"

    complete = weekly[
        f"{key}_complete_week"
    ].eq(True)

    valid = weekly.loc[
        complete,
        f"{key}_lift_pp"
    ].dropna()

    top_flag = detail[f"{mode}_top3"]

    top_values = detail.loc[
        top_flag, f"{mode}_y{target}"
    ].dropna()

    other_values = detail.loc[
        ~top_flag, f"{mode}_y{target}"
    ].dropna()

    return {
        "top3_total_n": int(top_flag.sum()),
        "top3_known_n": int(len(top_values)),
        "top3_unknown_n": int(
            top_flag.sum() - len(top_values)
        ),
        "top3_success_n": int(
            top_values.eq(1).sum()
        ),
        "top3_success_rate": (
            float(top_values.mean())
            if len(top_values) else None
        ),
        "other_total_n": int((~top_flag).sum()),
        "other_known_n": int(len(other_values)),
        "other_success_n": int(
            other_values.eq(1).sum()
        ),
        "other_success_rate": (
            float(other_values.mean())
            if len(other_values) else None
        ),
        "weekly_lift_mean_pp": (
            float(valid.mean())
            if len(valid) else None
        ),
        "weeks_compared": int(len(valid)),
        "complete_weeks": int(complete.sum()),
        "incomplete_weeks": int((~complete).sum()),
        "positive_weeks": int(valid.gt(0).sum()),
        "negative_weeks": int(valid.lt(0).sum()),
        "zero_weeks": int(valid.eq(0).sum()),
    }


def main():
    print("V1.3F-11 Stage F4")
    print("Out-of-Sample Confirmation")
    print("-----------------------------------", flush=True)

    stock, ihsg, exclusions = read_sources()

    require(
        exclusions == EXPECTED_EXCLUSIONS,
        "Calendar exclusion count differs",
    )

    base = prepare(
        stock, ihsg, "baseline"
    )

    scenario = prepare(
        stock.loc[~stock["exclude"]].copy(),
        ihsg,
        "scenario",
    )

    dates = select_oos_dates(base, scenario)

    print(
        "OOS weeks:", len(dates),
        "| First:", dates.iloc[0].date(),
        "| Last:", dates.iloc[-1].date(),
        flush=True,
    )

    records = []

    for date in dates:
        print(
            "Evaluating:", date.date(),
            flush=True,
        )

        result = score_decision(
            base, scenario, ihsg, date
        )

        records.append(result)

    detail = pd.concat(
        records, ignore_index=True
    )

    require(
        not detail.duplicated(
            ["decision_date", "ticker"]
        ).any(),
        "Duplicate OOS prediction keys",
    )

    detail = attach_historical_labels(
        detail, base, "baseline"
    )

    detail = detail.rename(
        columns={
            f"y{target}": f"baseline_y{target}"
            for target in TARGETS
        }
    )

    detail = attach_historical_labels(
        detail, scenario, "scenario"
    )

    detail = detail.rename(
        columns={
            f"y{target}": f"scenario_y{target}"
            for target in TARGETS
        }
    )

    weekly = evaluate_weekly(detail)

    metrics = {
        "baseline": {},
        "scenario": {},
    }

    for mode in ["baseline", "scenario"]:
        for target in TARGETS:
            result = summarize(
                weekly, detail, mode, target
            )

            metrics[mode][
                f"target_{target}"
            ] = result

            lift = result["weekly_lift_mean_pp"]

            lift_text = (
                f"{lift:+.2f}pp"
                if lift is not None
                else "UNKNOWN"
            )

            print(
                mode.upper(),
                f"TARGET {target}%",
                f"Top3={result['top3_success_n']}/"
                f"{result['top3_known_n']}",
                f"Other={result['other_success_n']}/"
                f"{result['other_known_n']}",
                f"WeeklyLift={lift_text}",
                flush=True,
            )

    primary = metrics[
        "baseline"
    ][f"target_{PRIMARY_TARGET}"]

    require(
        primary["weeks_compared"] > 0,
        "No evaluable primary-target weeks",
    )

    summary = {
        "version": "V1.3F-11",
        "stage": "F4_OOS_CONFIRMATION",
        "status": "RESEARCH_DIAGNOSTIC_ONLY",
        "primary_hypothesis": (
            "Baseline Top-3 mean weekly predictive "
            "lift is positive for the 8% label"
        ),
        "primary_target": PRIMARY_TARGET,
        "secondary_targets": [3, 5],
        "oos_start": str(OOS_START.date()),
        "first_decision_date": str(
            dates.iloc[0].date()
        ),
        "last_decision_date": str(
            dates.iloc[-1].date()
        ),
        "weekly_decisions": int(len(dates)),
        "candidate_observations": int(len(detail)),
        "model_fits": int(
            len(dates) * len(TARGETS) * 2
        ),
        "hypothetical_exclusions": int(exclusions),
        "metrics": metrics,
        "primary_direction_positive": bool(
            primary["weekly_lift_mean_pp"] > 0
        ),
        "limitations": [
            "Confirmation uses a later temporal period.",
            "Independence from all earlier research is not proven.",
            "Point-in-time corporate actions are not audited.",
            "Weekly horizons may overlap.",
            "Unknown labels are excluded from rate denominators.",
            "Positive lift does not establish statistical significance.",
            "No bootstrap confidence interval in this stage.",
            "Historical label success is not trading profit.",
            "No transaction costs, slippage or execution model.",
            "Calendar exclusions are hypothetical.",
            "No production trading readiness established.",
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
        DETAIL,
        index=False,
        date_format="%Y-%m-%d",
    )

    weekly.to_csv(
        WEEKLY,
        index=False,
        date_format="%Y-%m-%d",
    )

    SUMMARY.write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("WEEKS:", len(dates))
    print("CANDIDATES:", len(detail))
    print(
        "MODEL_FITS:",
        len(dates) * len(TARGETS) * 2,
    )
    print(
        "PRIMARY_8PCT_LIFT_POSITIVE:",
        summary["primary_direction_positive"],
    )
    print("STATUS: RESEARCH_DIAGNOSTIC_COMPLETED")


if __name__ == "__main__":
    main()
