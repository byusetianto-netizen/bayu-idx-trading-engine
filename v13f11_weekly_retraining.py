
"""
Bayu IDX Trading Engine
V1.3F-11 Stage E2 - Weekly Walk-Forward Retraining

RESEARCH ONLY - FOUR-WEEK PILOT

Baseline versus hypothetical calendar exclusion scenario.
Point-in-time label maturity, target-specific training,
historical liquidity, and common-universe ranking.

No production signals, no trading orders, no price repair.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from v13f11_calendar_sensitivity import (
    FEATURES,
    calculate_components,
    read_sources,
)
from v13f11_weekly_walkforward import (
    build_dataset,
    weekly_decision_dates,
)

OUT = Path("data/execution_probe")
SUMMARY = OUT / "v13f11_weekly_retraining_summary.json"
DETAIL = OUT / "v13f11_weekly_retraining_detail.csv"

TARGETS = [3, 5, 8]
PILOT_WEEKS = 4
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


def prepare(stock, ihsg, name):
    frame = build_dataset(stock, ihsg, name)

    raw = stock.sort_values(
        KEYS
    ).reset_index(drop=True).copy()

    require(
        frame[KEYS].equals(raw[KEYS]),
        f"{name}: raw-feature alignment failure",
    )

    raw["close"] = pd.to_numeric(
        raw["close"], errors="coerce"
    )
    raw["volume"] = pd.to_numeric(
        raw["volume"], errors="coerce"
    )

    raw["turnover"] = raw["close"] * raw["volume"]

    frame["close"] = raw["close"]

    frame["liq20"] = (
        raw.groupby("ticker")["turnover"]
        .transform(
            lambda s: s.rolling(
                20, min_periods=20
            ).median()
        )
    )

    # Historical observation count, including the decision date.
    frame["history_n"] = (
        frame.groupby("ticker").cumcount() + 1
    )

    require(
        not frame.duplicated(KEYS).any(),
        f"{name}: duplicate observation keys",
    )

    return frame


def training_mask(frame, target, decision_date):
    values = frame[FEATURES].to_numpy(dtype=float)

    feature_valid = np.isfinite(values).all(axis=1)

    label = pd.to_numeric(
        frame[f"y{target}"], errors="coerce"
    )
    label_valid = label.notna().to_numpy()

    maturity_valid = (
        frame["label_maturity_date"] < decision_date
    ).to_numpy()

    historical = (
        frame["date"] < decision_date
    ).to_numpy()

    return (
        feature_valid
        & label_valid
        & maturity_valid
        & historical
    )


def fit_target(frame, target, decision_date):
    mask = training_mask(
        frame, target, decision_date
    )

    x = frame.loc[mask, FEATURES]
    y = frame.loc[
        mask, f"y{target}"
    ].astype(int)

    require(
        len(x) > 0,
        f"No training rows for y{target}",
    )

    require(
        y.nunique() == 2,
        f"Single-class training for y{target}",
    )

    require(
        (
            frame.loc[
                mask, "label_maturity_date"
            ] < decision_date
        ).all(),
        "Unmatured label in training",
    )

    model = make_model()
    model.fit(x, y)

    return model, len(x)


def common_candidates(base, scenario, decision_date):
    b = base.loc[
        base["date"].eq(decision_date)
    ].copy()

    s = scenario.loc[
        scenario["date"].eq(decision_date)
    ].copy()

    common = b[KEYS].merge(
        s[KEYS],
        on=KEYS,
        validate="one_to_one",
    )

    b = common.merge(
        b, on=KEYS, validate="one_to_one"
    )

    s = common.merge(
        s, on=KEYS, validate="one_to_one"
    )

    require(
        b[KEYS].equals(s[KEYS]),
        "Candidate alignment failure",
    )

    return b, s


def candidate_mask(frame):
    finite_features = np.isfinite(
        frame[FEATURES].to_numpy(dtype=float)
    ).all(axis=1)

    return (
        finite_features
        & frame["history_n"].ge(200).to_numpy()
        & frame["liq20"].ge(1e9).to_numpy()
        & frame["close"].ge(100).to_numpy()
    )


def calculate_scores(frame):
    d = frame.copy()

    trend = (
        np.clip(
            (d["ma20_dist"] + 0.05) / 0.20,
            0, 1
        ) * 0.35
        + np.clip(
            (d["ma50_dist"] + 0.10) / 0.30,
            0, 1
        ) * 0.35
        + np.clip(
            (d["ret20"] + 0.15) / 0.30,
            0, 1
        ) * 0.30
    )

    momentum = (
        np.clip(
            (d["rsi"] - 35) / 40,
            0, 1
        ) * 0.50
        + np.clip(
            (d["ret10"] + 0.10) / 0.25,
            0, 1
        ) * 0.50
    )

    volume = np.clip(
        d["vol_ratio"] / 2, 0, 1
    )

    relative_strength = np.clip(
        (d["rs20"] + 0.15) / 0.30,
        0, 1
    )

    probability = (
        0.20 * d["p3"]
        + 0.45 * d["p5"]
        + 0.35 * d["p8"]
    )

    liquidity = np.clip(
        np.log10(d["liq20"].clip(lower=1)) / 12,
        0, 1
    )

    d["opportunity_score"] = 100 * (
        0.25 * trend
        + 0.15 * momentum
        + 0.10 * volume
        + 0.10 * relative_strength
        + 0.30 * probability
        + 0.10 * liquidity
    )

    # Engine parity: fixed 5% target and 3% stop.
    d["risk_reward_5"] = (
        (d["close"] * 1.05 - d["close"])
        / (d["close"] - d["close"] * 0.97)
    )

    d["decision"] = np.where(
        (d["opportunity_score"] >= 65)
        & (d["p5"] >= 0.45)
        & (d["risk_reward_5"] >= 1.5),
        "WATCH",
        "NO TRADE",
    )

    return d


def score_week(frame, models):
    result = frame.copy()

    for target in TARGETS:
        result[f"p{target}"] = (
            models[target].predict_proba(
                result[FEATURES]
            )[:, 1]
        )

    result = calculate_scores(result)

    require(
        np.isfinite(
            result["opportunity_score"]
            .to_numpy(dtype=float)
        ).all(),
        "Non-finite opportunity score",
    )

    return result.sort_values(
        ["opportunity_score", "p5", "ticker"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def market_regime(ihsg, decision_date):
    h = ihsg.loc[
        ihsg["date"] <= decision_date,
        ["date", "close"]
    ].sort_values("date").copy()

    require(
        not h["date"].duplicated().any(),
        "Duplicate IHSG dates",
    )

    if h.empty:
        return "UNKNOWN"

    close = pd.to_numeric(
        h["close"], errors="coerce"
    )

    ma50 = close.rolling(
        50, min_periods=50
    ).mean()

    ma200 = close.rolling(
        200, min_periods=200
    ).mean()

    last = close.iloc[-1]
    m50 = ma50.iloc[-1]
    m200 = ma200.iloc[-1]

    if not np.isfinite([last, m50, m200]).all():
        return "UNKNOWN"

    if last > m50 and last > m200:
        return "BULL"

    if last < m50 and last < m200:
        return "BEAR"

    return "NEUTRAL"


def main():
    print("V1.3F-11 Stage E2 - Four-week pilot")

    stock, ihsg, exclusions = read_sources()

    base = prepare(stock, ihsg, "baseline")

    scenario = prepare(
        stock.loc[~stock["exclude"]].copy(),
        ihsg,
        "scenario",
    )

    require(
        len(base) == len(stock),
        "Baseline count mismatch",
    )

    require(
        len(scenario) == len(stock) - exclusions,
        "Scenario count mismatch",
    )

    dates = weekly_decision_dates(
        base, scenario
    ).iloc[:PILOT_WEEKS]

    require(
        len(dates) == PILOT_WEEKS,
        "Insufficient pilot dates",
    )

    records = []
    weekly_metrics = []

    for decision_date in dates:
        print(
            "Evaluating:",
            decision_date.date(),
            flush=True,
        )

        b, s = common_candidates(
            base, scenario, decision_date
        )

        # Identical eligible universe for fair ranking.
        keep = candidate_mask(b) & candidate_mask(s)

        b = b.loc[keep].reset_index(drop=True)
        s = s.loc[keep].reset_index(drop=True)

        require(
            not b.empty,
            "No common eligible candidates",
        )

        require(
            b[KEYS].equals(s[KEYS]),
            "Filtered candidate misalignment",
        )

        b_models = {}
        s_models = {}
        b_train_n = {}
        s_train_n = {}

        for target in TARGETS:
            b_models[target], b_train_n[target] = (
                fit_target(base, target, decision_date)
            )

            s_models[target], s_train_n[target] = (
                fit_target(scenario, target, decision_date)
            )

        b_score = score_week(b, b_models)
        s_score = score_week(s, s_models)

        base_top3 = set(
            b_score.head(3)["ticker"]
        )
        scenario_top3 = set(
            s_score.head(3)["ticker"]
        )

        require(
            len(b_score) >= 3,
            "Fewer than three eligible candidates",
        )

        overlap = len(
            base_top3 & scenario_top3
        )

        regime = market_regime(
            ihsg, decision_date
        )

        weekly_metrics.append(
            {
                "decision_date": str(
                    decision_date.date()
                ),
                "market_regime": regime,
                "common_eligible_n": len(b),
                "top3_overlap_n": overlap,
                "top3_changed": overlap < 3,
                "baseline_train_n": b_train_n,
                "scenario_train_n": s_train_n,
            }
        )

        base_rank = {
            ticker: rank
            for rank, ticker in enumerate(
                b_score["ticker"], start=1
            )
        }

        scenario_rank = {
            ticker: rank
            for rank, ticker in enumerate(
                s_score["ticker"], start=1
            )
        }

        b_scored = b_score.set_index("ticker")
        s_scored = s_score.set_index("ticker")

        for ticker in sorted(
            set(base_rank) & set(scenario_rank)
        ):
            br = b_scored.loc[ticker]
            sr = s_scored.loc[ticker]

            records.append(
                {
                    "decision_date": decision_date,
                    "ticker": ticker,
                    "market_regime": regime,
                    "baseline_rank": base_rank[ticker],
                    "scenario_rank": scenario_rank[ticker],
                    "baseline_score": float(
                        br["opportunity_score"]
                    ),
                    "scenario_score": float(
                        sr["opportunity_score"]
                    ),
                    "baseline_p3": float(br["p3"]),
                    "scenario_p3": float(sr["p3"]),
                    "baseline_p5": float(br["p5"]),
                    "scenario_p5": float(sr["p5"]),
                    "baseline_p8": float(br["p8"]),
                    "scenario_p8": float(sr["p8"]),
                    "baseline_decision": br["decision"],
                    "scenario_decision": sr["decision"],
                    "baseline_top3": base_rank[ticker] <= 3,
                    "scenario_top3": (
                        scenario_rank[ticker] <= 3
                    ),
                }
            )

        print(
            "Eligible candidates:",
            len(b),
            "| Top-3 overlap:",
            overlap,
            "/ 3",
            flush=True,
        )

    detail = pd.DataFrame(records)

    require(
        not detail.empty,
        "No pilot predictions generated",
    )

    summary = {
        "version": "V1.3F-11",
        "stage": "E2_WEEKLY_WALKFORWARD_RETRAINING_PILOT",
        "status": "RESEARCH_DIAGNOSTIC_ONLY",
        "pilot_weeks": int(len(dates)),
        "hypothetical_exclusions": int(exclusions),
        "model_fits": int(
            len(dates) * len(TARGETS) * 2
        ),
        "top3_changed_weeks": int(
            sum(
                x["top3_changed"]
                for x in weekly_metrics
            )
        ),
        "weekly_results": weekly_metrics,
        "training_policy": (
            "Expanding historical training with "
            "target-specific mature labels"
        ),
        "ranking_policy": (
            "Actual engine opportunity_score on "
            "identical eligible candidate universe"
        ),
        "limitations": [
            "Four-week pilot only.",
            "Calendar exclusions are hypothetical.",
            "Weekly retraining is computationally costly.",
            "Market regime is diagnostic, not a score input.",
            "Predictions are made at decision-date close.",
            "No realized returns or execution backtest.",
            "Corporate-action point-in-time availability "
            "has not been independently verified.",
            "No calibrated probability guarantee.",
            "No production trading readiness.",
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

    SUMMARY.write_text(
        json.dumps(
            summary, indent=2, allow_nan=False
        ),
        encoding="utf-8",
    )

    print()
    print("Pilot weeks:", len(dates))
    print(
        "Top-3 changed:",
        summary["top3_changed_weeks"],
    )
    print("Model fits:", summary["model_fits"])
    print("STATUS: RESEARCH_DIAGNOSTIC_COMPLETED")


if __name__ == "__main__":
    main()
