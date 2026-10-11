
"""
Bayu IDX Trading Engine — V1.3F-11 Stage B
Calendar Exclusion Sensitivity

RESEARCH DIAGNOSTIC ONLY
- Compare original row-based indicators with a hypothetical
  off-IHSG carry-forward exclusion scenario.
- Exclusions are candidates, NOT verified invalid sessions.
- Do not modify prices, actual_engine.py, or production signals.
- No model retraining, AUC, ranking, or execution backtest.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


VERSION = "V1.3F-11"
STAGE = "B_CALENDAR_EXCLUSION_SENSITIVITY"

STOCK_PATH = Path("data/idx_stock_prices.csv")
IHSG_PATH = Path("data/idx_ihsg_index.csv")
STAGE_A_SUMMARY = Path(
    "data/execution_probe/v13f11_calendar_integrity_summary.json"
)
STAGE_A_DETAIL = Path(
    "data/execution_probe/v13f11_calendar_integrity_detail.csv"
)

OUT = Path("data/execution_probe")
SUMMARY_FILE = OUT / "v13f11_calendar_sensitivity_summary.json"
DETAIL_FILE = OUT / "v13f11_calendar_sensitivity_components.csv"

FEATURES = [
    "ret5", "ret10", "ret20", "ret60",
    "ma20_dist", "ma50_dist", "ma200_dist",
    "vol_ratio", "rsi", "volatility20", "rs20",
]
LABELS = ["y3", "y5", "y8"]
COMPONENTS = FEATURES + LABELS

RTOL = 1e-9
ATOL = 1e-10


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_files():
    for path in [
        STOCK_PATH, IHSG_PATH, STAGE_A_SUMMARY, STAGE_A_DETAIL
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")


def read_sources():
    require_files()

    audit = json.loads(STAGE_A_SUMMARY.read_text(encoding="utf-8"))

    if audit.get("stage") != "A_CALENDAR_INTEGRITY_DIAGNOSTIC":
        raise RuntimeError("Unexpected Stage A summary")

    sources = audit["sources"]
    if sources["stock_sha256"] != sha256(STOCK_PATH):
        raise RuntimeError("Stock dataset differs from Stage A")
    if sources["ihsg_sha256"] != sha256(IHSG_PATH):
        raise RuntimeError("IHSG dataset differs from Stage A")

    stock = pd.read_csv(STOCK_PATH)
    stock.columns = stock.columns.str.strip().str.lower()

    required = {
        "ticker", "date", "open", "high", "low", "close", "volume"
    }
    if not required.issubset(stock.columns):
        raise RuntimeError("Missing required stock columns")

    stock["date"] = pd.to_datetime(
        stock["date"], errors="coerce"
    ).dt.normalize()
    stock["ticker"] = (
        stock["ticker"].astype("string").str.strip().str.upper()
    )
    if stock["date"].isna().any() or stock["ticker"].isna().any():
        raise RuntimeError("Invalid stock key")

    for col in ["open", "high", "low", "close", "volume"]:
        stock[col] = pd.to_numeric(stock[col], errors="coerce")

    if stock.duplicated(["ticker", "date"]).any():
        raise RuntimeError("Duplicate stock ticker/date")

    stock = stock.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    ihsg = pd.read_csv(IHSG_PATH)
    ihsg.columns = ihsg.columns.str.strip().str.lower()
    date_col = "date" if "date" in ihsg else "price"
    if date_col not in ihsg:
        raise RuntimeError("IHSG date column not found")

    ihsg["date"] = pd.to_datetime(
        ihsg[date_col], errors="coerce"
    ).dt.normalize()

    close_col = "close"
    if close_col not in ihsg:
        raise RuntimeError("IHSG close column not found")

    ihsg["close"] = pd.to_numeric(
        ihsg["close"], errors="coerce"
    )
    ihsg = (
        ihsg[["date", "close"]]
        .dropna()
        .drop_duplicates("date")
        .sort_values("date")
    )

    detail = pd.read_csv(
        STAGE_A_DETAIL,
        usecols=[
            "ticker", "date", "classification",
            "scenario_exclusion_candidate",
        ],
    )
    detail["date"] = pd.to_datetime(
        detail["date"], errors="coerce"
    ).dt.normalize()
    detail["ticker"] = (
        detail["ticker"].astype("string").str.strip().str.upper()
    )

    if detail[["ticker", "date"]].isna().any().any():
        raise RuntimeError("Invalid Stage A detail keys")

    if detail.duplicated(["ticker", "date"]).any():
        raise RuntimeError("Duplicate Stage A detail keys")

    if len(detail) != len(stock):
        raise RuntimeError("Stage A detail row count mismatch")

    flag = (
        detail["scenario_exclusion_candidate"]
        .astype(str).str.lower()
    )
    if not flag.isin(["true", "false"]).all():
        raise RuntimeError("Invalid exclusion flag")

    detail["exclude"] = flag.eq("true")

    if (
        detail.loc[detail["exclude"], "classification"]
        != "OFF_IHSG_CARRY_FORWARD_CANDIDATE"
    ).any():
        raise RuntimeError("Unexpected excluded classification")

    merged = stock.merge(
        detail[["ticker", "date", "exclude"]],
        on=["ticker", "date"],
        how="left",
        validate="one_to_one",
    )

    if merged["exclude"].isna().any():
        raise RuntimeError("Stage A and stock row keys differ")

    exclusions = int(merged["exclude"].sum())
    expected = int(
        audit["statistics"]["scenario_exclusion_candidates"]
    )
    if exclusions != expected:
        raise RuntimeError(
            f"Exclusions {exclusions} != Stage A {expected}"
        )

    return merged, ihsg, exclusions


def calculate_components(frame, ihsg):
    """Research reproduction of actual_engine.py features/labels."""
    d = frame.drop(columns=["exclude"], errors="ignore").copy()
    d = d.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    g = d.groupby("ticker", group_keys=False)

    for n in [5, 10, 20, 60]:
        d[f"ret{n}"] = g["close"].pct_change(
            periods=n, fill_method=None
        )

    for n in [20, 50, 200]:
        moving = g["close"].transform(
            lambda s: s.rolling(n, min_periods=n).mean()
        )
        d[f"ma{n}_dist"] = d["close"] / moving - 1

    vol20 = g["volume"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
    d["vol_ratio"] = d["volume"] / vol20

    d["volatility20"] = (
        g["close"]
        .pct_change(5, fill_method=None)
        .groupby(d["ticker"])
        .transform(
            lambda s: s.rolling(4, min_periods=4).std()
        )
    )

    delta = g["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.groupby(d["ticker"]).transform(
        lambda s: s.rolling(14, min_periods=14).mean()
    )
    avg_loss = loss.groupby(d["ticker"]).transform(
        lambda s: s.rolling(14, min_periods=14).mean()
    )

    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["rsi"] = 100 - 100 / (1 + rs)

    ih = ihsg.set_index("date")["close"]
    ih_ret20 = ih.pct_change(20, fill_method=None)
    d["rs20"] = d["ret20"] - d["date"].map(ih_ret20)

    future_highs = pd.concat(
        [g["high"].shift(-i) for i in range(1, 6)],
        axis=1,
    )
    future_lows = pd.concat(
        [g["low"].shift(-i) for i in range(1, 6)],
        axis=1,
    )

    complete = (
        future_highs.notna().all(axis=1)
        & future_lows.notna().all(axis=1)
        & d["close"].notna()
    )
    high5 = future_highs.max(axis=1, skipna=False)
    low5 = future_lows.min(axis=1, skipna=False)

    for pct in [3, 5, 8]:
        target = pct / 100
        label = (
            (high5 >= d["close"] * (1 + target))
            & (low5 > d["close"] * 0.97)
        ).astype(float)
        d[f"y{pct}"] = label.where(complete, np.nan)

    return d[["ticker", "date"] + COMPONENTS]


def period_of(date):
    if date < pd.Timestamp("2023-01-01"):
        return "TRAIN"
    if date < pd.Timestamp("2025-01-01"):
        return "VALIDATION"
    return "POST_VALIDATION"


def compare(baseline, scenario):
    common = baseline.merge(
        scenario,
        on=["ticker", "date"],
        how="inner",
        validate="one_to_one",
        suffixes=("_base", "_scenario"),
    )

    common["period"] = common["date"].map(period_of)
    results = []
    impacted = np.zeros(len(common), dtype=bool)

    for col in COMPONENTS:
        a = common[f"{col}_base"].to_numpy(dtype=float)
        b = common[f"{col}_scenario"].to_numpy(dtype=float)

        both_available = np.isfinite(a) & np.isfinite(b)
        availability_changed = np.isfinite(a) ^ np.isfinite(b)

        numeric_changed = np.zeros(len(common), dtype=bool)
        numeric_changed[both_available] = ~np.isclose(
            a[both_available],
            b[both_available],
            rtol=RTOL,
            atol=ATOL,
        )

        impacted |= availability_changed | numeric_changed

        for period in ["TRAIN", "VALIDATION", "POST_VALIDATION"]:
            p = common["period"].eq(period).to_numpy()

            results.append({
                "component": col,
                "period": period,
                "common_rows": int(p.sum()),
                "both_available_rows": int(
                    (p & both_available).sum()
                ),
                "numeric_changed_rows": int(
                    (p & numeric_changed).sum()
                ),
                "availability_changed_rows": int(
                    (p & availability_changed).sum()
                ),
                "any_changed_rows": int(
                    (p & (numeric_changed | availability_changed)).sum()
                ),
            })

    common["any_component_changed"] = impacted

    by_period = {}
    for period in ["TRAIN", "VALIDATION", "POST_VALIDATION"]:
        p = common["period"].eq(period)
        by_period[period] = {
            "common_rows": int(p.sum()),
            "any_component_changed_rows": int(
                (p & common["any_component_changed"]).sum()
            ),
        }

    return pd.DataFrame(results), by_period


def main():
    stock, ihsg, exclusion_count = read_sources()

    baseline = calculate_components(stock, ihsg)
    filtered_stock = stock.loc[~stock["exclude"]].copy()
    scenario = calculate_components(filtered_stock, ihsg)

    if len(baseline) != len(stock):
        raise RuntimeError("Baseline row accounting failure")

    if len(scenario) != len(stock) - exclusion_count:
        raise RuntimeError("Scenario row accounting failure")

    component_detail, by_period = compare(
        baseline, scenario
    )

    summary = {
        "version": VERSION,
        "stage": STAGE,
        "status": "RESEARCH_DIAGNOSTIC_ONLY",
        "baseline_rows": int(len(baseline)),
        "scenario_rows": int(len(scenario)),
        "excluded_candidate_rows": int(exclusion_count),
        "common_rows": int(len(scenario)),
        "component_statistics": component_detail.to_dict(
            orient="records"
        ),
        "period_impact": by_period,
        "method": {
            "baseline": "ORIGINAL_ROW_SEQUENCE",
            "scenario": "EXCLUDE_STAGE_A_OFF_IHSG_CANDIDATES",
            "comparison": "SAME_TICKER_AND_DATE_ONLY",
            "numeric_rtol": RTOL,
            "numeric_atol": ATOL,
            "note": (
                "Indicator definitions reproduce actual_engine.py "
                "research logic; model fitting is not performed."
            ),
        },
        "policy": {
            "source_prices_modified": False,
            "actual_engine_modified": False,
            "official_calendar_verified": False,
            "model_retrained": False,
            "auc_measured": False,
            "trading_backtest_performed": False,
        },
        "limitations": [
            "Candidate exclusions are hypothetical, not verified.",
            "IHSG dates are contextual, not an official calendar.",
            "Missing and nonfinite values count as unavailable.",
            "Retained historical labels may shift their future "
            "five-row horizon after candidate exclusion.",
            "The reference engine's data-loading and missing-value "
            "behavior may differ for unusual source values.",
            "No production eligibility or ranking is tested.",
            "No profitability or execution readiness is inferred.",
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)

    component_detail.to_csv(DETAIL_FILE, index=False)
    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n=== V1.3F-11 STAGE B ===")
    print(f"Baseline rows: {len(baseline):,}")
    print(f"Scenario rows: {len(scenario):,}")
    print(f"Excluded candidates: {exclusion_count:,}")

    for period, result in by_period.items():
        print(
            f"{period}: common={result['common_rows']:,}, "
            f"any component changed="
            f"{result['any_component_changed_rows']:,}"
        )

    print("\nFeature/label impact:")
    print(
        component_detail[
            [
                "component", "period",
                "numeric_changed_rows",
                "availability_changed_rows",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved: {SUMMARY_FILE}")
    print(f"Saved: {DETAIL_FILE}")
    print("STATUS: RESEARCH_DIAGNOSTIC_COMPLETED")


if __name__ == "__main__":
    main()
