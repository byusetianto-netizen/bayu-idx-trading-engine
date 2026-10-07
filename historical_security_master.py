
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

DATA = Path("data")
PRICE_FILES = [
    DATA / "idx_stock_prices_expansion.csv",
    DATA / "idx_stock_prices.csv",
]
SECURITY = DATA / "security_master.csv"

OUT_MASTER = DATA / "historical_security_master.csv"
OUT_GAPS = DATA / "historical_universe_gap_flags.csv"
OUT_SUMMARY = DATA / "historical_universe_summary.json"
OUT_STATUS = DATA / "historical_universe_status.json"


def read_csv_flexible(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, low_memory=False, encoding="latin1")


def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in x.columns
    ]
    return x


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def clean_ticker(s):
    if pd.isna(s):
        return None
    return str(s).upper().strip().replace(".JK", "")


def main():
    DATA.mkdir(exist_ok=True)

    # Prefer the expanded research dataset. Fall back to the original panel.
    price_path = next((p for p in PRICE_FILES if p.exists()), None)
    if price_path is None:
        raise FileNotFoundError("No stock price dataset found.")

    prices = norm_cols(read_csv_flexible(price_path))
    tcol = find_col(prices, ["ticker", "code", "stock_code"])
    dcol = find_col(prices, ["date", "datetime", "price_date"])
    if not tcol or not dcol:
        raise ValueError("Price data must contain ticker and date columns.")

    prices["ticker_norm"] = prices[tcol].map(clean_ticker)
    prices["date_norm"] = pd.to_datetime(prices[dcol], errors="coerce")
    prices = prices.dropna(subset=["ticker_norm", "date_norm"])
    prices = prices.drop_duplicates(["ticker_norm", "date_norm"])

    # Optional current security master from V1.3A.
    sec = norm_cols(read_csv_flexible(SECURITY))
    if not sec.empty:
        stcol = find_col(sec, ["ticker", "code", "stock_code"])
        if stcol:
            sec["ticker_norm"] = sec[stcol].map(clean_ticker)
        else:
            sec["ticker_norm"] = None
    else:
        sec = pd.DataFrame(columns=["ticker_norm"])

    # Build one observation profile per ticker.
    rows = []
    for ticker, g in prices.groupby("ticker_norm", sort=True):
        dates = pd.DatetimeIndex(sorted(g["date_norm"].unique()))
        first_obs = dates.min()
        last_obs = dates.max()

        if len(dates) > 1:
            gaps = pd.Series(dates[1:]).diff().dt.days
            max_gap = int(gaps.max())
            gap_count_45 = int((gaps > 45).sum())
            gap_count_90 = int((gaps > 90).sum())
        else:
            max_gap = 0
            gap_count_45 = 0
            gap_count_90 = 0

        # IMPORTANT:
        # First observed price date is NOT automatically a listing date.
        rows.append({
            "ticker": ticker,
            "first_observed_price_date": first_obs.date().isoformat(),
            "last_observed_price_date": last_obs.date().isoformat(),
            "observed_rows": int(len(g)),
            "max_calendar_gap_days": max_gap,
            "gaps_over_45d": gap_count_45,
            "gaps_over_90d": gap_count_90,
            "current_price_data_present": True,
            "membership_basis": "price_observation",
            "membership_confidence": "LOW",
            "listing_date_status": "UNKNOWN_UNLESS_SOURCED",
            "delisting_date_status": "UNKNOWN",
            "pit_status": "NOT_SOLVED",
        })

    master = pd.DataFrame(rows)

    # Enrich from V1.3A security_master where fields exist.
    if not sec.empty and "ticker_norm" in sec.columns:
        sec_small = sec.drop_duplicates("ticker_norm")
        for src_name, candidates in [
            ("company_name", ["company_name", "name", "company"]),
            ("listed_date", ["listed_date", "listing_date"]),
            ("delisted_date", ["delisted_date", "delisting_date"]),
            ("board", ["board", "listing_board"]),
            ("sector", ["sector", "primary_sector"]),
            ("subsector", ["subsector", "sub_sector"]),
            ("trade_status", ["trade_status", "status"]),
            ("source", ["source"]),
            ("source_date", ["source_date"]),
            ("confidence", ["confidence"]),
        ]:
            c = find_col(sec_small, candidates)
            if c:
                master = master.merge(
                    sec_small[["ticker_norm", c]].rename(
                        columns={"ticker_norm": "ticker", c: f"_src_{src_name}"}
                    ),
                    on="ticker",
                    how="left",
                )

    # Prefer sourced listing date, but never silently convert first observed date
    # into an official listing date.
    master["listed_date"] = master.get("_src_listed_date", pd.Series(index=master.index))
    master["listed_date_source"] = master["listed_date"].notna().map(
        {True: "security_master", False: "unknown"}
    )
    master["listed_date_confidence"] = master["listed_date"].notna().map(
        {True: "INHERITED", False: "LOW"}
    )

    master["delisted_date"] = master.get("_src_delisted_date", pd.Series(index=master.index))
    master["delisted_date_source"] = master["delisted_date"].notna().map(
        {True: "security_master", False: "unknown"}
    )

    # Current dataset is a current-universe snapshot. Therefore active_to is
    # deliberately NOT treated as a delisting date.
    latest_data_date = master["last_observed_price_date"].max()
    master["active_from"] = master["listed_date"].fillna(master["first_observed_price_date"])
    master["active_to"] = master["delisted_date"].fillna(latest_data_date)
    master["active_to_status"] = master["delisted_date"].notna().map(
        {True: "DELISITING_DATE_SOURCED", False: "OPEN_ENDED_CURRENT_SNAPSHOT"}
    )

    # A gap is a REVIEW FLAG, not proof of suspension.
    master["possible_suspension_flag"] = master["gaps_over_45d"] > 0
    master["possible_long_gap_flag"] = master["gaps_over_90d"] > 0

    master["pit_eligibility_rule"] = (
        "ELIGIBLE_ONLY_IF_DATE>=active_from AND "
        "DATE<=active_to AND NOT_IN_KNOWN_SUSPENSION_INTERVAL"
    )

    keep = [
        "ticker",
        "listed_date",
        "listed_date_source",
        "listed_date_confidence",
        "first_observed_price_date",
        "last_observed_price_date",
        "active_from",
        "active_to",
        "active_to_status",
        "delisted_date",
        "delisted_date_source",
        "company_name",
        "board",
        "sector",
        "subsector",
        "trade_status",
        "observed_rows",
        "max_calendar_gap_days",
        "gaps_over_45d",
        "gaps_over_90d",
        "possible_suspension_flag",
        "possible_long_gap_flag",
        "membership_basis",
        "membership_confidence",
        "pit_status",
        "pit_eligibility_rule",
    ]
    for c in keep:
        if c not in master.columns:
            master[c] = pd.NA
    master = master[keep].sort_values("ticker")
    master.to_csv(OUT_MASTER, index=False)

    # Gap review table.
    gaps = master[
        (master["possible_suspension_flag"]) | (master["possible_long_gap_flag"])
    ].copy()
    gaps.to_csv(OUT_GAPS, index=False)

    summary = {
        "status": "V1.3C_PROBE_COMPLETE__NO_MAIN_ENGINE_CHANGE",
        "price_source": str(price_path),
        "ticker_count": int(len(master)),
        "tickers_with_sourced_listing_date": int(master["listed_date"].notna().sum()),
        "tickers_with_unknown_listing_date": int(master["listed_date"].isna().sum()),
        "tickers_with_sourced_delisted_date": int(master["delisted_date"].notna().sum()),
        "possible_suspension_or_long_gap_flags": int(master["possible_suspension_flag"].sum()),
        "possible_long_gap_over_90d": int(master["possible_long_gap_flag"].sum()),
        "latest_price_date": latest_data_date,
        "pit_security_master_solved": False,
        "survivorship_bias_solved": False,
        "warning": (
            "First observed price date is not treated as official listing date. "
            "Current-universe membership is not sufficient to reconstruct historical PIT membership. "
            "Gap flags are review signals, not confirmed suspensions."
        ),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    OUT_STATUS.write_text(
        json.dumps(
            {
                "status": summary["status"],
                "main_dataset_modified": False,
                "engine_inputs_modified": False,
                "next_step": "Source historical listing/delisting/suspension records before enabling PIT backtests.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== V1.3C Historical Universe Probe ===")
    print(f"Price source: {price_path}")
    print(f"Tickers: {len(master)}")
    print(f"Sourced listing dates: {master['listed_date'].notna().sum()}")
    print(f"Unknown listing dates: {master['listed_date'].isna().sum()}")
    print(f"Sourced delisted dates: {master['delisted_date'].notna().sum()}")
    print(f"Possible suspension/long-gap flags: {master['possible_suspension_flag'].sum()}")
    print(f"Latest price date: {latest_data_date}")
    print("STATUS: V1.3C_PROBE_COMPLETE__NO_MAIN_ENGINE_CHANGE")


if __name__ == "__main__":
    main()
