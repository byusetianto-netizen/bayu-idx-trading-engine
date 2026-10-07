
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

DATA = Path("data")
RECON = DATA / "historical_universe_reconciliation.csv"
EVENTS = DATA / "historical_event_evidence.csv"
PRICE = DATA / "idx_stock_prices_expansion.csv"

OUT_MASTER = DATA / "pit_security_master_v13c4.csv"
OUT_DAILY = DATA / "pit_daily_eligibility_sample.csv"
OUT_REVIEW = DATA / "pit_security_master_review_queue.csv"
OUT_SUMMARY = DATA / "pit_security_master_summary.json"
OUT_STATUS = DATA / "pit_security_master_status.json"


def clean(x):
    if pd.isna(x):
        return pd.NA
    return str(x).upper().strip().replace(".JK", "")


def main():
    if not RECON.exists():
        raise FileNotFoundError("Run V1.3C-3 first: historical_universe_reconciliation.csv missing.")
    recon = pd.read_csv(RECON, dtype=str)
    recon["ticker"] = recon["ticker"].map(clean)

    events = pd.read_csv(EVENTS, dtype=str) if EVENTS.exists() else pd.DataFrame()
    if not events.empty:
        for c in ["ticker", "old_ticker", "new_ticker"]:
            if c in events.columns:
                events[c] = events[c].map(clean)
        events["event_date"] = pd.to_datetime(events.get("event_date"), errors="coerce")
        events["end_date"] = pd.to_datetime(events.get("end_date"), errors="coerce")

    # PIT policy:
    # STRICT mode requires verified listing evidence when a historical start is needed.
    # Open-ended current membership is not treated as a verified delisting date.
    # Suspension requires an explicit event interval; price gaps alone never create one.
    out = recon.copy()

    out["pit_start"] = pd.to_datetime(out["listing_date_evidence"], errors="coerce")
    out["pit_end"] = pd.to_datetime(out["delisting_date_evidence"], errors="coerce")

    # A verified listing event is the only source that can make a start date PIT-safe.
    out["pit_start_confidence"] = out["listing_status"].fillna("UNKNOWN")
    out["pit_end_confidence"] = out["delisting_status"].fillna("UNKNOWN")

    out["strict_membership_status"] = "UNKNOWN"

    verified_start = out["listing_status"].eq("VERIFIED")
    no_delisting_or_verified_end = out["delisting_status"].isin(["UNKNOWN", "VERIFIED"])

    out.loc[verified_start & no_delisting_or_verified_end, "strict_membership_status"] = "PARTIAL_VERIFIED"

    # If verified delisting exists, interval is closed and can be used after reconciliation.
    out.loc[
        verified_start & out["delisting_status"].eq("VERIFIED"),
        "strict_membership_status"
    ] = "VERIFIED_INTERVAL"

    # Suspension evidence makes the ticker require interval-aware filtering.
    susp_counts = {}
    if not events.empty and "event_type" in events.columns:
        susp = events[events["event_type"].eq("suspension")]
        if not susp.empty:
            susp_counts = susp.groupby("ticker").size().to_dict()

    out["suspension_events"] = out["ticker"].map(susp_counts).fillna(0).astype(int)
    out["suspension_filter_required"] = out["suspension_events"] > 0

    # Strict PIT eligibility cannot be claimed yet because the event ledger is empty/unverified.
    out["pit_safe_for_backtest"] = out["strict_membership_status"].eq("VERIFIED_INTERVAL")
    out["research_proxy_allowed"] = out["strict_membership_status"].isin(
        ["PARTIAL_VERIFIED", "VERIFIED_INTERVAL"]
    )

    out["eligibility_rule"] = (
        "STRICT: date >= pit_start AND date <= pit_end(if verified) "
        "AND date not inside explicit suspension interval"
    )
    out["proxy_rule"] = (
        "RESEARCH_PROXY: first observed price may define a LOW-confidence start only; "
        "never label it official listing"
    )

    keep = [
        "ticker",
        "first_observed_price_date",
        "last_observed_price_date",
        "listing_date_evidence",
        "listing_status",
        "delisting_date_evidence",
        "delisting_status",
        "suspension_events",
        "suspension_filter_required",
        "pit_start",
        "pit_end",
        "pit_start_confidence",
        "pit_end_confidence",
        "strict_membership_status",
        "pit_safe_for_backtest",
        "research_proxy_allowed",
        "eligibility_rule",
        "proxy_rule",
        "review_priority",
        "review_reasons",
        "notes",
    ]
    for c in keep:
        if c not in out.columns:
            out[c] = pd.NA
    out[keep].to_csv(OUT_MASTER, index=False)

    review = out[
        (~out["pit_safe_for_backtest"]) |
        (out["suspension_filter_required"]) |
        (out["review_priority"].isin(["HIGH", "MEDIUM"]))
    ].copy()
    review[keep].to_csv(OUT_REVIEW, index=False)

    # Optional small sample of date-level logic using latest ~60 unique market dates.
    daily_sample = pd.DataFrame()
    if PRICE.exists():
        p = pd.read_csv(PRICE, usecols=["ticker", "date"], dtype=str)
        p["ticker"] = p["ticker"].map(clean)
        p["date"] = pd.to_datetime(p["date"], errors="coerce")
        p = p.dropna(subset=["ticker", "date"]).drop_duplicates()
        latest_dates = sorted(p["date"].unique())[-60:]
        d = p[p["date"].isin(latest_dates)].merge(
            out[["ticker", "pit_start", "pit_end", "pit_safe_for_backtest"]],
            on="ticker", how="left"
        )
        d["date_in_candidate_interval"] = (
            d["pit_start"].notna() &
            (d["date"] >= d["pit_start"]) &
            (d["pit_end"].isna() | (d["date"] <= d["pit_end"]))
        )
        d["strict_eligible"] = d["date_in_candidate_interval"] & d["pit_safe_for_backtest"].fillna(False)
        daily_sample = d
    daily_sample.to_csv(OUT_DAILY, index=False)

    summary = {
        "status": "V1.3C4_PIT_MASTER_BUILT__NOT_ENABLED_IN_ENGINE",
        "ticker_count": int(len(out)),
        "verified_interval_tickers": int((out["strict_membership_status"] == "VERIFIED_INTERVAL").sum()),
        "partial_verified_tickers": int((out["strict_membership_status"] == "PARTIAL_VERIFIED").sum()),
        "unknown_tickers": int((out["strict_membership_status"] == "UNKNOWN").sum()),
        "pit_safe_tickers": int(out["pit_safe_for_backtest"].sum()),
        "research_proxy_tickers": int(out["research_proxy_allowed"].sum()),
        "suspension_filter_tickers": int(out["suspension_filter_required"].sum()),
        "engine_changed": False,
        "warning": (
            "PIT master is an audit layer only. It is not connected to actual_engine.py. "
            "No ticker is PIT-safe unless listing evidence is VERIFIED and a closed interval is VERIFIED."
        ),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    OUT_STATUS.write_text(json.dumps({
        "status": summary["status"],
        "main_dataset_modified": False,
        "engine_inputs_modified": False,
        "engine_enabled": False,
        "strict_mode": True,
        "next_step": "Populate verified event evidence, then validate PIT eligibility before V1.3D backtest."
    }, indent=2), encoding="utf-8")

    print("=== V1.3C-4 PIT Security Master ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("STATUS: V1.3C4_PIT_MASTER_BUILT__NOT_ENABLED_IN_ENGINE")


if __name__ == "__main__":
    main()
