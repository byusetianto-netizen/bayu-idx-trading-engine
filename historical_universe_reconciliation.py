
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

DATA = Path("data")
SECURITY = DATA / "security_master.csv"
V13C2_MASTER = DATA / "historical_security_master_v13c2.csv"
EVENTS = DATA / "historical_event_evidence.csv"
PRICE = DATA / "idx_stock_prices_expansion.csv"

OUT_RECON = DATA / "historical_universe_reconciliation.csv"
OUT_QUEUE = DATA / "historical_universe_review_queue.csv"
OUT_SUMMARY = DATA / "historical_universe_reconciliation_summary.json"
OUT_STATUS = DATA / "historical_universe_reconciliation_status.json"


def clean_ticker(x):
    if pd.isna(x):
        return pd.NA
    return str(x).upper().strip().replace(".JK", "")


def read_csv(path, required=False):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, low_memory=False)


def main():
    # Price universe is used only to define the set of tickers requiring reconciliation.
    prices = read_csv(PRICE, required=True)
    if "ticker" not in prices.columns:
        raise ValueError("idx_stock_prices_expansion.csv must contain ticker")
    price_tickers = sorted(set(prices["ticker"].map(clean_ticker).dropna()))

    sec = read_csv(SECURITY)
    if not sec.empty and "ticker" in sec.columns:
        sec["ticker"] = sec["ticker"].map(clean_ticker)
        sec = sec.drop_duplicates("ticker")
    else:
        sec = pd.DataFrame({"ticker": price_tickers})

    base = pd.DataFrame({"ticker": price_tickers})
    base = base.merge(sec, on="ticker", how="left", suffixes=("", "_sec"))

    # V1.3C-2 event evidence.
    events = read_csv(EVENTS)
    if not events.empty:
        for c in ["ticker", "old_ticker", "new_ticker"]:
            if c in events.columns:
                events[c] = events[c].map(clean_ticker)
        events["event_date"] = pd.to_datetime(events.get("event_date"), errors="coerce")
        events["end_date"] = pd.to_datetime(events.get("end_date"), errors="coerce")
    else:
        events = pd.DataFrame(columns=[
            "event_type","ticker","event_date","end_date",
            "old_ticker","new_ticker","confidence","source"
        ])

    def event_count(t):
        return int((events["ticker"] == t).sum()) if len(events) else 0

    def first_event(t, typ):
        if not len(events):
            return pd.NaT
        x = events[(events["ticker"] == t) & (events["event_type"] == typ)]
        return x["event_date"].min() if len(x) else pd.NaT

    def evidence_conf(t, typ):
        if not len(events):
            return "UNKNOWN"
        x = events[(events["ticker"] == t) & (events["event_type"] == typ)]
        if x.empty:
            return "UNKNOWN"
        vals = x["confidence"].fillna("UNVERIFIED").str.upper()
        if vals.isin(["VERIFIED", "HIGH", "SOURCE_EVIDENCE"]).any():
            return "VERIFIED"
        if vals.isin(["INFERRED", "MEDIUM"]).any():
            return "INFERRED"
        return "UNVERIFIED"

    rows = []
    for t in base["ticker"]:
        b = base[base["ticker"] == t].iloc[0]
        listing = first_event(t, "listing")
        delisting = first_event(t, "delisting")
        susp = events[(events["ticker"] == t) & (events["event_type"] == "suspension")] if len(events) else pd.DataFrame()
        ticker_changes = events[(events["ticker"] == t) | (events.get("old_ticker", pd.Series(dtype=str)) == t) | (events.get("new_ticker", pd.Series(dtype=str)) == t)] if len(events) else pd.DataFrame()

        sourced_listing = b.get("listed_date", pd.NA)
        sourced_delisting = b.get("delisted_date", pd.NA)

        listing_source_status = "VERIFIED" if pd.notna(sourced_listing) else evidence_conf(t, "listing")
        delisting_source_status = "VERIFIED" if pd.notna(sourced_delisting) else evidence_conf(t, "delisting")

        # First observed date is explicitly treated as an observation, never proof.
        first_obs = b.get("first_observed_price_date", pd.NA)
        last_obs = b.get("last_observed_price_date", pd.NA)

        if listing_source_status == "VERIFIED":
            membership_start = sourced_listing
            start_basis = "VERIFIED_LISTING_EVIDENCE"
        elif listing_source_status == "INFERRED":
            membership_start = listing
            start_basis = "INFERRED_LISTING_EVIDENCE"
        else:
            membership_start = first_obs
            start_basis = "FIRST_PRICE_OBSERVATION_ONLY__NOT_LISTING_DATE"

        if delisting_source_status == "VERIFIED":
            membership_end = sourced_delisting
            end_basis = "VERIFIED_DELISTING_EVIDENCE"
        elif delisting_source_status == "INFERRED":
            membership_end = delisting
            end_basis = "INFERRED_DELISTING_EVIDENCE"
        else:
            membership_end = last_obs
            end_basis = "OPEN_ENDED_CURRENT_DATASET__NOT_DELISTING"

        if listing_source_status == "VERIFIED" and delisting_source_status in ["VERIFIED", "UNKNOWN"]:
            overall = "VERIFIED_PARTIAL"
        elif listing_source_status in ["INFERRED", "VERIFIED"] or delisting_source_status in ["INFERRED", "VERIFIED"]:
            overall = "PARTIAL_EVIDENCE"
        else:
            overall = "UNKNOWN"

        review_reasons = []
        if listing_source_status != "VERIFIED":
            review_reasons.append("LISTING_EVIDENCE_MISSING")
        if delisting_source_status == "UNKNOWN":
            review_reasons.append("DELISTING_STATUS_UNVERIFIED")
        if len(susp):
            review_reasons.append("SUSPENSION_EVENTS_PRESENT")
        if len(ticker_changes):
            review_reasons.append("TICKER_CHANGE_REVIEW")
        if start_basis.startswith("FIRST_PRICE"):
            review_reasons.append("FIRST_PRICE_IS_NOT_LISTING_DATE")
        if overall != "VERIFIED_PARTIAL":
            review_reasons.append("PIT_MEMBERSHIP_NOT_FULLY_VERIFIED")

        rows.append({
            "ticker": t,
            "first_observed_price_date": first_obs,
            "last_observed_price_date": last_obs,
            "listing_date_evidence": sourced_listing if pd.notna(sourced_listing) else (listing.strftime("%Y-%m-%d") if pd.notna(listing) else pd.NA),
            "listing_status": listing_source_status,
            "listing_basis": start_basis,
            "delisting_date_evidence": sourced_delisting if pd.notna(sourced_delisting) else (delisting.strftime("%Y-%m-%d") if pd.notna(delisting) else pd.NA),
            "delisting_status": delisting_source_status,
            "delisting_basis": end_basis,
            "suspension_event_count": int(len(susp)),
            "ticker_change_event_count": int(len(ticker_changes)),
            "membership_start_candidate": membership_start,
            "membership_end_candidate": membership_end,
            "overall_reconciliation_status": overall,
            "review_priority": "HIGH" if overall == "UNKNOWN" or listing_source_status != "VERIFIED" else ("MEDIUM" if len(susp) or len(ticker_changes) else "LOW"),
            "review_reasons": ";".join(review_reasons) if review_reasons else "NONE",
            "pit_safe_for_backtest": False,
            "notes": "Candidate interval only. Not enabled for engine."
        })

    recon = pd.DataFrame(rows)
    recon = recon.sort_values(["review_priority", "ticker"], ascending=[True, True])
    recon.to_csv(OUT_RECON, index=False)

    queue = recon[recon["review_priority"].isin(["HIGH", "MEDIUM"])].copy()
    queue.to_csv(OUT_QUEUE, index=False)

    summary = {
        "status": "V1.3C3_RECONCILIATION_COMPLETE__NO_ENGINE_CHANGE",
        "price_universe_tickers": int(len(recon)),
        "verified_partial": int((recon["overall_reconciliation_status"] == "VERIFIED_PARTIAL").sum()),
        "partial_evidence": int((recon["overall_reconciliation_status"] == "PARTIAL_EVIDENCE").sum()),
        "unknown": int((recon["overall_reconciliation_status"] == "UNKNOWN").sum()),
        "high_priority_review": int((recon["review_priority"] == "HIGH").sum()),
        "medium_priority_review": int((recon["review_priority"] == "MEDIUM").sum()),
        "low_priority_review": int((recon["review_priority"] == "LOW").sum()),
        "listing_verified": int((recon["listing_status"] == "VERIFIED").sum()),
        "delisting_verified": int((recon["delisting_status"] == "VERIFIED").sum()),
        "suspension_event_tickers": int((recon["suspension_event_count"] > 0).sum()),
        "ticker_change_event_tickers": int((recon["ticker_change_event_count"] > 0).sum()),
        "pit_safe_tickers": 0,
        "warning": "No ticker is marked PIT-safe in this probe. Candidate intervals are audit outputs only.",
        "methodology": "First observed price date is never treated as official listing date. Missing events remain UNKNOWN."
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    OUT_STATUS.write_text(json.dumps({
        "status": summary["status"],
        "main_dataset_modified": False,
        "engine_inputs_modified": False,
        "pit_enabled": False,
        "next_step": "Populate verified event evidence for high-priority review queue, then build PIT security master."
    }, indent=2), encoding="utf-8")

    print("=== V1.3C-3 Evidence Reconciliation ===")
    for k, v in summary.items():
        if k not in ["warning", "methodology"]:
            print(f"{k}: {v}")
    print("STATUS: V1.3C3_RECONCILIATION_COMPLETE__NO_ENGINE_CHANGE")


if __name__ == "__main__":
    main()
