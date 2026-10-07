
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

DATA = Path("data")
SECURITY = DATA / "security_master.csv"
EVENT_DIR = DATA / "historical_events"
OUT_EVENTS = DATA / "historical_event_evidence.csv"
OUT_MASTER = DATA / "historical_security_master_v13c2.csv"
OUT_SUMMARY = DATA / "historical_event_evidence_summary.json"
OUT_STATUS = DATA / "historical_event_evidence_status.json"

FILES = {
    "listing": EVENT_DIR / "listing_events.csv",
    "suspension": EVENT_DIR / "suspension_events.csv",
    "delisting": EVENT_DIR / "delisting_events.csv",
    "ticker_change": EVENT_DIR / "ticker_change_events.csv",
}

COLUMNS = [
    "event_type","ticker","company_name","event_date","end_date",
    "old_ticker","new_ticker","sector","board","source","source_date",
    "source_url","confidence","notes"
]

def read(path):
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype=str)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[COLUMNS]

def clean_ticker(x):
    if pd.isna(x): return pd.NA
    return str(x).upper().strip().replace(".JK","")

def main():
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for typ, path in FILES.items():
        df = read(path)
        if len(df):
            df["event_type"] = typ
            df["ticker"] = df["ticker"].map(clean_ticker)
            df["old_ticker"] = df["old_ticker"].map(clean_ticker)
            df["new_ticker"] = df["new_ticker"].map(clean_ticker)
            df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            frames.append(df)
    events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    if len(events):
        events = events.drop_duplicates()
        events["confidence"] = events["confidence"].fillna("UNVERIFIED")
    events.to_csv(OUT_EVENTS, index=False)

    # Build a non-destructive evidence-enriched master.
    sec = pd.read_csv(SECURITY, dtype=str) if SECURITY.exists() else pd.DataFrame(columns=["ticker"])
    if "ticker" not in sec.columns:
        sec["ticker"] = pd.Series(dtype=str)
    sec["ticker"] = sec["ticker"].map(clean_ticker)

    def event_date(typ, col="event_date"):
        x = events[events["event_type"] == typ].copy()
        if x.empty: return pd.Series(dtype=str)
        x[col] = pd.to_datetime(x[col], errors="coerce")
        return x.groupby("ticker")[col].min().dt.strftime("%Y-%m-%d")

    listed = event_date("listing")
    delisted = event_date("delisting")
    suspended = events[events["event_type"]=="suspension"].copy()
    suspended["event_date"] = pd.to_datetime(suspended["event_date"], errors="coerce")
    suspended["end_date"] = pd.to_datetime(suspended["end_date"], errors="coerce")

    sec["evidence_listing_date"] = sec["ticker"].map(listed)
    sec["evidence_delisting_date"] = sec["ticker"].map(delisted)
    sec["listing_evidence_confidence"] = sec["evidence_listing_date"].notna().map({True:"SOURCE_EVIDENCE",False:"UNKNOWN"})
    sec["delisting_evidence_confidence"] = sec["evidence_delisting_date"].notna().map({True:"SOURCE_EVIDENCE",False:"UNKNOWN"})

    susp_count = suspended.groupby("ticker").size() if len(suspended) else pd.Series(dtype=int)
    sec["suspension_event_count"] = sec["ticker"].map(susp_count).fillna(0).astype(int)

    # We do not create definitive suspension intervals unless end_date is supplied.
    open_susp = suspended[suspended["end_date"].isna()].groupby("ticker").size() if len(suspended) else pd.Series(dtype=int)
    sec["open_suspension_events"] = sec["ticker"].map(open_susp).fillna(0).astype(int)

    sec.to_csv(OUT_MASTER, index=False)

    summary = {
        "status": "V1.3C2_AUDIT_IMPORT_COMPLETE__NO_ENGINE_CHANGE",
        "event_rows": int(len(events)),
        "listing_events": int((events.event_type=="listing").sum()) if len(events) else 0,
        "delisting_events": int((events.event_type=="delisting").sum()) if len(events) else 0,
        "suspension_events": int((events.event_type=="suspension").sum()) if len(events) else 0,
        "ticker_change_events": int((events.event_type=="ticker_change").sum()) if len(events) else 0,
        "tickers_with_listing_evidence": int(sec["evidence_listing_date"].notna().sum()),
        "tickers_with_delisting_evidence": int(sec["evidence_delisting_date"].notna().sum()),
        "tickers_with_suspension_events": int((sec["suspension_event_count"]>0).sum()),
        "note": "Empty event templates are expected on first run. Main price dataset and trading engine are unchanged."
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    OUT_STATUS.write_text(json.dumps({
        "status": summary["status"],
        "main_dataset_modified": False,
        "engine_inputs_modified": False,
        "source_priority": "IDX official > IDX-derived > secondary cross-check",
        "next_step": "Populate event CSVs from verified IDX evidence, then run PIT reconciliation."
    }, indent=2), encoding="utf-8")

    print("=== V1.3C-2 Historical Event Evidence ===")
    print(f"Event rows: {len(events)}")
    print(f"Listing events: {summary['listing_events']}")
    print(f"Delisting events: {summary['delisting_events']}")
    print(f"Suspension events: {summary['suspension_events']}")
    print(f"Ticker changes: {summary['ticker_change_events']}")
    print("STATUS: V1.3C2_AUDIT_IMPORT_COMPLETE__NO_ENGINE_CHANGE")

if __name__ == "__main__":
    main()
