"""
V1.3A — BEI Universe Discovery & Security Master Audit.

Purpose
-------
Build a broader CURRENT listed-company universe using the IDX Company Profiles
page when it can be parsed, cross-check it against a daily all-stock snapshot,
and compare both with the local historical price dataset.

This is deliberately audit-first:
- It NEVER overwrites idx_stock_prices.csv.
- It does NOT infer listing dates from first observed prices.
- It does NOT claim to solve historical point-in-time membership or survivorship bias.
- Failure to read the official IDX page falls back to the daily snapshot and is
  explicitly recorded in universe_status.json.
"""
from pathlib import Path
from datetime import timedelta
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
STOCKS = DATA / "idx_stock_prices.csv"
MASTER = DATA / "security_master.csv"
STATUS = DATA / "universe_status.json"
AUDIT = DATA / "universe_audit.csv"

FALLBACK_BASE = "https://raw.githubusercontent.com/nofendian17/idx_dataset/main/data/stock_data_{date}.csv"
IDX_PROFILE_BASE = "https://www.idx.co.id/en/listed-companies/company-profiles/"
IDX_PROFILES_URL = "https://www.idx.co.id/en/listed-companies/company-profiles/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; IDX-Research-Engine/1.3; research use)"}


def clean_ticker(x):
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s or None


def fetch_daily_snapshot(d):
    url = FALLBACK_BASE.format(date=d.strftime("%Y-%m-%d"))
    try:
        r = requests.get(url, timeout=30, headers=HEADERS)
        if r.status_code != 200 or not r.text.strip():
            return None
        x = pd.read_csv(io.StringIO(r.text))
        x = x.rename(columns={
            "Date": "date", "Stock Code": "ticker", "Open Price": "open",
            "High Price": "high", "Low Price": "low", "Last Price": "close",
            "Volume": "volume", "Value": "value", "Board": "board",
            "Stock Name": "company_name",
        })
        if "ticker" not in x.columns:
            return None
        x["ticker"] = x["ticker"].map(clean_ticker)
        return x.dropna(subset=["ticker"]).copy()
    except Exception:
        return None


def discover_latest_snapshot(lookback_days=14):
    today = pd.Timestamp.now(tz="Asia/Jakarta").date()
    for i in range(lookback_days + 1):
        d = today - timedelta(days=i)
        x = fetch_daily_snapshot(pd.Timestamp(d))
        if x is not None and len(x):
            return d, x
    raise RuntimeError("No daily all-stock snapshot found in the lookback window.")


def parse_idx_profiles_page(html):
    """Best-effort parser for the official IDX Company Profiles listing table."""
    tables = pd.read_html(io.StringIO(html))
    candidates = []
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        joined = " | ".join(cols)
        if "code" in joined and "name" in joined and "listing date" in joined:
            candidates.append(t)
    if not candidates:
        return pd.DataFrame()

    t = max(candidates, key=len).copy()
    rename = {}
    for c in t.columns:
        lc = str(c).strip().lower()
        if lc == "code":
            rename[c] = "ticker"
        elif lc == "name":
            rename[c] = "company_name"
        elif lc == "listing date":
            rename[c] = "listed_date"
    t = t.rename(columns=rename)
    if "ticker" not in t.columns:
        return pd.DataFrame()
    t["ticker"] = t["ticker"].map(clean_ticker)
    if "company_name" in t.columns:
        t["company_name"] = t["company_name"].fillna("").astype(str).str.strip()
    if "listed_date" in t.columns:
        t["listed_date"] = pd.to_datetime(t["listed_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        t["listed_date"] = t["listed_date"].fillna("")
    return t.dropna(subset=["ticker"]).drop_duplicates("ticker")


def fetch_official_current_universe():
    try:
        r = requests.get(IDX_PROFILES_URL, timeout=45, headers=HEADERS)
        r.raise_for_status()
        parsed = parse_idx_profiles_page(r.text)
        if len(parsed):
            return parsed, {"status": "SUCCESS", "rows": int(len(parsed)), "url": IDX_PROFILES_URL}
        return pd.DataFrame(), {"status": "PARSE_EMPTY", "rows": 0, "url": IDX_PROFILES_URL}
    except Exception as e:
        return pd.DataFrame(), {"status": "FETCH_FAILED", "rows": 0, "url": IDX_PROFILES_URL, "error": str(e)[:300]}


def fetch_idx_profile_metadata(ticker):
    """Best-effort profile enrichment. Failure never blocks the audit."""
    url = IDX_PROFILE_BASE + ticker
    try:
        r = requests.get(url, timeout=15, headers=HEADERS)
        if r.status_code != 200:
            return ticker, {}
        text = re.sub(r"\s+", " ", r.text)
        out = {}
        patterns = {
            "listed_date": r"Listing Date.{0,250}?(\d{4}-\d{2}-\d{2})",
            "sector": r"Sector.{0,250}?([A-Za-z][A-Za-z &\-]+)",
            "board": r"Listing Board.{0,250}?([A-Za-z][A-Za-z &\-]+)",
        }
        for key, pat in patterns.items():
            m = re.search(pat, text, re.I)
            if m:
                out[key] = m.group(1).strip()
        if out:
            out["metadata_source"] = "IDX_company_profile"
            out["metadata_confidence"] = "HIGH"
        return ticker, out
    except Exception:
        return ticker, {}


def build_master(snapshot_date, snapshot, official):
    local = pd.read_csv(STOCKS, usecols=["ticker"]) if STOCKS.exists() else pd.DataFrame(columns=["ticker"])
    local_tickers = sorted(set(local["ticker"].map(clean_ticker).dropna()))
    daily_tickers = sorted(set(snapshot["ticker"].dropna()))
    official_tickers = sorted(set(official["ticker"].dropna())) if len(official) else []

    universe = sorted(set(local_tickers) | set(daily_tickers) | set(official_tickers))
    old = pd.read_csv(MASTER) if MASTER.exists() else pd.DataFrame()
    if len(old) and "ticker" in old.columns:
        old["ticker"] = old["ticker"].map(clean_ticker)
        old = old.drop_duplicates("ticker")
    old_by = {r["ticker"]: r for _, r in old.iterrows()} if len(old) else {}

    off_by = {r["ticker"]: r for _, r in official.iterrows()} if len(official) else {}
    snap = snapshot.drop_duplicates("ticker").set_index("ticker")

    rows = []
    for t in universe:
        prev = old_by.get(t, {})
        o = off_by.get(t, {})
        listed = o.get("listed_date", prev.get("listed_date", ""))
        company = o.get("company_name", prev.get("company_name", ""))
        board = o.get("board", "") or prev.get("board", "")
        if not board and t in snap.index and "board" in snap.columns:
            board = snap.loc[t, "board"]
        rows.append({
            "ticker": t,
            "company_name": company if pd.notna(company) else "",
            "listed_date": listed if pd.notna(listed) else "",
            "delisted_date": prev.get("delisted_date", ""),
            "suspension_start": prev.get("suspension_start", ""),
            "suspension_end": prev.get("suspension_end", ""),
            "board": board if pd.notna(board) else "",
            "sector": o.get("sector", prev.get("sector", "")),
            "trade_status": "CURRENT_LISTED" if t in official_tickers else ("OBSERVED_IN_DAILY_SNAPSHOT" if t in daily_tickers else "LOCAL_ONLY"),
            "in_official_current_universe": t in official_tickers,
            "in_daily_snapshot": t in daily_tickers,
            "in_local_price_data": t in local_tickers,
            "effective_from": snapshot_date.strftime("%Y-%m-%d"),
            "effective_to": "",
            "source": "IDX_company_profiles" if t in official_tickers else ("daily_all_stock_snapshot" if t in daily_tickers else "local_price_dataset"),
            "source_date": snapshot_date.strftime("%Y-%m-%d"),
            "confidence": "HIGH" if t in official_tickers else ("MEDIUM" if t in daily_tickers else "LOW"),
        })
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True), local_tickers, daily_tickers, official_tickers


def main():
    snapshot_date, snapshot = discover_latest_snapshot()
    official, official_meta = fetch_official_current_universe()

    # Only enrich a limited number of official rows when the listing table did not
    # expose board/sector. This keeps the workflow polite to IDX and audit-first.
    # The listing table itself remains the authoritative source for ticker/name/date.
    if len(official):
        official["board"] = ""
        official["sector"] = ""

    master, local_tickers, daily_tickers, official_tickers = build_master(snapshot_date, snapshot, official)

    current = set(official_tickers) if official_tickers else set(daily_tickers)
    local = set(local_tickers)
    daily = set(daily_tickers)

    audit_rows = []
    for t in sorted(set(official_tickers) | daily | local):
        audit_rows.append({
            "ticker": t,
            "in_official_current_universe": t in set(official_tickers),
            "in_daily_snapshot": t in daily,
            "in_local_price_data": t in local,
            "status": (
                "OFFICIAL_DAILY_LOCAL" if t in official_tickers and t in daily and t in local else
                "OFFICIAL_MISSING_LOCAL_PRICE" if t in official_tickers and t not in local else
                "OFFICIAL_NOT_IN_DAILY_SNAPSHOT" if t in official_tickers and t not in daily else
                "DAILY_AND_LOCAL_ONLY" if t in daily and t in local else
                "DAILY_MISSING_LOCAL_PRICE" if t in daily else
                "LOCAL_ONLY"
            )
        })
    pd.DataFrame(audit_rows).to_csv(AUDIT, index=False)
    master.to_csv(MASTER, index=False)

    result = {
        "version": "V1.3A",
        "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
        "official_idx_source": IDX_PROFILES_URL,
        "official_idx_fetch": official_meta,
        "official_current_universe_count": len(official_tickers),
        "daily_snapshot_universe_count": len(daily),
        "local_price_ticker_count": len(local),
        "official_and_local": len(set(official_tickers) & local),
        "official_missing_local_price": len(set(official_tickers) - local),
        "official_not_in_daily_snapshot": len(set(official_tickers) - daily),
        "daily_and_local": len(daily & local),
        "local_only": len(local - set(official_tickers) - daily),
        "official_to_local_coverage_pct": round(100 * len(set(official_tickers) & local) / len(official_tickers), 2) if official_tickers else None,
        "status": "AUDIT_READY__DO_NOT_EXPAND_ENGINE_UNTIL_REVIEWED",
        "notes": [
            "IDX Company Profiles is the authoritative current-universe reference when successfully parsed.",
            "Daily all-stock snapshot is a secondary discovery/cross-check source.",
            "Failure to parse IDX is explicit; the workflow never silently labels the snapshot as official.",
            "No listing date is inferred from first local price date.",
            "Historical point-in-time membership, delistings and survivorship bias remain unsolved in V1.3A.",
            "This phase never overwrites the local price dataset."
        ]
    }
    STATUS.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
