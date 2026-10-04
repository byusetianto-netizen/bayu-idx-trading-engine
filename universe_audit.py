"""
V1.3 Universe Audit — safe, read-only validation.

Run:
    python universe_audit.py

It compares the security master against the price dataset and produces
data/universe_audit_summary.json. It never rewrites market-price data.
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MASTER = DATA / "security_master.csv"
STOCKS = DATA / "idx_stock_prices.csv"
OUT = DATA / "universe_audit_summary.json"

def main():
    if not MASTER.exists():
        raise SystemExit("security_master.csv not found. Run universe_update.py first.")
    m = pd.read_csv(MASTER)
    p = pd.read_csv(STOCKS, usecols=["ticker", "date"])
    m["ticker"] = m["ticker"].astype(str).str.upper().str.strip()
    p["ticker"] = p["ticker"].astype(str).str.upper().str.strip()
    p["date"] = pd.to_datetime(p["date"], errors="coerce")

    current = set(m.ticker.dropna())
    local = set(p.ticker.dropna())
    latest = p["date"].max()

    counts = p.groupby("ticker").size()
    eligible_200 = int((counts >= 200).sum())
    summary = {
        "master_tickers": len(current),
        "price_data_tickers": len(local),
        "overlap_tickers": len(current & local),
        "coverage_pct": round(100 * len(current & local) / len(current), 2) if current else 0,
        "price_latest_date": latest.strftime("%Y-%m-%d") if pd.notna(latest) else None,
        "tickers_with_200plus_rows": eligible_200,
        "new_current_tickers_without_price_history": len(current - local),
        "local_only_tickers": len(local - current),
        "status": "RESEARCH_ONLY_UNTIL_PIT_SECURITY_MASTER_IS_COMPLETE"
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
