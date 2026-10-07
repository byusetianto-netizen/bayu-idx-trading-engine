#!/usr/bin/env python3
"""
V1.3D Research Universe Selector + Liquidity Gate

Purpose:
- Build a transparent research universe from the expanded price dataset.
- Apply minimum data-history, recency, liquidity and volatility gates.
- Combine with V1.3C-5 classifications without making classifications a BUY signal.
- Do NOT modify actual_engine.py or the main price dataset.

Outputs:
  data/research_universe.csv
  data/research_universe_review_queue.csv
  data/research_universe_summary.json
  data/research_universe_status.json
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

PRICE = DATA / "idx_stock_prices_expansion.csv"
CLASS = DATA / "universe_classification.csv"

MIN_HISTORY = 200
MAX_LAG_DAYS = 10
MIN_MEDIAN_VALUE_20D = 1_000_000_000       # Rp1bn/day
MIN_MEDIAN_VALUE_60D = 1_000_000_000       # Rp1bn/day
MAX_ANNUAL_VOL = 2.50                       # research gate only

if not PRICE.exists():
    raise SystemExit("Missing data/idx_stock_prices_expansion.csv")
if not CLASS.exists():
    raise SystemExit("Missing data/universe_classification.csv; run V1.3C-5 first.")

p = pd.read_csv(PRICE)
p.columns = [str(c).strip().lower() for c in p.columns]
required = {"ticker", "date", "close", "volume"}
missing = required - set(p.columns)
if missing:
    raise SystemExit(f"Missing price columns: {sorted(missing)}")

p["ticker"] = p["ticker"].astype(str).str.upper().str.replace(".JK", "", regex=False)
p["date"] = pd.to_datetime(p["date"], errors="coerce")
for c in ["close", "volume"]:
    p[c] = pd.to_numeric(p[c], errors="coerce")
p = p.dropna(subset=["ticker", "date", "close", "volume"])
p = p[p["close"] > 0]
p["trading_value"] = p["close"] * p["volume"]
p = p.sort_values(["ticker", "date"])

latest_market_date = p["date"].max()
rows = []

for ticker, g in p.groupby("ticker", sort=True):
    g = g.drop_duplicates(["ticker", "date"]).sort_values("date")
    ret = g["close"].pct_change()

    last_date = g["date"].max()
    lag_days = int((latest_market_date - last_date).days)
    history_rows = int(len(g))

    value20 = float(g["trading_value"].tail(20).median()) if len(g) >= 20 else np.nan
    value60 = float(g["trading_value"].tail(60).median()) if len(g) >= 60 else np.nan

    r20 = ret.tail(20).dropna()
    ann_vol = float(r20.std() * np.sqrt(252)) if len(r20) >= 10 else np.nan

    history_ok = history_rows >= MIN_HISTORY
    recency_ok = lag_days <= MAX_LAG_DAYS
    liquidity20_ok = pd.notna(value20) and value20 >= MIN_MEDIAN_VALUE_20D
    liquidity60_ok = pd.notna(value60) and value60 >= MIN_MEDIAN_VALUE_60D
    volatility_ok = pd.notna(ann_vol) and ann_vol <= MAX_ANNUAL_VOL

    # Core research universe: data integrity + recency + minimum liquidity.
    # Volatility is NOT a hard exclusion because high-volatility opportunities
    # can still be researched; it is exposed as a risk flag.
    eligible = history_ok and recency_ok and liquidity20_ok and liquidity60_ok

    if eligible and pd.notna(ann_vol) and ann_vol <= 0.45:
        liquidity_bucket = "LIQUID_CORE"
    elif eligible and pd.notna(ann_vol) and ann_vol <= 0.75:
        liquidity_bucket = "LIQUID_GROWTH_MOMENTUM"
    elif eligible:
        liquidity_bucket = "LIQUID_HIGH_VOL"
    else:
        liquidity_bucket = "RESEARCH_ONLY"

    reasons = []
    if not history_ok:
        reasons.append("history<200")
    if not recency_ok:
        reasons.append("stale_data")
    if not liquidity20_ok:
        reasons.append("median_value_20d<1bn")
    if not liquidity60_ok:
        reasons.append("median_value_60d<1bn")
    if not volatility_ok and pd.notna(ann_vol):
        reasons.append("volatility>250%")

    rows.append({
        "ticker": ticker,
        "latest_date": last_date.date().isoformat(),
        "lag_days": lag_days,
        "history_rows": history_rows,
        "median_value_20d": value20,
        "median_value_60d": value60,
        "annualized_volatility_20d": ann_vol,
        "history_gate": history_ok,
        "recency_gate": recency_ok,
        "liquidity_20d_gate": liquidity20_ok,
        "liquidity_60d_gate": liquidity60_ok,
        "volatility_flag": not volatility_ok if pd.notna(ann_vol) else True,
        "research_universe_eligible": eligible,
        "liquidity_bucket": liquidity_bucket,
        "gate_failures": ";".join(reasons) if reasons else "",
    })

q = pd.DataFrame(rows)

c = pd.read_csv(CLASS)
c.columns = [str(x).strip().lower() for x in c.columns]
if "ticker" in c.columns:
    c["ticker"] = c["ticker"].astype(str).str.upper().str.replace(".JK", "", regex=False)

keep = [x for x in [
    "ticker", "primary_class", "secondary_class",
    "index_membership_current_snapshot",
    "conglomerate_group_current_snapshot",
    "classification_confidence",
    "historical_membership_verified",
    "pit_safe_for_backtest"
] if x in c.columns]

q = q.merge(c[keep].drop_duplicates("ticker"), on="ticker", how="left")

# Priority is deliberately NOT based on classification. It is based on whether
# the stock is research-eligible and has enough liquidity.
def priority(r):
    if not bool(r["research_universe_eligible"]):
        return "EXCLUDED"
    if r.get("primary_class") == "BLUE_CHIP":
        return "CORE"
    if r.get("primary_class") == "CONGLOMERATE":
        return "GROUP_RESEARCH"
    if r.get("secondary_class") == "BUMN_AFFILIATED":
        return "BUMN_RESEARCH"
    if r.get("liquidity_bucket") == "LIQUID_CORE":
        return "LIQUID_CORE"
    return "BROAD_RESEARCH"

q["research_priority"] = q.apply(priority, axis=1)

review = q[~q["research_universe_eligible"]].copy()
review["review_reason"] = review["gate_failures"].replace("", "research gate failed")
review["review_priority"] = "MEDIUM"

summary = {
    "latest_market_date": latest_market_date.date().isoformat(),
    "input_tickers": int(q["ticker"].nunique()),
    "research_universe_eligible": int(q["research_universe_eligible"].sum()),
    "excluded": int((~q["research_universe_eligible"]).sum()),
    "blue_chip_eligible": int(((q["research_universe_eligible"]) & (q["primary_class"] == "BLUE_CHIP")).sum()),
    "conglomerate_eligible": int(((q["research_universe_eligible"]) & (q["primary_class"] == "CONGLOMERATE")).sum()),
    "bumn_eligible": int(((q["research_universe_eligible"]) & (q["secondary_class"] == "BUMN_AFFILIATED")).sum()),
    "liquid_core": int((q["liquidity_bucket"] == "LIQUID_CORE").sum()),
    "liquid_growth_momentum": int((q["liquidity_bucket"] == "LIQUID_GROWTH_MOMENTUM").sum()),
    "liquid_high_vol": int((q["liquidity_bucket"] == "LIQUID_HIGH_VOL").sum()),
    "pit_safe_tickers": int(q.get("pit_safe_for_backtest", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if "pit_safe_for_backtest" in q else 0,
    "engine_changed": False,
    "status": "RESEARCH_UNIVERSE_BUILT__NO_ENGINE_CHANGE",
    "gate_policy": {
        "min_history_rows": MIN_HISTORY,
        "max_lag_days": MAX_LAG_DAYS,
        "min_median_value_20d_idr": MIN_MEDIAN_VALUE_20D,
        "min_median_value_60d_idr": MIN_MEDIAN_VALUE_60D,
        "max_annualized_volatility_flag": MAX_ANNUAL_VOL
    }
}

q.to_csv(DATA / "research_universe.csv", index=False)
review.to_csv(DATA / "research_universe_review_queue.csv", index=False)
(DATA / "research_universe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(DATA / "research_universe_status.json").write_text(json.dumps({
    "status": summary["status"],
    "engine_changed": False,
    "pit_safe_for_backtest": False,
    "note": "This is a research/liquidity gate. It does not create historical PIT membership and does not alter actual_engine.py.",
    "official_idx_context": "IDX Ringkasan Saham exposes volume, value, frequency, listed shares and tradeable shares. This layer currently derives trading value from local OHLCV and does not yet replace it with an official PIT liquidity feed."
}, indent=2), encoding="utf-8")

print("Latest market date:", summary["latest_market_date"])
print("Input tickers:", summary["input_tickers"])
print("Research universe eligible:", summary["research_universe_eligible"])
print("Excluded:", summary["excluded"])
print("Blue-chip eligible:", summary["blue_chip_eligible"])
print("Conglomerate eligible:", summary["conglomerate_eligible"])
print("BUMN eligible:", summary["bumn_eligible"])
print("Liquid core:", summary["liquid_core"])
print("Liquid growth/momentum:", summary["liquid_growth_momentum"])
print("Liquid high volatility:", summary["liquid_high_vol"])
print("STATUS:", summary["status"])
