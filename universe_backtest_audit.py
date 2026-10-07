#!/usr/bin/env python3
"""
V1.3E — Research Universe Backtest Audit

Purpose:
Compare the same existing rule/model logic on:
A) full available 903-ticker expansion universe
B) V1.3D research universe

This is an AUDIT ONLY. It does not modify actual_engine.py and does not
claim that a better result proves predictive edge.

The audit uses simple, transparent forward-return diagnostics:
- next-session open -> 5-session close return
- next-session open -> 20-session close return
- hit rate
- average/median return
- profit factor
- max drawdown of an equal-weight sequential portfolio proxy
- counts
- classification buckets where available

No threshold is optimized on the comparison sample.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

PRICE = DATA / "idx_stock_prices_expansion.csv"
RESEARCH = DATA / "research_universe.csv"
CLASS = DATA / "universe_classification.csv"

if not PRICE.exists():
    raise SystemExit("Missing idx_stock_prices_expansion.csv")
if not RESEARCH.exists():
    raise SystemExit("Missing research_universe.csv; run V1.3D first.")

p = pd.read_csv(PRICE)
p.columns = [str(c).strip().lower() for c in p.columns]
required = {"ticker", "date", "open", "close"}
if not required.issubset(p.columns):
    raise SystemExit(f"Missing required price columns: {sorted(required-set(p.columns))}")

p["ticker"] = p["ticker"].astype(str).str.upper().str.replace(".JK", "", regex=False)
p["date"] = pd.to_datetime(p["date"], errors="coerce")
for c in ["open", "close"]:
    p[c] = pd.to_numeric(p[c], errors="coerce")
p = p.dropna(subset=["ticker", "date", "open", "close"])
p = p[(p["open"] > 0) & (p["close"] > 0)]
p = p.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"])

r = pd.read_csv(RESEARCH)
r["ticker"] = r["ticker"].astype(str).str.upper().str.replace(".JK", "", regex=False)
eligible = set(r.loc[r["research_universe_eligible"] == True, "ticker"])

if CLASS.exists():
    c = pd.read_csv(CLASS)
    c["ticker"] = c["ticker"].astype(str).str.upper().str.replace(".JK", "", regex=False)
else:
    c = pd.DataFrame(columns=["ticker"])

# Build forward-return event table. This is deliberately independent of the
# current actual_engine scoring, so it measures universe effects without
# silently changing the engine.
events = []
for ticker, g in p.groupby("ticker", sort=True):
    g = g.reset_index(drop=True)
    for i in range(len(g) - 20):
        entry = g.loc[i + 1, "open"]
        if not np.isfinite(entry) or entry <= 0:
            continue
        ret5 = g.loc[i + 5, "close"] / entry - 1.0
        ret20 = g.loc[i + 20, "close"] / entry - 1.0
        events.append({
            "ticker": ticker,
            "signal_date": g.loc[i, "date"],
            "entry_date": g.loc[i + 1, "date"],
            "entry_open": entry,
            "ret5": ret5,
            "ret20": ret20,
            "research_eligible": ticker in eligible,
        })

e = pd.DataFrame(events)
if e.empty:
    raise SystemExit("No forward-return events produced.")

def pf(x):
    wins = x[x > 0].sum()
    losses = -x[x < 0].sum()
    return float(wins / losses) if losses > 0 else np.nan

def max_dd(returns):
    if len(returns) == 0:
        return np.nan
    eq = (1 + returns).cumprod()
    peak = eq.cummax()
    return float((eq / peak - 1).min())

def metrics(x, label):
    x = pd.Series(x).dropna()
    return {
        "universe": label,
        "events": int(len(x)),
        "unique_tickers": int(e.loc[x.index, "ticker"].nunique()) if len(x) else 0,
        "win_rate_5d": float((x > 0).mean()) if len(x) else np.nan,
        "avg_return_5d": float(x.mean()) if len(x) else np.nan,
        "median_return_5d": float(x.median()) if len(x) else np.nan,
        "profit_factor_5d": pf(x) if len(x) else np.nan,
        "max_drawdown_sequential_5d": max_dd(x),
    }

all5 = metrics(e["ret5"], "FULL_903")
res5 = metrics(e.loc[e["research_eligible"], "ret5"], "RESEARCH_297")

# 20-session diagnostics are included as a research horizon, not a holding rule.
all20 = metrics(e["ret20"], "FULL_903_20D")
res20 = metrics(e.loc[e["research_eligible"], "ret20"], "RESEARCH_297_20D")

summary = {
    "latest_price_date": p["date"].max().date().isoformat(),
    "full_universe_tickers": int(p["ticker"].nunique()),
    "research_universe_tickers": int(len(eligible)),
    "metrics": [all5, res5, all20, res20],
    "comparison": {
        "avg_return_5d_delta_research_minus_full": float(res5["avg_return_5d"] - all5["avg_return_5d"]),
        "win_rate_5d_delta_research_minus_full": float(res5["win_rate_5d"] - all5["win_rate_5d"]),
        "profit_factor_5d_delta_research_minus_full": (
            float(res5["profit_factor_5d"] - all5["profit_factor_5d"])
            if np.isfinite(res5["profit_factor_5d"]) and np.isfinite(all5["profit_factor_5d"]) else np.nan
        ),
    },
    "method_note": "This is not a backtest of the current Opportunity Score. It is a universe-effect audit using fixed forward-return horizons and no threshold optimization.",
    "engine_changed": False,
    "pit_safe_for_backtest": False,
    "status": "UNIVERSE_BACKTEST_AUDIT_COMPLETE__NO_ENGINE_CHANGE",
}

pd.DataFrame([all5, res5, all20, res20]).to_csv(DATA / "universe_backtest_comparison.csv", index=False)

# Per-class diagnostics for the research universe.
if "primary_class" in c.columns:
    e2 = e[e["research_eligible"]].merge(
        c[["ticker", "primary_class", "secondary_class"]].drop_duplicates("ticker"),
        on="ticker", how="left"
    )
    class_rows = []
    for label, g in e2.groupby("primary_class", dropna=False):
        x = g["ret5"].dropna()
        class_rows.append({
            "primary_class": str(label),
            "events": int(len(x)),
            "unique_tickers": int(g["ticker"].nunique()),
            "win_rate_5d": float((x > 0).mean()) if len(x) else np.nan,
            "avg_return_5d": float(x.mean()) if len(x) else np.nan,
            "median_return_5d": float(x.median()) if len(x) else np.nan,
            "profit_factor_5d": pf(x) if len(x) else np.nan,
        })
    pd.DataFrame(class_rows).to_csv(DATA / "universe_backtest_by_class.csv", index=False)

(DATA / "universe_backtest_comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(DATA / "universe_backtest_audit_status.json").write_text(json.dumps({
    "status": summary["status"],
    "engine_changed": False,
    "pit_safe_for_backtest": False,
    "warning": "The audit compares forward-return distributions. It is not evidence of a profitable strategy and is not a replacement for the validated actual_engine walk-forward tests."
}, indent=2), encoding="utf-8")

print("Latest price date:", summary["latest_price_date"])
print("Full universe tickers:", summary["full_universe_tickers"])
print("Research universe tickers:", summary["research_universe_tickers"])
print("FULL 5D avg:", all5["avg_return_5d"])
print("RESEARCH 5D avg:", res5["avg_return_5d"])
print("FULL 5D win:", all5["win_rate_5d"])
print("RESEARCH 5D win:", res5["win_rate_5d"])
print("FULL 5D PF:", all5["profit_factor_5d"])
print("RESEARCH 5D PF:", res5["profit_factor_5d"])
print("STATUS:", summary["status"])
