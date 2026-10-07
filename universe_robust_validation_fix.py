#!/usr/bin/env python3
"""
V1.3E-2 — Robust Universe Validation Fix

Fixes V1.3E-1 pandas boolean-index alignment warnings and adds:
- matched bucket comparison without chained indexing
- bootstrap CI for the difference in mean return
- explicit stability interpretation
- no engine change
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
IHSG = DATA / "idx_ihsg_index.csv"

for f in [PRICE, RESEARCH, IHSG]:
    if not f.exists():
        raise SystemExit(f"Missing required file: {f}")

p = pd.read_csv(PRICE)
p.columns = [str(c).strip().lower() for c in p.columns]
req = {"ticker", "date", "open", "close"}
if not req.issubset(p.columns):
    raise SystemExit(f"Missing columns: {sorted(req-set(p.columns))}")
p["ticker"] = p["ticker"].astype(str).str.upper().str.replace(".JK", "", regex=False)
p["date"] = pd.to_datetime(p["date"], errors="coerce")
for c in ["open", "close"]:
    p[c] = pd.to_numeric(p[c], errors="coerce")
p = p.dropna(subset=["ticker","date","open","close"])
p = p[(p["open"] > 0) & (p["close"] > 0)]
p = p.drop_duplicates(["ticker","date"]).sort_values(["ticker","date"])

r = pd.read_csv(RESEARCH)
r["ticker"] = r["ticker"].astype(str).str.upper().str.replace(".JK", "", regex=False)
eligible = set(r.loc[r["research_universe_eligible"].eq(True), "ticker"])

c = pd.read_csv(CLASS) if CLASS.exists() else pd.DataFrame(columns=["ticker"])
if not c.empty:
    c["ticker"] = c["ticker"].astype(str).str.upper().str.replace(".JK", "", regex=False)

ih = pd.read_csv(IHSG)
ih.columns = [str(x).strip() for x in ih.columns]
if "Price" in ih.columns:
    d = pd.to_datetime(ih["Price"], errors="coerce")
    close = pd.to_numeric(ih.get("Close"), errors="coerce")
else:
    dc = next((x for x in ih.columns if x.lower()=="date"), None)
    cc = next((x for x in ih.columns if x.lower()=="close"), None)
    if dc is None or cc is None:
        raise SystemExit("Cannot parse IHSG.")
    d = pd.to_datetime(ih[dc], errors="coerce")
    close = pd.to_numeric(ih[cc], errors="coerce")
ih = pd.DataFrame({"date": d, "ihsg_close": close}).dropna().drop_duplicates("date").sort_values("date")
ih["ihsg_20d"] = ih["ihsg_close"].pct_change(20)
ih["regime"] = np.select(
    [ih["ihsg_20d"] > 0.02, ih["ihsg_20d"] < -0.02],
    ["BULL", "BEAR"], default="SIDEWAYS"
)

events = []
for ticker, g in p.groupby("ticker", sort=True):
    g = g.reset_index(drop=True)
    for i in range(len(g)-20):
        entry = g.loc[i+1, "open"]
        if not np.isfinite(entry) or entry <= 0:
            continue
        events.append({
            "ticker": ticker,
            "signal_date": g.loc[i, "date"],
            "entry_date": g.loc[i+1, "date"],
            "year": int(g.loc[i+1, "date"].year),
            "ret5": g.loc[i+5, "close"]/entry - 1,
            "ret20": g.loc[i+20, "close"]/entry - 1,
            "research_eligible": ticker in eligible
        })

e = pd.DataFrame(events)
e = e.merge(ih[["date","regime"]].rename(columns={"date":"signal_date"}),
            on="signal_date", how="left")
if not c.empty:
    cols = [x for x in ["ticker","primary_class","secondary_class"] if x in c.columns]
    e = e.merge(c[cols].drop_duplicates("ticker"), on="ticker", how="left")
else:
    e["primary_class"] = np.nan
    e["secondary_class"] = np.nan

def pf(x):
    x = pd.Series(x).dropna()
    gains = x[x>0].sum()
    losses = -x[x<0].sum()
    return float(gains/losses) if losses > 0 else np.nan

def boot_mean_ci(x, seed=42, n=2000):
    x = pd.Series(x).dropna().to_numpy(float)
    if len(x) < 30:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    for i in range(n):
        vals[i] = rng.choice(x, size=len(x), replace=True).mean()
    return float(np.quantile(vals,.025)), float(np.quantile(vals,.975))

def boot_diff_ci(a, b, seed=42, n=2000):
    a = pd.Series(a).dropna().to_numpy(float)
    b = pd.Series(b).dropna().to_numpy(float)
    if len(a) < 30 or len(b) < 30:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    for i in range(n):
        ma = rng.choice(a, size=len(a), replace=True).mean()
        mb = rng.choice(b, size=len(b), replace=True).mean()
        vals[i] = mb - ma
    return float(np.quantile(vals,.025)), float(np.quantile(vals,.975))

def stat(g, universe, dim, bucket):
    x = g["ret5"].dropna()
    lo, hi = boot_mean_ci(x)
    return {
        "universe": universe, "dimension": dim, "bucket": bucket,
        "events": int(len(x)), "unique_tickers": int(g["ticker"].nunique()),
        "win_rate_5d": float((x>0).mean()) if len(x) else np.nan,
        "avg_return_5d": float(x.mean()) if len(x) else np.nan,
        "median_return_5d": float(x.median()) if len(x) else np.nan,
        "profit_factor_5d": pf(x),
        "mean_ci_low": lo, "mean_ci_high": hi
    }

rows=[]
for universe, mask in [("FULL_903", pd.Series(True,index=e.index)),
                        ("RESEARCH_297", e["research_eligible"])]:
    u=e.loc[mask].copy()
    rows.append(stat(u,universe,"overall","ALL"))
    for y,g in u.groupby("year"):
        rows.append(stat(g,universe,"year",str(y)))
    for reg,g in u.groupby("regime",dropna=False):
        rows.append(stat(g,universe,"regime",str(reg)))
    for cls,g in u.groupby("primary_class",dropna=False):
        rows.append(stat(g,universe,"primary_class",str(cls)))

stats=pd.DataFrame(rows)

comparisons=[]
for dim in ["overall","year","regime","primary_class"]:
    a=stats.loc[(stats["universe"]=="FULL_903") & (stats["dimension"]==dim)].copy()
    b=stats.loc[(stats["universe"]=="RESEARCH_297") & (stats["dimension"]==dim)].copy()
    keys=sorted(set(a["bucket"]) & set(b["bucket"]))
    for key in keys:
        aa=a.loc[a["bucket"].eq(key)].iloc[0]
        bb=b.loc[b["bucket"].eq(key)].iloc[0]
        ga=e.loc[
            (e["research_eligible"].eq(False)) &
            ((e["year"].astype(str).eq(key)) if dim=="year" else True) &
            ((e["regime"].astype(str).eq(key)) if dim=="regime" else True) &
            ((e["primary_class"].astype(str).eq(key)) if dim=="primary_class" else True),
            "ret5"
        ]
        gb=e.loc[
            (e["research_eligible"].eq(True)) &
            ((e["year"].astype(str).eq(key)) if dim=="year" else True) &
            ((e["regime"].astype(str).eq(key)) if dim=="regime" else True) &
            ((e["primary_class"].astype(str).eq(key)) if dim=="primary_class" else True),
            "ret5"
        ]
        if dim=="overall":
            ga=e.loc[e["research_eligible"].eq(False),"ret5"]
            gb=e.loc[e["research_eligible"].eq(True),"ret5"]
        dlo,dhi=boot_diff_ci(ga,gb)
        dwin=(bb["win_rate_5d"]-aa["win_rate_5d"])*100
        davg=(bb["avg_return_5d"]-aa["avg_return_5d"])*100
        comparisons.append({
            "dimension":dim,"bucket":key,
            "full_events":int(aa["events"]),"research_events":int(bb["events"]),
            "delta_win_rate_pp":float(dwin),
            "delta_avg_return_pp":float(davg),
            "delta_profit_factor":float(bb["profit_factor_5d"]-aa["profit_factor_5d"])
                if np.isfinite(bb["profit_factor_5d"]) and np.isfinite(aa["profit_factor_5d"]) else np.nan,
            "bootstrap_delta_mean_ci_low":dlo,
            "bootstrap_delta_mean_ci_high":dhi,
            "delta_mean_ci_excludes_zero":bool(np.isfinite(dlo) and np.isfinite(dhi) and (dlo>0 or dhi<0)),
            "positive_both":bool(dwin>0 and davg>0)
        })

cmp=pd.DataFrame(comparisons)
valid=cmp.dropna(subset=["delta_win_rate_pp","delta_avg_return_pp"])
positive_both=int(valid["positive_both"].sum())
total=int(len(valid))

# Conservative interpretation:
# "robust positive" requires the average-return difference CI to be entirely >0
# AND win-rate and average-return deltas both positive.
robust_positive=int(((valid["positive_both"]) &
                     (valid["bootstrap_delta_mean_ci_low"]>0)).sum()) if len(valid) else 0

overall=cmp[cmp["dimension"].eq("overall")]
summary={
    "latest_price_date":p["date"].max().date().isoformat(),
    "full_universe_tickers":int(p["ticker"].nunique()),
    "research_universe_tickers":int(len(eligible)),
    "matched_buckets":total,
    "positive_both_winrate_and_avg_return_buckets":positive_both,
    "positive_both_ratio":float(positive_both/total) if total else np.nan,
    "robust_positive_buckets":robust_positive,
    "overall_comparison":overall.to_dict("records"),
    "method":{
        "entry":"next trading session OPEN",
        "diagnostic_horizon":"5 sessions",
        "bootstrap_resamples":2000,
        "bootstrap_seed":42,
        "regime_proxy":"IHSG trailing 20-session return >2% BULL, <-2% BEAR, otherwise SIDEWAYS",
        "threshold_optimization":False,
        "comparison_fixed":"FULL vs RESEARCH, not in-sample threshold selection"
    },
    "interpretation":"Universe effect remains exploratory unless differences are stable across periods/regimes and survive uncertainty checks.",
    "engine_changed":False,
    "pit_safe_for_backtest":False,
    "status":"ROBUST_UNIVERSE_VALIDATION_FIX_COMPLETE__NO_ENGINE_CHANGE"
}

stats.to_csv(DATA/"universe_robust_validation_stats.csv",index=False)
cmp.to_csv(DATA/"universe_robust_validation_comparison.csv",index=False)
(DATA/"universe_robust_validation_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
(DATA/"universe_robust_validation_status.json").write_text(json.dumps({
    "status":summary["status"],
    "engine_changed":False,
    "pit_safe_for_backtest":False,
    "warnings_fixed":["pandas boolean Series reindexing/chained indexing"],
    "note":"This remains a universe diagnostic, not a strategy profitability test."
},indent=2),encoding="utf-8")

print("Latest price date:",summary["latest_price_date"])
print("Full universe:",summary["full_universe_tickers"])
print("Research universe:",summary["research_universe_tickers"])
print("Matched buckets:",total)
print("Positive both win-rate and average-return:",positive_both)
print("Positive-both ratio:",summary["positive_both_ratio"])
print("Robust positive buckets (delta CI > 0 + both positive):",robust_positive)
print("STATUS:",summary["status"])
