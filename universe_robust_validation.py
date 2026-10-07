#!/usr/bin/env python3
"""
V1.3E-1 Robust Universe Validation

Tests whether the V1.3D research universe advantage is stable across:
- calendar year
- market regime (proxy from IHSG returns)
- classification

No threshold optimization. No changes to actual_engine.py.
The analysis is diagnostic and does not establish strategy profitability.
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
eligible = set(r.loc[r["research_universe_eligible"] == True, "ticker"])

c = pd.read_csv(CLASS) if CLASS.exists() else pd.DataFrame(columns=["ticker"])
if not c.empty:
    c["ticker"] = c["ticker"].astype(str).str.upper().str.replace(".JK", "", regex=False)

# IHSG parser supports common schemas from the project.
ih = pd.read_csv(IHSG)
ih.columns = [str(x).strip() for x in ih.columns]
if "Price" in ih.columns:
    ih_date = pd.to_datetime(ih["Price"], errors="coerce")
    ih_close = pd.to_numeric(ih.get("Close"), errors="coerce")
elif "date" in [x.lower() for x in ih.columns]:
    dc = next(x for x in ih.columns if x.lower() == "date")
    cc = next(x for x in ih.columns if x.lower() == "close")
    ih_date = pd.to_datetime(ih[dc], errors="coerce")
    ih_close = pd.to_numeric(ih[cc], errors="coerce")
else:
    raise SystemExit("Cannot parse IHSG date/close columns.")
ih = pd.DataFrame({"date": ih_date, "ihsg_close": ih_close}).dropna().drop_duplicates("date").sort_values("date")
ih["ihsg_ret"] = ih["ihsg_close"].pct_change()
# Regime proxy is deliberately simple and fixed:
# BULL if trailing 20 sessions return > +2%; BEAR if < -2%; otherwise SIDEWAYS.
ih["ihsg_20d"] = ih["ihsg_close"].pct_change(20)
ih["regime"] = np.select(
    [ih["ihsg_20d"] > 0.02, ih["ihsg_20d"] < -0.02],
    ["BULL", "BEAR"],
    default="SIDEWAYS"
)

# Forward events.
events = []
for ticker, g in p.groupby("ticker", sort=True):
    g = g.reset_index(drop=True)
    for i in range(len(g) - 20):
        entry_date = g.loc[i+1, "date"]
        entry = g.loc[i+1, "open"]
        if not np.isfinite(entry) or entry <= 0:
            continue
        signal_date = g.loc[i, "date"]
        ret5 = g.loc[i+5, "close"] / entry - 1
        ret20 = g.loc[i+20, "close"] / entry - 1
        events.append({
            "ticker": ticker,
            "signal_date": signal_date,
            "entry_date": entry_date,
            "year": int(entry_date.year),
            "ret5": ret5,
            "ret20": ret20,
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

def profit_factor(x):
    x = pd.Series(x).dropna()
    gains = x[x > 0].sum()
    losses = -x[x < 0].sum()
    return float(gains / losses) if losses > 0 else np.nan

def bootstrap_ci(x, seed=42, n=2000):
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    if len(x) < 30:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(n):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))

def stat_row(g, universe, dimension, bucket):
    x = g["ret5"].dropna()
    lo, hi = bootstrap_ci(x)
    return {
        "universe": universe,
        "dimension": dimension,
        "bucket": bucket,
        "events": int(len(x)),
        "unique_tickers": int(g["ticker"].nunique()),
        "win_rate_5d": float((x > 0).mean()) if len(x) else np.nan,
        "avg_return_5d": float(x.mean()) if len(x) else np.nan,
        "median_return_5d": float(x.median()) if len(x) else np.nan,
        "profit_factor_5d": profit_factor(x),
        "bootstrap_avg_return_ci_low": lo,
        "bootstrap_avg_return_ci_high": hi,
    }

rows = []
# Overall, year, regime, class
for universe, mask in [
    ("FULL_903", pd.Series(True, index=e.index)),
    ("RESEARCH_297", e["research_eligible"])
]:
    u = e[mask]
    rows.append(stat_row(u, universe, "overall", "ALL"))
    for year, g in u.groupby("year"):
        rows.append(stat_row(g, universe, "year", str(year)))
    for regime, g in u.groupby("regime", dropna=False):
        rows.append(stat_row(g, universe, "regime", str(regime)))
    for cls, g in u.groupby("primary_class", dropna=False):
        rows.append(stat_row(g, universe, "primary_class", str(cls)))

stats = pd.DataFrame(rows)

# Direct research-minus-full comparison by matched bucket.
comparisons = []
for dim in ["overall","year","regime","primary_class"]:
    a = stats[stats["universe"]=="FULL_903"][stats["dimension"]==dim]
    b = stats[stats["universe"]=="RESEARCH_297"][stats["dimension"]==dim]
    if dim == "overall":
        keys = ["ALL"]
    else:
        keys = sorted(set(a["bucket"]) & set(b["bucket"]))
    for key in keys:
        aa = a[a["bucket"]==key].iloc[0]
        bb = b[b["bucket"]==key].iloc[0]
        comparisons.append({
            "dimension": dim,
            "bucket": key,
            "full_events": int(aa["events"]),
            "research_events": int(bb["events"]),
            "delta_win_rate_pp": float((bb["win_rate_5d"]-aa["win_rate_5d"])*100),
            "delta_avg_return_pp": float((bb["avg_return_5d"]-aa["avg_return_5d"])*100),
            "delta_profit_factor": (
                float(bb["profit_factor_5d"]-aa["profit_factor_5d"])
                if np.isfinite(bb["profit_factor_5d"]) and np.isfinite(aa["profit_factor_5d"]) else np.nan
            ),
            "research_ci_excludes_zero": bool(
                np.isfinite(bb["bootstrap_avg_return_ci_low"])
                and bb["bootstrap_avg_return_ci_low"] > 0
            ) if np.isfinite(bb["bootstrap_avg_return_ci_low"]) else False,
        })

cmp = pd.DataFrame(comparisons)

# A conservative stability summary: count matched buckets where research has
# positive delta in average return and positive delta in win rate.
valid = cmp.dropna(subset=["delta_win_rate_pp","delta_avg_return_pp"])
stable_positive = int(((valid["delta_win_rate_pp"] > 0) & (valid["delta_avg_return_pp"] > 0)).sum())
stable_total = int(len(valid))

summary = {
    "latest_price_date": p["date"].max().date().isoformat(),
    "full_universe_tickers": int(p["ticker"].nunique()),
    "research_universe_tickers": int(len(eligible)),
    "matched_bucket_count": stable_total,
    "buckets_positive_both_winrate_and_avg_return": stable_positive,
    "positive_both_ratio": float(stable_positive/stable_total) if stable_total else np.nan,
    "overall": cmp[(cmp["dimension"]=="overall")].to_dict("records"),
    "method": {
        "entry": "next trading session OPEN",
        "forward_horizon": "5 sessions for diagnostic consistency",
        "bootstrap": "2000 resamples, fixed seed 42, CI on mean return",
        "regime_proxy": "IHSG trailing 20-session return >2% BULL, <-2% BEAR, otherwise SIDEWAYS",
        "threshold_optimization": False
    },
    "interpretation": "Diagnostic only; not proof of strategy profitability and not PIT-safe backtest evidence.",
    "engine_changed": False,
    "pit_safe_for_backtest": False,
    "status": "ROBUST_UNIVERSE_VALIDATION_COMPLETE__NO_ENGINE_CHANGE"
}

stats.to_csv(DATA / "universe_robust_validation_stats.csv", index=False)
cmp.to_csv(DATA / "universe_robust_validation_comparison.csv", index=False)
(DATA / "universe_robust_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(DATA / "universe_robust_validation_status.json").write_text(json.dumps({
    "status": summary["status"],
    "engine_changed": False,
    "pit_safe_for_backtest": False,
    "note": "This validates stability of the universe effect across periods/regimes/classes. It does not enable the research universe in actual_engine."
}, indent=2), encoding="utf-8")

print("Latest price date:", summary["latest_price_date"])
print("Full universe:", summary["full_universe_tickers"])
print("Research universe:", summary["research_universe_tickers"])
print("Matched buckets:", stable_total)
print("Positive both win-rate and average-return buckets:", stable_positive)
print("Positive-both ratio:", summary["positive_both_ratio"])
print("STATUS:", summary["status"])
