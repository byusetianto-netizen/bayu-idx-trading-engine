#!/usr/bin/env python3
"""
V1.3C-5 Universe Classification Layer

Purpose:
- Classify the current research universe into transparent, auditable buckets.
- Keep classification separate from the trading engine.
- Never infer historical index membership from today's membership.
- Preserve source/date/confidence for every externally sourced classification.

Inputs:
  data/idx_stock_prices_expansion.csv
  data/security_master.csv or data/historical_security_master_v13c2.csv (optional)
  data/universe_audit.csv (optional)
  data/universe_classification_seed.csv (bundled seed)

Outputs:
  data/universe_classification.csv
  data/universe_classification_review_queue.csv
  data/universe_classification_summary.json
  data/universe_classification_status.json
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

PRICE_FILES = [
    DATA / "idx_stock_prices_expansion.csv",
    DATA / "idx_stock_prices.csv",
]
SEED = DATA / "universe_classification_seed.csv"

def read_first(paths):
    for p in paths:
        if p.exists():
            return pd.read_csv(p)
    return pd.DataFrame()

def norm_ticker(x):
    return str(x).strip().upper().replace(".JK", "")

prices = read_first(PRICE_FILES)
if prices.empty or "ticker" not in prices.columns:
    raise SystemExit("Price universe not found.")

prices["ticker"] = prices["ticker"].map(norm_ticker)
universe = sorted(prices["ticker"].dropna().unique())

seed = pd.read_csv(SEED) if SEED.exists() else pd.DataFrame()
if not seed.empty:
    seed["ticker"] = seed["ticker"].map(norm_ticker)

# Collapse multiple seed rows into a single auditable row per ticker.
# Membership is represented as a semicolon-separated list.
rows = []
for ticker in universe:
    s = seed[seed["ticker"] == ticker] if not seed.empty else pd.DataFrame()

    index_memberships = []
    categories = []
    groups = []
    sources = []
    dates = []
    confidences = []
    bases = []

    for _, r in s.iterrows():
        for col, target in [
            ("index_membership", index_memberships),
            ("category", categories),
            ("group_name", groups),
        ]:
            v = str(r.get(col, "")).strip()
            if v and v.lower() != "nan" and v not in target:
                target.append(v)
        for col, target in [
            ("source", sources),
            ("source_date", dates),
            ("confidence", confidences),
            ("basis", bases),
        ]:
            v = str(r.get(col, "")).strip()
            if v and v.lower() != "nan" and v not in target:
                target.append(v)

    if any("LQ45" in x for x in index_memberships):
        primary = "BLUE_CHIP"
    elif any("IDX30" in x or "IDX80" in x for x in index_memberships):
        primary = "BLUE_CHIP"
    elif groups:
        primary = "CONGLOMERATE"
    elif categories:
        primary = categories[0]
    else:
        primary = "UNCLASSIFIED"

    if groups:
        secondary = "CONGLOMERATE"
    elif any("BUMN" in x.upper() for x in categories):
        secondary = "BUMN_AFFILIATED"
    elif any("LQ45" in x for x in index_memberships):
        secondary = "BLUE_CHIP"
    else:
        secondary = ""

    confidence = "HIGH" if any(c == "HIGH" for c in confidences) else ("MEDIUM" if confidences else "LOW")

    rows.append({
        "ticker": ticker,
        "primary_class": primary,
        "secondary_class": secondary,
        "index_membership_current_snapshot": ";".join(index_memberships),
        "conglomerate_group_current_snapshot": ";".join(groups),
        "other_category": ";".join(categories),
        "classification_confidence": confidence,
        "source": ";".join(sources),
        "source_date": ";".join(dates),
        "basis": ";".join(bases),
        "historical_membership_verified": False,
        "pit_safe_for_backtest": False,
        "notes": (
            "Current/recent classification snapshot only; not valid as historical "
            "index membership unless dated membership evidence is supplied."
        ),
    })

out_df = pd.DataFrame(rows)

# Add a simple quantitative research-risk bucket, deliberately separate from
# blue-chip/conglomerate labels. This is not an investment recommendation.
p = prices.copy()
for c in ["close", "volume"]:
    if c in p.columns:
        p[c] = pd.to_numeric(p[c], errors="coerce")
p = p.sort_values(["ticker", "date"]) if "date" in p.columns else p

metrics = []
for ticker, g in p.groupby("ticker"):
    g = g.dropna(subset=["close"]).copy()
    if len(g) < 20:
        continue
    ret = g["close"].pct_change()
    med_value = np.nan
    if "volume" in g.columns:
        med_value = np.nanmedian((g["close"] * g["volume"]).tail(60))
    vol20 = np.nanstd(ret.tail(20)) * np.sqrt(252) if ret.notna().sum() >= 10 else np.nan
    metrics.append({
        "ticker": ticker,
        "median_value_60d": med_value,
        "annualized_vol_20d": vol20,
        "history_rows": len(g),
    })

m = pd.DataFrame(metrics)
out_df = out_df.merge(m, on="ticker", how="left")

def risk_bucket(r):
    v = r.get("annualized_vol_20d", np.nan)
    value = r.get("median_value_60d", np.nan)
    if pd.notna(value) and value >= 5_000_000_000 and pd.notna(v) and v < 0.45:
        return "LIQUID_CORE"
    if pd.notna(value) and value >= 1_000_000_000 and pd.notna(v) and v < 0.75:
        return "LIQUID_GROWTH_OR_MOMENTUM"
    if pd.notna(v) and v >= 0.75:
        return "HIGH_VOLATILITY"
    return "RESEARCH_ONLY"

out_df["quant_research_bucket"] = out_df.apply(risk_bucket, axis=1)

# Review queue: every classification with a source that is not explicitly
# point-in-time verified should be reviewed before historical backtesting.
review = out_df[
    (out_df["primary_class"] != "UNCLASSIFIED")
    | (out_df["secondary_class"] != "")
].copy()
review["review_priority"] = np.where(
    review["classification_confidence"].eq("HIGH"), "MEDIUM", "HIGH"
)
review["review_reason"] = (
    "Classification is a current/recent snapshot; historical effective dates "
    "are not verified. Do not use it to reconstruct historical membership."
)

summary = {
    "universe_tickers": int(len(universe)),
    "classified_tickers": int((out_df["primary_class"] != "UNCLASSIFIED").sum()),
    "unclassified_tickers": int((out_df["primary_class"] == "UNCLASSIFIED").sum()),
    "blue_chip_tickers": int((out_df["primary_class"] == "BLUE_CHIP").sum()),
    "conglomerate_tickers": int((out_df["primary_class"] == "CONGLOMERATE").sum()),
    "bumn_affiliated_secondary": int((out_df["secondary_class"] == "BUMN_AFFILIATED").sum()),
    "pit_safe_tickers": int(out_df["pit_safe_for_backtest"].sum()),
    "engine_changed": False,
    "status": "CLASSIFICATION_AUDIT_COMPLETE__NO_ENGINE_CHANGE",
}

out_df.to_csv(DATA / "universe_classification.csv", index=False)
review.to_csv(DATA / "universe_classification_review_queue.csv", index=False)
(DATA / "universe_classification_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(DATA / "universe_classification_status.json").write_text(
    json.dumps({
        "status": summary["status"],
        "source": "IDX index definitions + dated secondary current snapshot seed",
        "historical_membership_verified": False,
        "pit_safe_for_backtest": False,
        "engine_changed": False,
        "note": "This layer classifies the research universe only. It does not enable PIT backtesting.",
    }, indent=2),
    encoding="utf-8",
)

print(f"Universe tickers: {len(universe)}")
print(f"Classified: {summary['classified_tickers']}")
print(f"Blue-chip: {summary['blue_chip_tickers']}")
print(f"Conglomerate: {summary['conglomerate_tickers']}")
print(f"BUMN-affiliated secondary: {summary['bumn_affiliated_secondary']}")
print(f"Unclassified: {summary['unclassified_tickers']}")
print("STATUS:", summary["status"])
