import json, os
import pandas as pd
import numpy as np

INPUT = "data/macro/macro_inputs.csv"
SEED = "data/macro/macro_inputs_seed_2026-10-07.csv"
OUT = "data/macro/macro_regime_latest.csv"
STATUS = "data/topdown_status.json"

def load_inputs():
    path = INPUT if os.path.exists(INPUT) and os.path.getsize(INPUT) > 50 else SEED
    df = pd.read_csv(path)
    if df.empty:
        return df, path
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["date","metric","value"]).copy(), path

def latest_map(df):
    return df.sort_values("date").groupby("metric", as_index=False).tail(1).set_index("metric")

def score_macro(r):
    # Conservative: only score dimensions that are present.
    scores = {}
    if "bi_rate" in r.index:
        # Level alone is not bullish/bearish; use change if history exists elsewhere.
        scores["bi_rate"] = 0
    if "inflation_yoy" in r.index:
        x = float(r.loc["inflation_yoy","value"])
        scores["inflation_yoy"] = 1 if x <= 3.5 else (-1 if x >= 5 else 0)
    if "usd_idr" in r.index:
        scores["usd_idr"] = 0  # level requires history; do not invent direction.
    if "indonia" in r.index:
        scores["indonia"] = 0
    if "reserves_usd_bn" in r.index:
        scores["reserves_usd_bn"] = 0
    known = [v for v in scores.values()]
    if not known:
        regime = "UNKNOWN"
    else:
        s = sum(known)
        regime = "RISK_ON" if s >= 2 else ("RISK_OFF" if s <= -2 else "NEUTRAL")
    return scores, regime

def main():
    df, source = load_inputs()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    os.makedirs(os.path.dirname(STATUS), exist_ok=True)
    if df.empty:
        out = pd.DataFrame([{"regime":"UNKNOWN","evidence_score":0,"data_status":"NO_DATA"}])
        out.to_csv(OUT, index=False)
        json.dump({"status":"NO_DATA","source":source}, open(STATUS,"w"), indent=2)
        return
    r = latest_map(df)
    scores, regime = score_macro(r)
    rows = []
    for metric, score in scores.items():
        rows.append({
            "metric": metric,
            "value": float(r.loc[metric,"value"]),
            "unit": r.loc[metric,"unit"],
            "signal_score": score,
            "source": r.loc[metric,"source"],
            "source_date": str(r.loc[metric,"source_date"]),
            "confidence": r.loc[metric,"confidence"],
        })
    out = pd.DataFrame(rows)
    evidence_score = int(out["signal_score"].sum()) if not out.empty else 0
    out["regime"] = regime
    out["evidence_score"] = evidence_score
    out["data_status"] = "PARTIAL" if len(out) < 6 else "AVAILABLE"
    out.to_csv(OUT, index=False)
    json.dump({
        "status":"SUCCESS",
        "source":source,
        "regime":regime,
        "evidence_score":evidence_score,
        "metrics_available":sorted(list(r.index)),
        "warning":"Current/static macro observations are not a forecast and must be treated as point-in-time only by source_date."
    }, open(STATUS,"w"), indent=2)

if __name__ == "__main__":
    main()
