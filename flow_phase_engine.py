from pathlib import Path
import numpy as np
import pandas as pd

PRICE = Path("data/idx_stock_prices.csv")
FLOW = Path("data/broker/broker_flow_features.csv")
FOREIGN = Path("data/broker/foreign_flow_daily.csv")
OUT = Path("data/broker/flow_phase_features.csv")

def zscore(s, w=60):
    m = s.rolling(w, min_periods=max(10, w//2)).mean()
    sd = s.rolling(w, min_periods=max(10, w//2)).std()
    return (s - m) / sd.replace(0, np.nan)

def main():
    p = pd.read_csv(PRICE)
    p.columns = [c.lower() for c in p.columns]
    if "ticker" not in p or "date" not in p:
        raise RuntimeError("idx_stock_prices.csv needs ticker/date")
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values(["ticker","date"])

    for c in ["close","high","low","open","volume"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")

    p["ma20"] = p.groupby("ticker")["close"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    p["ma50"] = p.groupby("ticker")["close"].transform(lambda s: s.rolling(50, min_periods=25).mean())
    p["ret20"] = p.groupby("ticker")["close"].transform(lambda s: s.pct_change(20))
    p["range20"] = p.groupby("ticker")["close"].transform(
        lambda s: s.rolling(20, min_periods=10).max() / s.rolling(20, min_periods=10).min() - 1
    )
    p["vol20"] = p.groupby("ticker")["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    p["volume_ratio"] = p["volume"] / p["vol20"].replace(0, np.nan)
    p["hh20"] = p.groupby("ticker")["high"].transform(lambda s: s.rolling(20, min_periods=10).max())
    p["ll20"] = p.groupby("ticker")["low"].transform(lambda s: s.rolling(20, min_periods=10).min())
    p["breakout20"] = p["close"] >= p["hh20"].shift(1)
    p["breakdown20"] = p["close"] <= p["ll20"].shift(1)
    p["atr14"] = p.groupby("ticker").apply(
        lambda g: (pd.concat([
            g["high"] - g["low"],
            (g["high"] - g["close"].shift()).abs(),
            (g["low"] - g["close"].shift()).abs()
        ], axis=1).max(axis=1)).rolling(14, min_periods=7).mean()
    ).reset_index(level=0, drop=True)
    p["atr_pct"] = p["atr14"] / p["close"]

    f = pd.read_csv(FLOW)
    f["date"] = pd.to_datetime(f["date"])
    df = p.merge(f, on=["ticker","date"], how="left")

    if FOREIGN.exists():
        ff = pd.read_csv(FOREIGN)
        ff["date"] = pd.to_datetime(ff["date"])
        df = df.merge(ff, on=["ticker","date"], how="left", suffixes=("", "_foreign"))

    # Evidence is deliberately transparent rather than a single opaque score.
    df["flow_positive"] = df["net_value_20d"] > 0
    df["flow_negative"] = df["net_value_20d"] < 0
    df["trend_positive"] = (df["ma20"] > df["ma50"]) & (df["ret20"] > 0)
    df["trend_negative"] = (df["ma20"] < df["ma50"]) & (df["ret20"] < 0)
    df["compressed"] = df["range20"] < df["atr_pct"].clip(lower=0.001) * 12
    df["volume_confirmed"] = df["volume_ratio"] >= 1.5
    df["extended"] = df["ret20"] > 0.15
    df["weak_progress"] = df["volume_ratio"] >= 1.5

    # Phase logic is a research heuristic; thresholds are not validated trading rules.
    phase = np.full(len(df), "INCONCLUSIVE", dtype=object)
    confidence = np.full(len(df), "LOW", dtype=object)

    acc = (
        df["flow_positive"] &
        (df["buyer_persistence_20d"] >= 0.55) &
        (~df["extended"]) &
        (df["ret20"].fillna(0) > -0.05)
    )
    markup = (
        df["flow_positive"] &
        df["trend_positive"] &
        (df["breakout20"] | df["volume_confirmed"])
    )
    distribution = (
        df["flow_negative"] &
        (df["seller_persistence_20d"] >= 0.55) &
        (df["extended"] | (~df["trend_positive"]))
    )
    markdown = (
        df["flow_negative"] &
        df["trend_negative"] &
        (df["breakdown20"] | df["volume_confirmed"])
    )

    phase[acc] = "ACCUMULATION"
    phase[markup] = "MARK-UP"
    phase[distribution] = "DISTRIBUTION"
    phase[markdown] = "MARK-DOWN"

    # More specific phase wins when conditions overlap.
    confidence[(acc | markup | distribution | markdown).to_numpy()] = "MEDIUM"
    confidence[(markup & df["volume_confirmed"]).to_numpy()] = "HIGH"
    confidence[(markdown & df["volume_confirmed"]).to_numpy()] = "HIGH"

    df["flow_phase"] = phase
    df["phase_confidence"] = confidence

    evidence_cols = [
        "net_value_20d","buyer_persistence_20d","seller_persistence_20d",
        "top3_buyer_share","top3_seller_share",
        "estimated_big_money_avg_price","ret20","volume_ratio",
        "breakout20","breakdown20","flow_phase","phase_confidence"
    ]
    keep = ["date","ticker"] + [c for c in evidence_cols if c in df.columns]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[keep].sort_values(["date","ticker"]).to_csv(OUT, index=False)

if __name__ == "__main__":
    main()
