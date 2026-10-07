from pathlib import Path
import numpy as np
import pandas as pd

INFILE = Path("data/broker/broker_summary_daily.csv")
OUTFILE = Path("data/broker/broker_flow_features.csv")

def safe_div(a, b):
    return np.where(np.abs(b) > 0, a / b, np.nan)

def concentration(group, col, n):
    s = group[col].clip(lower=0).sort_values(ascending=False)
    total = s.sum()
    return float(s.head(n).sum() / total) if total > 0 else np.nan

def weighted_big_money_avg(group):
    # Only brokers with positive net value are included.
    g = group[group["net_value"] > 0].copy()
    g = g.dropna(subset=["buy_avg"])
    if g.empty or g["net_value"].sum() <= 0:
        return np.nan
    return float(np.average(g["buy_avg"], weights=g["net_value"]))

def build_daily(df):
    df = df.copy()
    numeric = [
        "buy_freq","buy_volume","buy_value",
        "sell_freq","sell_volume","sell_value",
        "buy_avg","sell_avg"
    ]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["net_value"] = df["buy_value"] - df["sell_value"]
    df["net_volume"] = df["buy_volume"] - df["sell_volume"]
    df["buy_share"] = safe_div(df["buy_value"], df.groupby(["date","ticker"])["buy_value"].transform("sum"))
    df["sell_share"] = safe_div(df["sell_value"], df.groupby(["date","ticker"])["sell_value"].transform("sum"))

    rows = []
    for (date, ticker), g in df.groupby(["date","ticker"], sort=False):
        net = g["net_value"]
        buyers = g[net > 0]
        sellers = g[net < 0]

        row = {
            "date": date,
            "ticker": ticker,
            "gross_buy_value": g["buy_value"].sum(),
            "gross_sell_value": g["sell_value"].sum(),
            "net_value": g["net_value"].sum(),
            "net_volume": g["net_volume"].sum(),
            "buyer_count": int((net > 0).sum()),
            "seller_count": int((net < 0).sum()),
            "top1_buyer_share": concentration(g, "buy_value", 1),
            "top3_buyer_share": concentration(g, "buy_value", 3),
            "top5_buyer_share": concentration(g, "buy_value", 5),
            "top1_seller_share": concentration(g, "sell_value", 1),
            "top3_seller_share": concentration(g, "sell_value", 3),
            "top5_seller_share": concentration(g, "sell_value", 5),
            "estimated_big_money_avg_price": weighted_big_money_avg(g),
            "buy_avg_weighted_all": safe_div(g["buy_value"].sum(), g["buy_volume"].sum()),
            "sell_avg_weighted_all": safe_div(g["sell_value"].sum(), g["sell_volume"].sum()),
        }
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["ticker","date"])
    for w in [5, 20]:
        out[f"net_value_{w}d"] = out.groupby("ticker")["net_value"].transform(
            lambda s: s.rolling(w, min_periods=max(2, w//2)).sum()
        )
        out[f"big_money_avg_{w}d"] = out.groupby("ticker")["estimated_big_money_avg_price"].transform(
            lambda s: s.rolling(w, min_periods=max(2, w//2)).mean()
        )
        out[f"buyer_persistence_{w}d"] = out.groupby("ticker")["net_value"].transform(
            lambda s: (s > 0).rolling(w, min_periods=max(2, w//2)).mean()
        )
        out[f"seller_persistence_{w}d"] = out.groupby("ticker")["net_value"].transform(
            lambda s: (s < 0).rolling(w, min_periods=max(2, w//2)).mean()
        )

    out["flow_pressure"] = safe_div(
        out["net_value"],
        out["gross_buy_value"] + out["gross_sell_value"]
    )
    out["accumulation_evidence"] = (
        (out["net_value_20d"] > 0) &
        (out["buyer_persistence_20d"] >= 0.55)
    )
    out["distribution_evidence"] = (
        (out["net_value_20d"] < 0) &
        (out["seller_persistence_20d"] >= 0.55)
    )
    return out

def main():
    if not INFILE.exists():
        raise FileNotFoundError(INFILE)
    df = pd.read_csv(INFILE)
    if df.empty:
        raise RuntimeError("Broker summary file is empty")
    out = build_daily(df)
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTFILE, index=False)

if __name__ == "__main__":
    main()
