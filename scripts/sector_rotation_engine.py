import os, json
import pandas as pd
import numpy as np

PRICE = "data/idx_stock_prices.csv"
IHSG = "data/idx_ihsg_index.csv"
MAP = "data/sector/sector_map.csv"
OUT = "data/sector/sector_rotation_latest.csv"
TS = "data/sector/sector_rotation_timeseries.csv"
STATUS = "data/topdown_status.json"

def clean_price(df):
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}
    df = df.rename(columns={cols.get("date","date"):"date", cols.get("ticker","ticker"):"ticker",
                            cols.get("close","close"):"close", cols.get("volume","volume"):"volume"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna(subset=["date","ticker","close"])

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if not (os.path.exists(PRICE) and os.path.exists(MAP) and os.path.exists(IHSG)):
        pd.DataFrame().to_csv(OUT, index=False)
        json.dump({"status":"WAITING_FOR_DATA","reason":"Need price, IHSG and sector_map"}, open(STATUS,"w"), indent=2)
        return

    px = clean_price(pd.read_csv(PRICE))
    sm = pd.read_csv(MAP)
    required = {"ticker","sector_code","sector_name"}
    if not required.issubset(sm.columns):
        json.dump({"status":"INVALID_SECTOR_MAP","required":sorted(required)}, open(STATUS,"w"), indent=2)
        return

    px = px.merge(sm[["ticker","sector_code","sector_name"]].drop_duplicates("ticker"), on="ticker", how="inner")
    if px.empty:
        json.dump({"status":"NO_SECTOR_OVERLAP"}, open(STATUS,"w"), indent=2)
        return

    ih = pd.read_csv(IHSG)
    ihcols = {c.lower(): c for c in ih.columns}
    ih = ih.rename(columns={ihcols.get("date","date"):"date", ihcols.get("close","close"):"close"})
    ih["date"] = pd.to_datetime(ih["date"], errors="coerce")
    ih["close"] = pd.to_numeric(ih["close"], errors="coerce")
    ih = ih.dropna(subset=["date","close"]).sort_values("date")
    ih["ret20"] = ih["close"].pct_change(20)
    ih["ret60"] = ih["close"].pct_change(60)

    # Equal-weight stock returns by sector; avoids market-cap assumptions.
    g = px.sort_values(["sector_code","ticker","date"])
    g["ret20"] = g.groupby("ticker")["close"].pct_change(20)
    g["ret60"] = g.groupby("ticker")["close"].pct_change(60)
    g["above_ma20"] = g["close"] > g.groupby("ticker")["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())

    latest = g["date"].max()
    rows = []
    for (scode, sname), d in g.groupby(["sector_code","sector_name"]):
        d = d[d["date"] <= latest]
        recent = d[d["date"] == d["date"].max()]
        r20 = recent["ret20"].median()
        r60 = recent["ret60"].median()
        breadth = recent["above_ma20"].mean()
        ih20 = ih.loc[ih["date"] == recent["date"].max(), "ret20"]
        ih60 = ih.loc[ih["date"] == recent["date"].max(), "ret60"]
        ih20 = float(ih20.iloc[0]) if len(ih20) else np.nan
        ih60 = float(ih60.iloc[0]) if len(ih60) else np.nan
        rs20 = r20 - ih20 if pd.notna(ih20) else np.nan
        rs60 = r60 - ih60 if pd.notna(ih60) else np.nan
        # Descriptive score, not a probability.
        parts = [x for x in [r20, r60, rs20, rs60, breadth-0.5] if pd.notna(x)]
        score = float(np.mean(parts)) if parts else np.nan
        label = "UNKNOWN"
        if pd.notna(score):
            if score >= 0.08: label = "LEADERSHIP"
            elif score >= 0.03: label = "IMPROVING"
            elif score > -0.03: label = "NEUTRAL"
            elif score > -0.08: label = "WEAKENING"
            else: label = "LAGGING"
        rows.append({
            "date": latest, "sector_code":scode, "sector_name":sname,
            "median_ret20":r20, "median_ret60":r60,
            "rs_vs_ihsg20":rs20, "rs_vs_ihsg60":rs60,
            "breadth_above_ma20":breadth, "rotation_score":score,
            "rotation_state":label, "stock_count":recent["ticker"].nunique()
        })

    out = pd.DataFrame(rows).sort_values("rotation_score", ascending=False)
    out.to_csv(OUT, index=False)
    # Keep a dated timeseries snapshot for future validation.
    if os.path.exists(TS):
        old = pd.read_csv(TS)
        out_all = pd.concat([old, out], ignore_index=True).drop_duplicates(["date","sector_code"], keep="last")
    else:
        out_all = out
    out_all.to_csv(TS, index=False)
    json.dump({
        "status":"SUCCESS",
        "latest_date":str(latest.date()),
        "mapped_tickers":int(px["ticker"].nunique()),
        "sector_count":int(out["sector_code"].nunique()),
        "warning":"Static sector mapping is not PIT-safe unless effective dates are supplied."
    }, open(STATUS,"w"), indent=2)

if __name__ == "__main__":
    main()
