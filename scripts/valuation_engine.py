import json, math
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(".")
FUND=ROOT/"data/fundamental"
VAL=ROOT/"data/valuation"; VAL.mkdir(parents=True, exist_ok=True)
PRICE=ROOT/"data/idx_stock_prices.csv"
METRICS=FUND/"fundamental_metrics.csv"
FS=FUND/"financial_statements.csv"
PIT=FUND/"pit_validation_status.json"
SECTOR=ROOT/"data/sector/sector_map.csv"

def col(df, names, default=np.nan):
    for n in names:
        if n in df.columns: return n
    return None

def safe_num(x):
    try:
        if pd.isna(x): return np.nan
        return float(x)
    except: return np.nan

def load_prices():
    if not PRICE.exists(): return pd.DataFrame()
    p=pd.read_csv(PRICE)
    p.columns=[str(c).strip().lower() for c in p.columns]
    if "ticker" not in p or "date" not in p or "close" not in p: return pd.DataFrame()
    p["date"]=pd.to_datetime(p["date"], errors="coerce")
    p["close"]=pd.to_numeric(p["close"], errors="coerce")
    return p.dropna(subset=["ticker","date","close"])

def load_metrics():
    if not METRICS.exists(): return pd.DataFrame()
    m=pd.read_csv(METRICS)
    m.columns=[str(c).strip().lower() for c in m.columns]
    return m

def load_sector():
    if not SECTOR.exists(): return pd.DataFrame()
    s=pd.read_csv(SECTOR)
    s.columns=[str(c).strip().lower() for c in s.columns]
    return s

def latest_pit_metrics(m):
    if m.empty: return pd.DataFrame()
    if "publication_date" in m.columns:
        m["publication_date"]=pd.to_datetime(m["publication_date"], errors="coerce")
    if "period_end" in m.columns:
        m["period_end"]=pd.to_datetime(m["period_end"], errors="coerce")
    analysis_date = pd.Timestamp.now(tz=None).normalize()
    if "publication_date" in m.columns:
        m=m[(m["publication_date"].isna()) | (m["publication_date"]<=analysis_date)]
    # Prefer latest period per ticker, but retain metric rows.
    return m

def metric_value(m, ticker, metric_names):
    if m.empty or "ticker" not in m.columns: return np.nan
    x=m[m["ticker"].astype(str).str.upper()==ticker]
    if "metric" not in x.columns or "value" not in x.columns: return np.nan
    for name in metric_names:
        z=x[x["metric"].astype(str).str.lower()==name.lower()]
        if not z.empty:
            return safe_num(z.sort_values("period_end").iloc[-1]["value"] if "period_end" in z.columns else z.iloc[-1]["value"])
    return np.nan

def main():
    m=load_metrics(); p=load_prices(); sec=load_sector()
    if m.empty or p.empty:
        status={"status":"INSUFFICIENT_DATA","engine_changed":False,"notes":["Need fundamental_metrics.csv and price data."]}
        (VAL/"valuation_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
        return
    m=latest_pit_metrics(m)
    latest_date=p["date"].max()
    px=p[p["date"]==latest_date].drop_duplicates("ticker")
    rows=[]
    for _,r in px.iterrows():
        t=str(r["ticker"]).upper(); price=safe_num(r["close"])
        rev_growth=metric_value(m,t,["revenue_growth","revenue_growth_pct"])
        ni_growth=metric_value(m,t,["net_profit_growth","net_income_growth","net_profit_growth_pct"])
        roe=metric_value(m,t,["roe","roe_annualized_estimate","roe_annualized"])
        npm=metric_value(m,t,["net_margin","npm"])
        bvps=metric_value(m,t,["book_value_per_share","bvps","book value per share"])
        eps=metric_value(m,t,["eps","eps_ttm"])
        # Metric engine may store only aggregate values; infer P/E/PBV from EPS/BVPS if available.
        per = price/eps if np.isfinite(eps) and eps>0 else np.nan
        pbv = price/bvps if np.isfinite(bvps) and bvps>0 else np.nan
        ey = 1/per if np.isfinite(per) and per>0 else np.nan
        # GARP diagnostic, not a PEG buy rule.
        growth=ni_growth
        peg = per/growth if np.isfinite(per) and np.isfinite(growth) and growth>0 else np.nan
        flags=[]
        if np.isfinite(eps) and eps<=0: flags.append("NEGATIVE_EPS")
        if np.isfinite(bvps) and bvps<=0: flags.append("NEGATIVE_BOOK_VALUE")
        if np.isfinite(roe) and roe<0: flags.append("NEGATIVE_ROE")
        if np.isfinite(ni_growth) and ni_growth<0: flags.append("EARNINGS_DECLINE")
        if np.isfinite(npm) and npm<0: flags.append("NEGATIVE_MARGIN")
        trap=False
        if ("NEGATIVE_EPS" in flags or "NEGATIVE_BOOK_VALUE" in flags) and (np.isfinite(rev_growth) and rev_growth<0 or np.isfinite(ni_growth) and ni_growth<0):
            trap=True; flags.append("VALUE_TRAP_RISK")
        # Classification is deliberately conservative.
        if not np.isfinite(per) and not np.isfinite(pbv):
            cls="INSUFFICIENT_DATA"
        elif trap:
            cls="VALUE_TRAP_RISK"
        elif np.isfinite(peg) and peg<=1.0 and growth>0:
            cls="GARP_CANDIDATE"
        elif np.isfinite(per) and per<=10:
            cls="LOW_MULTIPLE"
        elif np.isfinite(per) and per>=25:
            cls="EXPENSIVE_MULTIPLE"
        else:
            cls="FAIR_OR_CONTEXT_DEPENDENT"
        sector=""
        if not sec.empty and "ticker" in sec.columns:
            ss=sec[sec["ticker"].astype(str).str.upper()==t]
            if not ss.empty:
                sector=str(ss.iloc[-1].get("sector_name",""))
        rows.append({
            "analysis_date":latest_date.date().isoformat(),"ticker":t,"close":price,
            "eps":eps,"bvps":bvps,"per":per,"pbv":pbv,"earnings_yield":ey,
            "revenue_growth_pct":rev_growth,"net_profit_growth_pct":ni_growth,
            "roe_pct":roe,"net_margin_pct":npm,"peg_diagnostic":peg,
            "sector":sector,"classification":cls,"value_trap_risk":trap,
            "flags":";".join(flags),
            "data_quality":"CURRENT_SNAPSHOT_ONLY"
        })
    out=pd.DataFrame(rows)
    out.to_csv(VAL/"valuation_snapshot.csv",index=False)
    status={
        "status":"BUILT",
        "engine_changed":False,
        "analysis_date":latest_date.date().isoformat(),
        "tickers":int(len(out)),
        "notes":[
            "Valuation is evidence/classification only; no trade decision.",
            "PER is invalid for non-positive EPS; PBV is invalid for non-positive book value.",
            "Historical valuation bands and peer-relative ranks require PIT historical snapshots and comparable peer coverage; they are not inferred from current data.",
            "GARP is a diagnostic, not a guaranteed return forecast."
        ]
    }
    (VAL/"valuation_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
if __name__=="__main__": main()
