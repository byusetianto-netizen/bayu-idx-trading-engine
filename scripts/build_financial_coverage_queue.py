
import os, json, pandas as pd
from datetime import datetime, timezone

ROOT="."
UNIVERSE="data/research_universe.csv"
FS="data/fundamental/financial_statements.csv"
PE="data/fundamental/publication_evidence.csv"
OUT="data/fundamental/acquisition_queue.csv"
STATUS="data/fundamental/coverage_expansion_status.json"

def read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

u=read_csv(UNIVERSE)
if "ticker" not in u.columns:
    # fallback to price data
    p=read_csv("data/idx_stock_prices.csv")
    u=pd.DataFrame({"ticker": sorted(p["ticker"].dropna().astype(str).str.upper().unique())}) if "ticker" in p else pd.DataFrame(columns=["ticker"])
u["ticker"]=u["ticker"].astype(str).str.upper().str.strip()
u=u.drop_duplicates("ticker")

fs=read_csv(FS)
pe=read_csv(PE)
if not fs.empty:
    fs["ticker"]=fs["ticker"].astype(str).str.upper().str.strip()
    fs["publication_date"]=pd.to_datetime(fs.get("publication_date"), errors="coerce")
    fs["period_end"]=pd.to_datetime(fs.get("period_end"), errors="coerce")
    fs_pit=fs[fs["publication_date"].notna()]
    periods=fs_pit.groupby("ticker")["period_end"].nunique()
    latest_pub=fs_pit.groupby("ticker")["publication_date"].max()
    metric_count=fs_pit.groupby("ticker")["metric"].nunique()
else:
    periods=pd.Series(dtype=float); latest_pub=pd.Series(dtype="datetime64[ns]"); metric_count=pd.Series(dtype=float)

if not pe.empty:
    pe["ticker"]=pe["ticker"].astype(str).str.upper().str.strip()
    pe["publication_date"]=pd.to_datetime(pe.get("publication_date"), errors="coerce")
    pit_verified=set(pe.loc[(pe["publication_date"].notna()) & (pe.get("source_status","")=="VERIFIED"),"ticker"])
else:
    pit_verified=set()

rows=[]
for _,r in u.iterrows():
    t=r["ticker"]
    n=int(periods.get(t,0))
    q="HIGH" if n>=2 and t in pit_verified else ("MEDIUM" if n>=1 else "HIGH")
    if n==0:
        priority="P0"
        reason="No authoritative PIT financial statement coverage"
    elif n==1:
        priority="P1"
        reason="Only one PIT financial period; need another period for growth/history"
    elif n>=2:
        priority="P2"
        reason="Coverage exists; expand older periods and verify core metrics"
    rows.append({
        "ticker":t,
        "priority":priority,
        "reason":reason,
        "pit_periods":n,
        "latest_publication_date": latest_pub.get(t, pd.NaT),
        "metric_count":int(metric_count.get(t,0)),
        "pit_verified": t in pit_verified,
        "suggested_import":"Place official IDX FinancialStatement XLSX + publication evidence in data/fundamental/inbox/",
    })
out=pd.DataFrame(rows)
rank={"P0":0,"P1":1,"P2":2}
if not out.empty:
    out["_rank"]=out["priority"].map(rank)
    out=out.sort_values(["_rank","pit_periods","ticker"]).drop(columns="_rank")
out.to_csv(OUT,index=False)

status={
 "status":"COVERAGE_QUEUE_BUILT",
 "engine_changed":False,
 "universe_tickers":int(len(u)),
 "pit_covered_tickers":int(sum(out["pit_verified"])) if not out.empty else 0,
 "tickers_with_0_periods":int((out["pit_periods"]==0).sum()) if not out.empty else 0,
 "tickers_with_1_period":int((out["pit_periods"]==1).sum()) if not out.empty else 0,
 "tickers_with_2plus_periods":int((out["pit_periods"]>=2).sum()) if not out.empty else 0,
 "generated_at":datetime.now(timezone.utc).isoformat(),
 "notes":[
  "Queue does not fabricate financial data.",
  "Period_end is accounting period, not availability date.",
  "P0/P1 are the main coverage expansion targets.",
  "Official IDX Financial Data & Ratio may be used as secondary cross-check, not as a replacement for detailed statements."
 ]
}
json.dump(status,open(STATUS,"w"),indent=2,default=str)
print(json.dumps(status,indent=2,default=str))
