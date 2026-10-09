from pathlib import Path
import json, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
FS=ROOT/"data"/"fundamental"/"financial_statements.csv"; EVIDENCE=ROOT/"data"/"fundamental"/"publication_evidence.csv"
OUT=ROOT/"data"/"fundamental"/"financial_statements_pit.csv"; AUDIT=ROOT/"data"/"fundamental"/"pit_reconciliation_audit.csv"; STATUS=ROOT/"data"/"fundamental"/"pit_reconciliation_status.json"
def read(p): return pd.read_csv(p,dtype=str,sep=None,engine="python",encoding="utf-8-sig").fillna("")
def norm(s): return pd.to_datetime(s,errors="coerce",utc=True).dt.strftime("%Y-%m-%d").fillna("")
if not FS.exists(): raise FileNotFoundError(FS)
if not EVIDENCE.exists(): raise FileNotFoundError(EVIDENCE)
fs,ev=read(FS),read(EVIDENCE)
for name,d in [("financial_statements",fs),("publication_evidence",ev)]:
    for c in ["ticker","period_end"]:
        if c not in d.columns: raise ValueError(f"{name} missing {c}")
if "publication_timestamp" not in ev.columns: raise ValueError("publication_evidence missing publication_timestamp")
fs["_tk"]=fs.ticker.str.strip().str.upper(); ev["_tk"]=ev.ticker.str.strip().str.upper()
fs["_pk"]=norm(fs.period_end); ev["_pk"]=norm(ev.period_end)
ev["_tsraw"]=ev.publication_timestamp.str.strip(); ev["_ts"]=pd.to_datetime(ev["_tsraw"],errors="coerce",utc=True); ev["_valid"]=ev["_ts"].notna()
ev["_date"]=ev.publication_date.str.strip() if "publication_date" in ev else ""; ev["_time"]=ev.publication_time.str.strip() if "publication_time" in ev else ""
ev=ev.sort_values(["_tk","_pk","_valid","_ts"],na_position="first").drop_duplicates(["_tk","_pk"],keep="last")
m=fs.merge(ev[["_tk","_pk","_tsraw","_ts","_date","_time"]],on=["_tk","_pk"],how="left")
m["publication_timestamp"]=m["_tsraw"].fillna(""); m["publication_date"]=m["_date"].fillna(""); m["publication_time"]=m["_time"].fillna("")
period_ok=pd.to_datetime(m.period_end,errors="coerce",utc=True).notna(); ts_ok=m["_ts"].notna(); m["pit_ready"]=period_ok & ts_ok
m=m.drop(columns=["_tk","_pk","_tsraw","_ts","_date","_time"])
cols=list(m.columns)
for c in ["publication_date","publication_time","publication_timestamp","pit_ready"]:
    if c in cols: cols.remove(c)
i=cols.index("period_end")+1
m=m[cols[:i]+["publication_date","publication_time","publication_timestamp","pit_ready"]+cols[i:]]
OUT.parent.mkdir(parents=True,exist_ok=True); m.to_csv(OUT,index=False)
a=m.groupby(["ticker","period_end"],dropna=False).agg(financial_rows=("ticker","size"),publication_date=("publication_date","first"),publication_time=("publication_time","first"),publication_timestamp=("publication_timestamp","first"),pit_ready=("pit_ready","all")).reset_index()
a["match_status"]=a.pit_ready.map(lambda x:"MATCHED" if bool(x) else "MISSING_OR_INVALID_TIMESTAMP"); a.to_csv(AUDIT,index=False)
pts=pd.to_datetime(m.publication_timestamp,errors="coerce",utc=True); raw=m.publication_timestamp.str.strip()
miss=int(raw.eq("").sum()); invalid=int((raw.ne("")&pts.isna()).sum())
status={"status":"PASS" if miss==0 and invalid==0 else "REVIEW","engine_changed":False,"financial_rows":len(m),"tickers":m.ticker.nunique(),"evidence_rows":len(ev),"pit_ready_rows":int(m.pit_ready.sum()),"pit_ready_tickers":m.loc[m.pit_ready,"ticker"].nunique(),"publication_timestamp_missing_rows":miss,"publication_timestamp_invalid_rows":invalid,"join_key":"ticker + period_end","ticker_only_fallback":False,"publication_date_fallback":False,"period_end_used_as_publication_date":False,"notes":["publication_timestamp is authoritative for PIT evidence readiness.","publication_date/publication_time are metadata only; no fallback.","Historical cutoff belongs downstream: publication_timestamp <= analysis_timestamp."]}
STATUS.write_text(json.dumps(status,indent=2),encoding="utf-8"); print(json.dumps(status,indent=2))
