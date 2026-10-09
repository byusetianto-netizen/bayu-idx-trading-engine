from pathlib import Path
import json, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; INPUT=ROOT/"data"/"fundamental"/"financial_statements_pit.csv"; OUT=ROOT/"data"/"fundamental"/"pit_reconciliation_validation.json"
df=pd.read_csv(INPUT,dtype=str).fillna("")
req={"ticker","period_end","publication_timestamp","pit_ready"}; missing=sorted(req-set(df.columns))
if missing: raise ValueError(f"Missing required columns: {missing}")
period=pd.to_datetime(df.period_end,errors="coerce",utc=True); ts=pd.to_datetime(df.publication_timestamp,errors="coerce",utc=True); raw=df.publication_timestamp.str.strip()
tsmiss=int(raw.eq("").sum()); tsbad=int((raw.ne("")&ts.isna()).sum()); pmiss=int(period.isna().sum())
dups=int(df.duplicated(["ticker","period_end","metric"],keep=False).sum()) if "metric" in df else 0
expected=ts.notna()&period.notna(); actual=df.pit_ready.str.strip().str.lower().isin(["true","1","yes","y"]); mismatch=int((expected!=actual).sum())
status="PASS" if tsmiss==0 and tsbad==0 and pmiss==0 and dups==0 and mismatch==0 else "REVIEW"
r={"status":status,"engine_changed":False,"financial_rows":len(df),"tickers":df.ticker.nunique(),"pit_ready_rows":int(expected.sum()),"pit_ready_tickers":df.loc[expected,"ticker"].nunique(),"publication_timestamp_missing":tsmiss,"publication_timestamp_invalid":tsbad,"period_end_missing":pmiss,"duplicate_key_rows":dups,"pit_ready_mismatch_rows":mismatch,"publication_after_period_end":int((ts.notna()&period.notna()&(ts>period)).sum()),"publication_after_period_end_is_expected":True,"timezone_normalization":"UTC-aware parsing/comparison","publication_date_fallback":False,"notes":["publication_timestamp is authoritative.","publication_date is metadata only; never a fallback.","Downstream historical rule: publication_timestamp <= analysis_timestamp."]}
OUT.write_text(json.dumps(r,indent=2),encoding="utf-8"); print(json.dumps(r,indent=2))
