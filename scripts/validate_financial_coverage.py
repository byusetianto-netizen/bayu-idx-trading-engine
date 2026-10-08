
import os, json, pandas as pd
from datetime import datetime, timezone
FS="data/fundamental/financial_statements.csv"
PE="data/fundamental/publication_evidence.csv"
OUT="data/fundamental/coverage_validation_status.json"
fs=pd.read_csv(FS) if os.path.exists(FS) else pd.DataFrame()
pe=pd.read_csv(PE) if os.path.exists(PE) else pd.DataFrame()
checks={}
if not fs.empty:
    fs["ticker"]=fs["ticker"].astype(str).str.upper()
    fs["publication_date"]=pd.to_datetime(fs.get("publication_date"),errors="coerce")
    fs["period_end"]=pd.to_datetime(fs.get("period_end"),errors="coerce")
    checks["financial_statement_rows"]=len(fs)
    checks["duplicate_key_rows"]=int(fs.duplicated(["ticker","statement","metric","period_end","publication_date"]).sum())
    checks["missing_publication_date"]=int(fs["publication_date"].isna().sum())
    checks["missing_period_end"]=int(fs["period_end"].isna().sum())
else:
    checks={"financial_statement_rows":0,"duplicate_key_rows":0,"missing_publication_date":0,"missing_period_end":0}
if not pe.empty:
    pe["publication_date"]=pd.to_datetime(pe.get("publication_date"),errors="coerce")
    checks["publication_evidence_rows"]=len(pe)
    checks["evidence_missing_publication"]=int(pe["publication_date"].isna().sum())
    checks["verified_rows"]=int((pe.get("source_status","")=="VERIFIED").sum())
else:
    checks["publication_evidence_rows"]=0
    checks["evidence_missing_publication"]=0
    checks["verified_rows"]=0

status="PASS" if checks["duplicate_key_rows"]==0 else "REVIEW"
payload={
 "status":status,"engine_changed":False,"checks":checks,
 "generated_at":datetime.now(timezone.utc).isoformat(),
 "notes":[
  "PASS means structural validation only.",
  "Rows without publication_date are not PIT-ready.",
  "No financial values are inferred or backfilled."
 ]
}
json.dump(payload,open(OUT,"w"),indent=2,default=str)
print(json.dumps(payload,indent=2,default=str))
