from pathlib import Path
import pandas as pd
import json

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'data/fundamental/publication_input.csv'
EVID=ROOT/'data/fundamental/publication_evidence.csv'
OUT=ROOT/'data/fundamental/publication_manual_validation.json'

df=pd.read_csv(INPUT,dtype=str).fillna('')
checks=[]
for c in ['ticker','period_end','publication_date']:
    checks.append({'check':f'missing_{c}','count':int((df[c].str.strip()=='').sum())})
if not df.empty:
    checks.append({'check':'duplicate_ticker_period','count':int(df.duplicated(['ticker','period_end']).sum())})
else:
    checks.append({'check':'duplicate_ticker_period','count':0})

status='PASS' if all(x['count']==0 for x in checks) else 'REVIEW'
res={'status':status,'rows':int(len(df)),'checks':checks,'evidence_rows':int(len(pd.read_csv(EVID))) if EVID.exists() else 0,'notes':['publication_date is mandatory; publication_time is optional.']}
OUT.write_text(json.dumps(res,indent=2),encoding='utf-8')
print(json.dumps(res,indent=2))
