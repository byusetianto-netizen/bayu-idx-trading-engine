from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'data' / 'fundamental' / 'publication_input.csv'
OUTPUT = ROOT / 'data' / 'fundamental' / 'publication_evidence.csv'
STATUS = ROOT / 'data' / 'fundamental' / 'publication_manual_status.json'

required = ['ticker','period_end','publication_date','publication_time','source_url','notes']
df = pd.read_csv(INPUT, dtype=str).fillna('')
for c in required:
    if c not in df.columns:
        raise ValueError(f'Missing required column: {c}')

def make_timestamp(r):
    d = r['publication_date'].strip()
    if not d:
        return ''
    t = r['publication_time'].strip() or '23:59:59'
    return f'{d} {t}+07:00'

rows=[]
for _, r in df.iterrows():
    ticker=r['ticker'].strip().upper()
    period=r['period_end'].strip()
    pubdate=r['publication_date'].strip()
    if not ticker or not period:
        continue
    ts=make_timestamp(r)
    confidence='HIGH' if r['publication_time'].strip() else ('MEDIUM' if pubdate else 'UNKNOWN')
    status='VERIFIED' if pubdate else 'MISSING'
    rows.append({
        'ticker':ticker,
        'period_end':period,
        'publication_date':pubdate,
        'publication_timestamp':ts,
        'source':'MANUAL_IDX_EVIDENCE',
        'source_url':r['source_url'].strip(),
        'publication_confidence':confidence,
        'source_status':status,
        'pit_rule_ready':'true' if pubdate else 'false',
        'notes':r['notes'].strip() or ('Date-only entry: timestamp conservatively set to 23:59:59 WIB.' if pubdate and not r['publication_time'].strip() else '')
    })
out=pd.DataFrame(rows)
if not out.empty:
    out=out.drop_duplicates(['ticker','period_end'], keep='last').sort_values(['ticker','period_end'])
out.to_csv(OUTPUT,index=False)
status={
    'status':'BUILT',
    'rows':int(len(out)),
    'verified_rows':int((out['source_status']=='VERIFIED').sum()) if not out.empty else 0,
    'missing_rows':int((out['source_status']=='MISSING').sum()) if not out.empty else 0,
    'date_only_rows':int((out['publication_confidence']=='MEDIUM').sum()) if not out.empty else 0,
    'engine_changed':False,
    'notes':['Edit only data/fundamental/publication_input.csv. One row per ticker + period_end.','Exact time is optional. If omitted, publication is treated conservatively as 23:59:59 WIB.','This file is evidence metadata only; it does not invent financial values.']
}
STATUS.write_text(json.dumps(status,indent=2),encoding='utf-8')
print(json.dumps(status,indent=2))
