#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import pandas as pd

HARD=['pit_safety','accounting_integrity','comparable_period','critical_source_coverage','critical_metric_coverage','currency_unit_integrity','source_reliability']

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='data/fundamental/fundamental_reliability.csv')
    ap.add_argument('--source',default='data/fundamental/financial_statements_pit.csv')
    ap.add_argument('--analysis-date',required=True)
    ap.add_argument('--analysis-time',default='23:59:59')
    ap.add_argument('--output',default='data/fundamental/fundamental_reliability_validation.json')
    args=ap.parse_args()
    r=pd.read_csv(args.input); s=pd.read_csv(args.source)
    analysis_ts=pd.Timestamp(f"{args.analysis_date} {args.analysis_time}")
    s['publication_date']=pd.to_datetime(s.publication_date,errors='coerce')
    expected=sorted(s[s.publication_date.notna()&(s.publication_date<=pd.Timestamp(args.analysis_date))].ticker.astype(str).unique())
    actual=sorted(r.ticker.astype(str).unique()); d=[]
    if expected!=actual:d.append(f'ticker coverage mismatch expected={expected} actual={actual}')
    if r.ticker.duplicated().any():d.append('duplicate ticker rows')
    if not set(r.reliability_status).issubset({'PASS','DEGRADED','BLOCKED'}):d.append('invalid reliability_status')
    elig=r.eligible_for_fundamental_scoring.astype(str).str.lower().isin(['true','1'])
    if (r.reliability_status.eq('BLOCKED')&elig).any():d.append('BLOCKED ticker marked eligible')
    for c in HARD:
        if c not in r.columns:d.append(f'missing gate column {c}')
    if not d:
        nb=r[~r.reliability_status.eq('BLOCKED')]
        for c in HARD:
            if not nb[c].eq('PASS').all():d.append(f'non-blocked row has {c} != PASS')
        # Fail closed specifically on unknown/review PIT.
        if (r.pit_safety.ne('PASS') & elig).any(): d.append('non-PASS PIT ticker marked eligible')
    st={'status':'PASS' if not d else 'FAIL','analysis_timestamp':analysis_ts.isoformat(),'source_tickers':expected,'reliability_tickers':actual,'rows':len(r),'eligible':int(elig.sum()),'blocked':int(r.reliability_status.eq('BLOCKED').sum()),'diagnostics':d}
    Path(args.output).write_text(json.dumps(st,indent=2),encoding='utf-8'); print(json.dumps(st,indent=2))
    raise SystemExit(0 if not d else 1)
if __name__=='__main__': main()
