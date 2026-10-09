#!/usr/bin/env python3
"""V2.2 valuation validator — timestamp-authoritative PIT integration checks."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path('.')
V=ROOT/'data/valuation'
OUT=V/'valuation_snapshot.csv'
STATUS=V/'valuation_status.json'
FS_PIT=ROOT/'data/fundamental/financial_statements_pit.csv'

def read(path):
    if not path.exists(): return pd.DataFrame()
    try: return pd.read_csv(path)
    except Exception: return pd.DataFrame()

def norm_tickers(df):
    if df.empty or 'ticker' not in df.columns: return set()
    return set(df['ticker'].astype(str).str.upper().str.strip().dropna())

def strict_bool(s):
    return s.astype(str).str.strip().str.lower().isin({'true','1','yes','y'})

def main():
    checks={'snapshot_present':OUT.exists(),'status_present':STATUS.exists()}
    if not OUT.exists() or not STATUS.exists():
        result={'status':'FAIL','engine_changed':False,'checks':checks,
                'notes':['Required valuation outputs are missing.']}
    else:
        d=read(OUT); fs=read(FS_PIT)
        try: engine_status=json.loads(STATUS.read_text(encoding='utf-8'))
        except Exception: engine_status={}
        analysis_ts=pd.to_datetime(engine_status.get('analysis_timestamp'),errors='coerce',utc=True,format='mixed')

        fs_schema_ok=(not fs.empty and {'ticker','publication_timestamp','pit_ready'}.issubset(fs.columns))
        pit_expected=set(); missing_ts=invalid_ts=future_ts=0
        if fs_schema_ok and pd.notna(analysis_ts):
            fs=fs.copy()
            fs['ticker']=fs['ticker'].astype(str).str.upper().str.strip()
            raw_ts=fs['publication_timestamp'].astype(str).str.strip()
            parsed=pd.to_datetime(fs['publication_timestamp'],errors='coerce',utc=True,format='mixed')
            ready=strict_bool(fs['pit_ready'])
            missing_mask=raw_ts.eq('')|raw_ts.str.lower().isin({'nan','none','nat'})
            invalid_mask=(~missing_mask)&parsed.isna()
            future_mask=parsed.notna()&(parsed>analysis_ts)
            eligible=fs[ready & parsed.notna() & (parsed<=analysis_ts)].copy()
            pit_expected=set(eligible['ticker'])
            missing_ts=int(missing_mask.sum()); invalid_ts=int(invalid_mask.sum()); future_ts=int(future_mask.sum())

        snapshot_verified=set()
        if {'pit_current_status','ticker'}.issubset(d.columns):
            snapshot_verified=set(d.loc[d.pit_current_status.astype(str).eq('PIT_VERIFIED'),'ticker']
                                  .astype(str).str.upper().str.strip())
        pit_overlap=pit_expected & norm_tickers(d)

        snap_ts_ok=False; snap_future=0
        if 'analysis_timestamp' in d.columns and pd.notna(analysis_ts):
            st=pd.to_datetime(d['analysis_timestamp'],errors='coerce',utc=True,format='mixed')
            snap_ts_ok=bool(st.notna().all() and (st==analysis_ts).all())
        if {'pit_publication_timestamp','pit_current_status'}.issubset(d.columns) and pd.notna(analysis_ts):
            pt=pd.to_datetime(d['pit_publication_timestamp'],errors='coerce',utc=True,format='mixed')
            verified=d['pit_current_status'].astype(str).eq('PIT_VERIFIED')
            snap_future=int((verified & (pt.isna() | (pt>analysis_ts))).sum())

        checks.update({
            'rows':len(d),
            'duplicate_ticker':int(d.ticker.duplicated().sum()) if 'ticker' in d else -1,
            'negative_eps_per_guard':int(((pd.to_numeric(d.eps,errors='coerce')<=0)&d.per.notna()).sum()) if {'eps','per'}.issubset(d.columns) else -1,
            'negative_bvps_pbv_guard':int(((pd.to_numeric(d.bvps,errors='coerce')<=0)&d.pbv.notna()).sum()) if {'bvps','pbv'}.issubset(d.columns) else -1,
            'canonical_pit_schema_ok':bool(fs_schema_ok),
            'analysis_timestamp_valid':bool(pd.notna(analysis_ts)),
            'snapshot_analysis_timestamp_consistent':bool(snap_ts_ok),
            'pit_source_rows':int(len(fs)),
            'publication_timestamp_missing_rows':missing_ts,
            'publication_timestamp_invalid_rows':invalid_ts,
            'future_publication_rows':future_ts,
            'pit_source_eligible_tickers':int(len(pit_expected)),
            'pit_source_overlap_with_snapshot':int(len(pit_overlap)),
            'pit_verified_rows':int((d.pit_current_status=='PIT_VERIFIED').sum()) if 'pit_current_status' in d else -1,
            'pit_verified_tickers':sorted(snapshot_verified),
            'pit_expected_in_snapshot':sorted(pit_overlap),
            'pit_integration_mismatch':sorted(snapshot_verified.symmetric_difference(pit_overlap)),
            'verified_snapshot_timestamp_violations':snap_future,
            'pit_unknown_rows':int((d.pit_current_status!='PIT_VERIFIED').sum()) if 'pit_current_status' in d else -1,
            'historical_verified_rows':int((d.historical_status=='VERIFIED_PIT_RANGE').sum()) if 'historical_status' in d else 0,
            'peer_available_rows':int((d.peer_status=='PEER_SET_AVAILABLE').sum()) if 'peer_status' in d else 0,
            'classification_present':int(d.classification.notna().sum()) if 'classification' in d else 0,
            'publication_date_fallback':bool(engine_status.get('publication_date_fallback',True)),
            'raw_financial_fallback':bool(engine_status.get('raw_financial_fallback',True)),
        })
        ok=(checks['duplicate_ticker']==0
            and checks['negative_eps_per_guard']==0
            and checks['negative_bvps_pbv_guard']==0
            and checks['classification_present']==checks['rows']
            and checks['canonical_pit_schema_ok']
            and checks['analysis_timestamp_valid']
            and checks['snapshot_analysis_timestamp_consistent']
            and len(checks['pit_integration_mismatch'])==0
            and checks['verified_snapshot_timestamp_violations']==0
            and checks['publication_timestamp_missing_rows']==0
            and checks['publication_timestamp_invalid_rows']==0
            and not checks['publication_date_fallback']
            and not checks['raw_financial_fallback'])
        result={'status':'PASS' if ok else 'REVIEW','engine_changed':False,'checks':checks,
                'notes':['PIT integration is validated with publication_timestamp <= analysis_timestamp and pit_ready=true.',
                         'Missing/invalid publication_timestamp fails closed; publication_date is never a fallback.',
                         'Canonical financial_statements_pit.csv is required; raw financial fallback is prohibited.',
                         'PASS validates structural/PIT integrity only; it does not prove valuation accuracy or investment performance.']}
    (V/'valuation_validation_status.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    print(json.dumps(result,indent=2,default=str))

if __name__=='__main__': main()
