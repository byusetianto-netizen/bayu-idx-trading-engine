#!/usr/bin/env python3
"""V2.2 valuation validator with explicit PIT integration checks."""
from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT=Path('.')
V=ROOT/'data/valuation'
OUT=V/'valuation_snapshot.csv'
FS_PIT=ROOT/'data/fundamental/financial_statements_pit.csv'
PRICE=ROOT/'data/idx_stock_prices.csv'


def read(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def norm_tickers(df):
    if df.empty or 'ticker' not in df.columns:
        return set()
    return set(df['ticker'].astype(str).str.upper().str.strip().dropna())


def main():
    checks={'snapshot_present':OUT.exists()}
    if not OUT.exists():
        result={'status':'FAIL','engine_changed':False,'checks':checks}
    else:
        d=read(OUT)
        fs=read(FS_PIT)
        prices=read(PRICE)

        pit_expected=set()
        pit_overlap=set()
        if not fs.empty and {'ticker','publication_date'}.issubset(fs.columns):
            fs=fs.copy()
            fs['ticker']=fs['ticker'].astype(str).str.upper().str.strip()
            fs['publication_date']=pd.to_datetime(fs['publication_date'],errors='coerce')
            if 'pit_ready' in fs.columns:
                ready=fs['pit_ready'].astype(str).str.lower().isin({'true','1','yes','y'})
            else:
                ready=pd.Series(True,index=fs.index)
            # Snapshot analysis date is authoritative for PIT availability.
            analysis_date=pd.to_datetime(d['analysis_date'],errors='coerce').max() if 'analysis_date' in d else pd.NaT
            if pd.notna(analysis_date):
                eligible=fs[ready & fs['publication_date'].notna() & (fs['publication_date']<=analysis_date)]
            else:
                eligible=fs[ready & fs['publication_date'].notna()]
            pit_expected=set(eligible['ticker'])
            pit_overlap=pit_expected & norm_tickers(d)

        snapshot_verified=set()
        if 'pit_current_status' in d and 'ticker' in d:
            snapshot_verified=set(
                d.loc[d['pit_current_status'].astype(str).eq('PIT_VERIFIED'),'ticker']
                .astype(str).str.upper().str.strip()
            )

        checks.update({
            'rows':len(d),
            'duplicate_ticker':int(d.ticker.duplicated().sum()) if 'ticker' in d else -1,
            'negative_eps_per_guard':int(((pd.to_numeric(d.eps,errors='coerce')<=0)&d.per.notna()).sum()) if {'eps','per'}.issubset(d.columns) else -1,
            'negative_bvps_pbv_guard':int(((pd.to_numeric(d.bvps,errors='coerce')<=0)&d.pbv.notna()).sum()) if {'bvps','pbv'}.issubset(d.columns) else -1,
            'pit_source_rows':int(len(fs)),
            'pit_source_eligible_tickers':int(len(pit_expected)),
            'pit_source_overlap_with_snapshot':int(len(pit_overlap)),
            'pit_verified_rows':int((d.pit_current_status=='PIT_VERIFIED').sum()) if 'pit_current_status' in d else -1,
            'pit_verified_tickers':sorted(snapshot_verified),
            'pit_expected_in_snapshot':sorted(pit_overlap),
            'pit_integration_mismatch':sorted(snapshot_verified.symmetric_difference(pit_overlap)),
            'pit_unknown_rows':int((d.pit_current_status!='PIT_VERIFIED').sum()) if 'pit_current_status' in d else -1,
            'historical_verified_rows':int((d.historical_status=='VERIFIED_PIT_RANGE').sum()) if 'historical_status' in d else 0,
            'peer_available_rows':int((d.peer_status=='PEER_SET_AVAILABLE').sum()) if 'peer_status' in d else 0,
            'classification_present':int(d.classification.notna().sum()) if 'classification' in d else 0,
        })

        ok=(
            checks['duplicate_ticker']==0
            and checks['negative_eps_per_guard']==0
            and checks['negative_bvps_pbv_guard']==0
            and checks['classification_present']==checks['rows']
            and len(checks['pit_integration_mismatch'])==0
        )
        result={
            'status':'PASS' if ok else 'REVIEW',
            'engine_changed':False,
            'checks':checks,
            'notes':[
                'PIT integration is validated against financial_statements_pit.csv using publication_date <= analysis_date.',
                'EPS/BVPS are optional for PIT verification; if absent, PER/PBV remain UNKNOWN.',
                'Historical valuation still requires sufficient multi-period PIT evidence.',
                'PASS validates structural/PIT integrity only; it does not prove valuation accuracy or investment performance.'
            ]
        }
    (V/'valuation_validation_status.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    print(json.dumps(result,indent=2,default=str))


if __name__=='__main__':
    main()
