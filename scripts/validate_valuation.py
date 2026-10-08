#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT=Path('.'); V=ROOT/'data/valuation'; OUT=V/'valuation_snapshot.csv'

def main():
    checks={'snapshot_present':OUT.exists()}
    if not OUT.exists():
        result={'status':'FAIL','engine_changed':False,'checks':checks}
    else:
        d=pd.read_csv(OUT)
        checks.update({
            'rows':len(d),
            'duplicate_ticker':int(d.ticker.duplicated().sum()) if 'ticker' in d else -1,
            'negative_eps_per_guard':int(((pd.to_numeric(d.eps,errors='coerce')<=0)&d.per.notna()).sum()) if {'eps','per'}.issubset(d.columns) else -1,
            'negative_bvps_pbv_guard':int(((pd.to_numeric(d.bvps,errors='coerce')<=0)&d.pbv.notna()).sum()) if {'bvps','pbv'}.issubset(d.columns) else -1,
            'pit_unknown_rows':int((d.pit_current_status!='PIT_VERIFIED').sum()) if 'pit_current_status' in d else -1,
            'historical_verified_rows':int((d.historical_status=='VERIFIED_PIT_RANGE').sum()) if 'historical_status' in d else 0,
            'peer_available_rows':int((d.peer_status=='PEER_SET_AVAILABLE').sum()) if 'peer_status' in d else 0,
            'classification_present':int(d.classification.notna().sum()) if 'classification' in d else 0,
        })
        ok=(checks['duplicate_ticker']==0 and checks['negative_eps_per_guard']==0 and checks['negative_bvps_pbv_guard']==0 and checks['classification_present']==checks['rows'])
        result={'status':'PASS' if ok else 'REVIEW','engine_changed':False,'checks':checks,
                'notes':['PASS validates structural safety only. It does not prove valuation accuracy or investment performance.',
                         'PIT UNKNOWN rows must remain excluded from historical valuation claims.']}
    (V/'valuation_validation_status.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
