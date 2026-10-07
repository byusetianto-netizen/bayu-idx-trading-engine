from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/fundamental/financial_statements.csv'
STATUS = ROOT / 'data/fundamental/acquisition_status.json'
REQ = ['ticker','statement','metric','value','period_start','period_end','period_type','is_cumulative','publication_date','report_date','source','source_url','document_id','version','confidence']

def main():
    s = json.loads(STATUS.read_text(encoding='utf-8')) if STATUS.exists() else {}
    df = pd.read_csv(RAW)
    s.update({'raw_records': int(len(df)), 'pit_safe_records': 0, 'engine_changed': False})
    missing = [c for c in REQ if c not in df.columns]
    if missing:
        s['status'] = 'IMPORT_SCHEMA_INVALID'
        s['notes'] = [f'Missing columns: {missing}']
        STATUS.write_text(json.dumps(s, indent=2), encoding='utf-8')
        raise SystemExit(1)
    if df.empty:
        s['status'] = 'FOUNDATION_READY_NO_DATA'
        s['notes'] = ['No financial observations loaded. This is intentional until an authoritative source is imported.']
    else:
        pub = pd.to_datetime(df['publication_date'], errors='coerce')
        ps = pd.to_datetime(df['period_start'], errors='coerce')
        pe = pd.to_datetime(df['period_end'], errors='coerce')
        pit = pub.notna() & ps.notna() & pe.notna() & (ps <= pe)
        s['pit_safe_records'] = int(pit.sum())
        s['invalid_period_rows'] = int((~pit).sum())
        s['status'] = 'IMPORT_VALIDATED' if pit.all() else 'IMPORT_REQUIRES_REVIEW'
    STATUS.write_text(json.dumps(s, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
