import argparse, json
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',default='data/fundamental/financial_statements.csv'); ap.add_argument('--analysis-date',default=None); ap.add_argument('--output-dir',default='data/fundamental'); args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.input)
    required=['ticker','metric_key','value','period_end','publication_date','source','document_id','confidence']
    missing=[c for c in required if c not in df.columns]
    errors=[]
    if missing: errors.append('missing_columns:'+','.join(missing))
    if not df.empty:
        df['publication_date']=pd.to_datetime(df['publication_date'],errors='coerce')
        df['period_end']=pd.to_datetime(df['period_end'],errors='coerce')
        if df['publication_date'].isna().any(): errors.append('invalid_publication_date')
        if df['period_end'].isna().any(): errors.append('invalid_period_end')
        if (df['publication_date'] < df['period_end']).any(): errors.append('publication_before_period_end_detected_review')
        if df['document_id'].isna().any() or (df['document_id'].astype(str).str.len()<32).any(): errors.append('missing_or_short_document_id')
        if df['ticker'].nunique()!=1: errors.append('multi_ticker_sample_requires_batch_validation')
        if args.analysis_date:
            ad=pd.Timestamp(args.analysis_date)
            allowed=(df['publication_date']<=ad)
            pit_count=int(allowed.sum()); blocked_count=int((~allowed).sum())
        else:
            pit_count=None; blocked_count=None
    else: pit_count=blocked_count=0
    status='PIT_VALIDATION_PASS' if not errors else 'PIT_VALIDATION_REVIEW'
    result={'status':status,'engine_changed':False,'rows':int(len(df)),'tickers':sorted(df.ticker.dropna().unique().tolist()) if not df.empty else [],'publication_dates':sorted(df.publication_date.dt.strftime('%Y-%m-%d').dropna().unique().tolist()) if not df.empty else [],'analysis_date':args.analysis_date,'pit_allowed_rows':pit_count,'pit_blocked_rows':blocked_count,'errors':errors}
    (out/'pit_validation_status.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    raise SystemExit(0 if status=='PIT_VALIDATION_PASS' else 1)
if __name__=='__main__': main()
