#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--metrics',default='data/fundamental/fundamental_metrics.csv'); ap.add_argument('--assessment',default='data/fundamental/fundamental_assessment.csv'); ap.add_argument('--analysis-date',default='2026-09-01'); ap.add_argument('--output',default='data/fundamental/fundamental_metrics_validation.json'); args=ap.parse_args()
    m=pd.read_csv(args.metrics); a=pd.read_csv(args.assessment)
    ad=pd.Timestamp(args.analysis_date)
    m['period_end']=pd.to_datetime(m.period_end,errors='coerce'); m['source_document_id']=m.source_document_id.astype(str)
    # Metrics inherit PIT from assessment/document lineage; verify assessment publication date.
    a['publication_date']=pd.to_datetime(a.publication_date,errors='coerce')
    bad=int((a.publication_date>ad).sum())
    expected={'revenue_growth','gross_margin','net_margin','roa_period','roe_period','debt_to_equity','current_ratio','cfo','free_cash_flow','cfo_to_net_income'}
    present=set(m.metric)
    missing=sorted(expected-present)
    status={'status':'PASS' if bad==0 and not missing and len(a)>0 else 'FAIL','analysis_date':args.analysis_date,'assessment_rows':int(len(a)),'metric_rows':int(len(m)),'publication_dates_after_analysis':bad,'missing_required_metrics':missing,'tickers':sorted(a.ticker.astype(str).unique().tolist())}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(status,indent=2),encoding='utf-8'); print(json.dumps(status,indent=2)); raise SystemExit(0 if status['status']=='PASS' else 1)
if __name__=='__main__': main()
