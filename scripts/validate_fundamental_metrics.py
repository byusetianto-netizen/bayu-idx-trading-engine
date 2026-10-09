#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd

BASE_EXPECTED = {'gross_margin','net_margin','cfo','free_cash_flow','cfo_to_net_income'}

def read_csv_safe(path, required_columns):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=required_columns), f'missing file: {p}'
    if p.stat().st_size == 0:
        return pd.DataFrame(columns=required_columns), f'empty file: {p}'
    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=required_columns), f'empty CSV: {p}'
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        return df, f'missing columns in {p}: {missing}'
    return df, None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--metrics',default='data/fundamental/fundamental_metrics.csv')
    ap.add_argument('--assessment',default='data/fundamental/fundamental_assessment.csv')
    ap.add_argument('--source',default='data/fundamental/financial_statements_pit.csv')
    ap.add_argument('--analysis-date',default='2026-09-01')
    ap.add_argument('--output',default='data/fundamental/fundamental_metrics_validation.json')
    args=ap.parse_args()

    m,me=read_csv_safe(args.metrics,['ticker','metric','period_end','source_document_id'])
    a,ae=read_csv_safe(args.assessment,['ticker','publication_date'])
    s,se=read_csv_safe(args.source,['ticker','metric','period_end','publication_date'])
    ad=pd.Timestamp(args.analysis_date)
    diagnostics=[x for x in (me,ae,se) if x]
    warnings=[]; bad=0; tickers=[]; present=set()

    if not me:
        present=set(m['metric'].dropna().astype(str))
    if not ae:
        a['publication_date']=pd.to_datetime(a['publication_date'],errors='coerce')
        bad=int((a['publication_date']>ad).sum())
        tickers=sorted(a['ticker'].dropna().astype(str).unique().tolist())

    expected=set(BASE_EXPECTED)
    if not se:
        s['publication_date']=pd.to_datetime(s['publication_date'],errors='coerce')
        s=s[s['publication_date'].notna() & (s['publication_date']<=ad)].copy()
        sm=set(s['metric'].dropna().astype(str))

        if {'total_current_assets','total_current_liabilities'} <= sm:
            expected.add('current_ratio')
        else:
            warnings.append('current_ratio not required: source lacks current assets/current liabilities')

        debt={'current_maturities_of_bank_loans','long_term_bank_loans',
              'current_maturities_of_finance_lease_liabilities',
              'long_term_finance_lease_liabilities'}
        if 'total_equity' in sm and debt & sm:
            expected.add('debt_to_equity')
        else:
            warnings.append('debt_to_equity not required: source lacks identified debt fields')

        s['period_end_dt']=pd.to_datetime(s['period_end'],errors='coerce')
        has_yoy=False
        for _,g in s[s['metric'].eq('revenue')].groupby('ticker'):
            if g['period_end_dt'].dropna().dt.year.nunique() >= 2:
                has_yoy=True; break
        if has_yoy:
            expected.add('revenue_growth')
        else:
            warnings.append('revenue_growth not required: no prior-year revenue history in source')

    missing=sorted(expected-present)
    if len(m)==0: diagnostics.append('fundamental_metrics.csv contains zero metric rows')
    if len(a)==0: diagnostics.append('fundamental_assessment.csv contains zero assessment rows')
    if bad: diagnostics.append(f'{bad} assessment row(s) have publication_date after analysis_date')
    if missing: diagnostics.append(f'missing source-supported required metrics: {missing}')

    passed=not diagnostics and len(m)>0 and len(a)>0
    status={'status':'PASS' if passed else 'FAIL','analysis_date':args.analysis_date,
            'assessment_rows':int(len(a)),'metric_rows':int(len(m)),
            'publication_dates_after_analysis':bad,'required_metrics':sorted(expected),
            'missing_required_metrics':missing,'tickers':tickers,
            'warnings':warnings,'diagnostics':diagnostics}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status,indent=2))
    raise SystemExit(0 if passed else 1)

if __name__=='__main__':
    main()
