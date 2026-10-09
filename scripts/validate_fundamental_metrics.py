#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import pandas as pd

CORE={'revenue_growth','gross_margin','net_margin','roa_period','roe_period','current_ratio','cfo','free_cash_flow','cfo_to_net_income'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--metrics',default='data/fundamental/fundamental_metrics.csv')
    ap.add_argument('--assessment',default='data/fundamental/fundamental_assessment.csv')
    ap.add_argument('--source',default='data/fundamental/financial_statements_pit.csv')
    ap.add_argument('--analysis-date',default='2026-09-01')
    ap.add_argument('--output',default='data/fundamental/fundamental_metrics_validation.json')
    args=ap.parse_args()
    diagnostics=[]; warnings=[]
    try: m=pd.read_csv(args.metrics); a=pd.read_csv(args.assessment); s=pd.read_csv(args.source)
    except Exception as exc:
        raise SystemExit(f'Cannot read validation inputs: {exc}')
    ad=pd.Timestamp(args.analysis_date)
    for c in ['period_start','period_end','publication_date']:
        s[c]=pd.to_datetime(s[c],errors='coerce')
    a['publication_date']=pd.to_datetime(a.publication_date,errors='coerce')
    eligible=s[s.publication_date.notna()&(s.publication_date<=ad)].copy()
    source_tickers=sorted(eligible.ticker.dropna().astype(str).unique())
    assessed=sorted(a.ticker.dropna().astype(str).unique())
    missing_tickers=sorted(set(source_tickers)-set(assessed))
    extra_tickers=sorted(set(assessed)-set(source_tickers))
    if missing_tickers: diagnostics.append(f'source ticker(s) not assessed: {missing_tickers}')
    if extra_tickers: diagnostics.append(f'assessed ticker(s) absent from PIT source: {extra_tickers}')
    bad_pub=int((a.publication_date>ad).sum())
    if bad_pub: diagnostics.append(f'{bad_pub} assessment row(s) after analysis_date')

    identity={}
    for t in source_tickers:
        z=eligible[eligible.ticker.astype(str).eq(t)]
        ends=z[z.metric.eq('total_assets')].period_end.dropna()
        if len(ends)==0:
            diagnostics.append(f'{t}: no total_assets period'); continue
        e=ends.max(); q=z[z.period_end.eq(e)]
        def v(k):
            x=q[q.metric.eq(k)].value
            return None if x.empty else float(x.iloc[0])
        A,L,E=v('total_assets'),v('total_liabilities'),v('total_equity')
        if None in (A,L,E):
            diagnostics.append(f'{t}: incomplete accounting identity components'); continue
        gap=abs(A-(L+E))/max(abs(A),1)
        identity[t]=gap
        if gap>0.01: diagnostics.append(f'{t}: accounting identity relative gap {gap:.6f} > 1%')

        flows=z[z.metric.eq('revenue')]
        cur_end=flows.period_end.max() if len(flows) else None
        if cur_end is None: diagnostics.append(f'{t}: no revenue period'); continue
        cur=flows[flows.period_end.eq(cur_end)]
        starts=cur.period_start.dropna()
        if starts.empty: diagnostics.append(f'{t}: current revenue period_start missing'); continue
        ps=starts.min(); pe=pd.Timestamp(cur_end)
        prior=flows[(flows.period_start.eq(ps-pd.DateOffset(years=1)))&(flows.period_end.eq(pe-pd.DateOffset(years=1)))]
        if prior.empty: diagnostics.append(f'{t}: exact prior-year comparable revenue period missing')

        tm=m[m.ticker.astype(str).eq(t)]
        present=set(tm.metric.astype(str))
        miss=sorted(CORE-present)
        # Debt/equity is required only if identified debt fields exist in current source.
        debt_fields={'short_term_bank_loans','current_maturities_of_bank_loans','long_term_bank_loans',
                     'current_maturities_of_finance_lease_liabilities','long_term_finance_lease_liabilities'}
        if set(q.metric.astype(str)) & debt_fields: miss += ([] if 'debt_to_equity' in present else ['debt_to_equity'])
        if miss: diagnostics.append(f'{t}: missing required metric(s): {sorted(set(miss))}')

    passed=not diagnostics and len(a)>0 and len(m)>0
    out={'status':'PASS' if passed else 'FAIL','analysis_date':args.analysis_date,
         'source_tickers':source_tickers,'assessed_tickers':assessed,'assessment_rows':len(a),'metric_rows':len(m),
         'publication_dates_after_analysis':bad_pub,'accounting_identity_relative_gap':identity,
         'diagnostics':diagnostics,'warnings':warnings}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2)); raise SystemExit(0 if passed else 1)
if __name__=='__main__': main()
