#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import pandas as pd

CRITICAL_SOURCE={'revenue','net_income','total_assets','total_liabilities','total_equity','cash_from_operations'}
CRITICAL_METRICS={'revenue_growth','gross_margin','net_margin','current_ratio','cfo','free_cash_flow','cfo_to_net_income'}
IDENTITY_TOL=0.01

def bseries(x):
    return x.astype(str).str.strip().str.lower().isin(['true','1','yes'])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',default='data/fundamental/financial_statements_pit.csv')
    ap.add_argument('--metrics',default='data/fundamental/fundamental_metrics.csv')
    ap.add_argument('--analysis-date',required=True)
    ap.add_argument('--analysis-time',default='23:59:59')
    ap.add_argument('--output',default='data/fundamental/fundamental_reliability.csv')
    ap.add_argument('--status-output',default='data/fundamental/fundamental_reliability_status.json')
    args=ap.parse_args()
    s=pd.read_csv(args.source); m=pd.read_csv(args.metrics)
    for c in ['period_start','period_end','publication_date','publication_timestamp']:
        if c in s: s[c]=pd.to_datetime(s[c],errors='coerce')
    analysis_ts=pd.Timestamp(f"{args.analysis_date} {args.analysis_time}")
    # Candidate rows are selected conservatively: timestamp is authoritative; date is not substituted for missing timestamp.
    known=s[s.publication_timestamp.notna() & (s.publication_timestamp<=analysis_ts)].copy()
    tickers=sorted(set(s.loc[s.publication_date.notna() & (s.publication_date<=pd.Timestamp(args.analysis_date)),'ticker'].astype(str)) | set(known.ticker.astype(str)))
    rows=[]
    for t in tickers:
        all_t=s[s.ticker.astype(str).eq(t)].copy()
        z=known[known.ticker.astype(str).eq(t)].copy()
        reasons=[]; warnings=[]
        # Timestamp readiness is evaluated on the reporting rows that would otherwise be eligible by publication date.
        date_candidate=all_t[all_t.publication_date.notna() & (all_t.publication_date<=pd.Timestamp(args.analysis_date))]
        missing_ts=date_candidate.publication_timestamp.isna().any() if len(date_candidate) else True
        future_ts=(date_candidate.publication_timestamp.notna() & (date_candidate.publication_timestamp>analysis_ts)).any() if len(date_candidate) else False
        if missing_ts:
            pit='REVIEW'; reasons.append('publication_timestamp unavailable for one or more candidate rows')
        elif future_ts:
            pit='FAIL'; reasons.append('publication_timestamp after analysis_timestamp')
        else: pit='PASS'
        rev=z[z.metric.eq('revenue')]
        current_end=None if rev.empty else rev.period_end.max()
        q=z[z.period_end.eq(current_end)] if current_end is not None else z.iloc[0:0]
        def v(k):
            x=q[q.metric.eq(k)].value
            return None if x.empty else float(x.iloc[0])
        A,L,E=v('total_assets'),v('total_liabilities'),v('total_equity')
        gap=None if None in (A,L,E) else abs(A-(L+E))/max(abs(A),1.0)
        identity='PASS' if gap is not None and gap<=IDENTITY_TOL else 'FAIL'
        if identity=='FAIL': reasons.append('accounting identity failed')
        comparable='FAIL'
        if current_end is not None:
            cur=rev[rev.period_end.eq(current_end)]; starts=cur.period_start.dropna()
            if len(starts):
                ps=starts.min(); pe=pd.Timestamp(current_end)
                prior=rev[(rev.period_start.eq(ps-pd.DateOffset(years=1)))&(rev.period_end.eq(pe-pd.DateOffset(years=1)))]
                if len(prior): comparable='PASS'
        if comparable=='FAIL': reasons.append('exact prior-year comparable period unavailable')
        present=set(q.metric.astype(str)); miss_src=sorted(CRITICAL_SOURCE-present)
        src_cov='PASS' if not miss_src else 'FAIL'
        if miss_src: reasons.append('missing critical source metrics: '+','.join(miss_src))
        tm=m[m.ticker.astype(str).eq(t)]; miss_met=sorted(CRITICAL_METRICS-set(tm.metric.astype(str)))
        met_cov='PASS' if not miss_met else 'FAIL'
        if miss_met: reasons.append('missing critical derived metrics: '+','.join(miss_met))
        currencies=sorted(q.currency.dropna().astype(str).unique()) if 'currency' in q else []
        units=sorted(q.unit.dropna().astype(str).unique()) if 'unit' in q else []
        cu='PASS' if len(currencies)==1 and len(units)==1 else 'FAIL'
        if cu=='FAIL': reasons.append(f'inconsistent current-period currency/unit: {currencies}/{units}')
        # Preserve old V2.1C provenance principle: identifiable evidence + confidence.
        prov='FAIL'
        if len(q) and {'document_id','confidence'}.issubset(q.columns):
            bad=q.document_id.isna()|q.document_id.astype(str).str.strip().eq('')|~q.confidence.astype(str).str.upper().isin(['HIGH','MEDIUM'])
            prov='PASS' if not bad.any() else 'FAIL'
        if prov=='FAIL': reasons.append('source provenance/confidence insufficient')
        latest=q.publication_timestamp.max() if len(q) else pd.NaT
        age=None if pd.isna(latest) else int((analysis_ts.normalize()-latest.normalize()).days)
        if age is not None and age>180: warnings.append(f'latest report is {age} days old; freshness is observed, not hard-gated without validated threshold')
        status='BLOCKED' if reasons else ('DEGRADED' if warnings else 'PASS')
        rows.append(dict(ticker=t,analysis_timestamp=analysis_ts.isoformat(),current_period_end=None if current_end is None else pd.Timestamp(current_end).date().isoformat(),
            pit_safety=pit,accounting_integrity=identity,accounting_identity_relative_gap=gap,comparable_period=comparable,
            critical_source_coverage=src_cov,critical_metric_coverage=met_cov,currency_unit_integrity=cu,source_reliability=prov,
            publication_age_days=age,freshness='OBSERVED',reliability_status=status,eligible_for_fundamental_scoring=(status!='BLOCKED'),
            block_reasons=' | '.join(reasons),warning_reasons=' | '.join(warnings)))
    out=pd.DataFrame(rows); Path(args.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.output,index=False)
    st={'status':'BUILT','analysis_timestamp':analysis_ts.isoformat(),'tickers':len(out),'pass':int((out.reliability_status=='PASS').sum()),'degraded':int((out.reliability_status=='DEGRADED').sum()),'blocked':int((out.reliability_status=='BLOCKED').sum()),'eligible':int(out.eligible_for_fundamental_scoring.sum()),'engine_changed':False,'notes':['Reliability/coverage layer only; no investment or trade decision.','publication_timestamp is authoritative for PIT; publication_date never substitutes for a missing timestamp.','Missing values are never imputed as zero.','Freshness is observed but not hard-gated without a validated threshold.']}
    Path(args.status_output).write_text(json.dumps(st,indent=2),encoding='utf-8'); print(json.dumps(st,indent=2))
if __name__=='__main__': main()
