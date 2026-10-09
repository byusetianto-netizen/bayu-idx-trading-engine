#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_INPUT=ROOT/'data/fundamental/financial_statements_pit.csv'
DEFAULT_METRICS=ROOT/'data/fundamental/fundamental_metrics.csv'
DEFAULT_ASSESS=ROOT/'data/fundamental/fundamental_assessment.csv'
DEFAULT_STATUS=ROOT/'data/fundamental/fundamental_metrics_status.json'

REQ=['ticker','metric','value','unit','currency','period_start','period_end',
     'publication_date','period_type','is_cumulative','document_id','confidence']

INCOME={'revenue','gross_profit','profit_before_tax','net_income','net_income_parent'}
BALANCE={'total_assets','total_current_assets','total_liabilities','total_current_liabilities',
         'total_equity','total_equity_attributable_to_equity_owners_of_parent_entity',
         'cash','short_term_bank_loans','current_maturities_of_bank_loans','long_term_bank_loans',
         'current_maturities_of_finance_lease_liabilities','long_term_finance_lease_liabilities'}
CASHFLOW={'cash_from_operations','cash_from_investing','cash_from_financing',
          'payments_for_acquisition_of_property_plant_and_equipment',
          'payments_for_acquisition_of_mining_properties',
          'payments_for_acquisition_of_intangible_assets'}
CAPEX={'payments_for_acquisition_of_property_plant_and_equipment',
       'payments_for_acquisition_of_mining_properties',
       'payments_for_acquisition_of_intangible_assets'}
DEBT={'short_term_bank_loans','current_maturities_of_bank_loans','long_term_bank_loans',
      'current_maturities_of_finance_lease_liabilities','long_term_finance_lease_liabilities'}

def div(a,b):
    if a is None or b is None or pd.isna(a) or pd.isna(b) or abs(float(b))<1e-12:return None
    return float(a)/float(b)
def yoy(a,b):
    if a is None or b is None or pd.isna(a) or pd.isna(b) or abs(float(b))<1e-12:return None
    return float(a)/float(b)-1
def val(q,metric):
    z=q[q.metric.eq(metric)].copy()
    if z.empty:return None
    z['rank']=z.confidence.map({'HIGH':3,'MEDIUM':2,'LOW':1}).fillna(0)
    return float(z.sort_values(['rank','publication_date','document_id'],ascending=[False,False,True]).iloc[0].value)
def unit_for(q):
    x=q.unit.dropna().astype(str)
    return x.iloc[0] if len(x) else 'reported'
def emit(rows,ticker,name,value,unit,s,e,basis,doc,note='',confidence='HIGH'):
    if value is None or pd.isna(value):return
    rows.append(dict(ticker=ticker,metric=name,value=float(value),unit=unit,period_start=s,
                     period_end=e,basis=basis,source_document_id=doc,confidence=confidence,note=note))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default=str(DEFAULT_INPUT))
    ap.add_argument('--metrics-output',default=str(DEFAULT_METRICS))
    ap.add_argument('--assessment-output',default=str(DEFAULT_ASSESS))
    ap.add_argument('--status-output',default=str(DEFAULT_STATUS))
    ap.add_argument('--analysis-date',required=True)
    args=ap.parse_args()
    d=pd.read_csv(args.input)
    miss=[c for c in REQ if c not in d.columns]
    if miss: raise SystemExit(f'Missing columns: {miss}')
    for c in ['period_start','period_end','publication_date']:
        d[c]=pd.to_datetime(d[c],errors='coerce')
    ad=pd.Timestamp(args.analysis_date)
    d=d[d.publication_date.notna() & (d.publication_date<=ad)].copy()
    d['ticker']=d.ticker.astype(str).str.strip()
    d['metric']=d.metric.astype(str).str.strip()
    d['document_id']=d.document_id.fillna('').astype(str)
    rows=[]; assessments=[]; skipped=[]

    for ticker in sorted(d.ticker.unique()):
        t=d[d.ticker.eq(ticker)].copy()
        flows=t[t.metric.isin(INCOME)]
        if flows.empty:
            skipped.append({'ticker':ticker,'reason':'no income statement rows'}); continue
        ends=sorted(flows.period_end.dropna().unique())
        if not ends:
            skipped.append({'ticker':ticker,'reason':'no valid income period_end'}); continue
        e=pd.Timestamp(ends[-1])
        cur=t[t.period_end.eq(e)]
        inc=cur[cur.metric.isin(INCOME)]
        bal=cur[cur.metric.isin(BALANCE)]
        cf=cur[cur.metric.isin(CASHFLOW)]
        starts=inc.period_start.dropna()
        if starts.empty:
            skipped.append({'ticker':ticker,'reason':'current income period_start missing'}); continue
        s=pd.Timestamp(starts.min())
        target_s=s-pd.DateOffset(years=1); target_e=e-pd.DateOffset(years=1)
        prev=t[(t.period_start.eq(target_s))&(t.period_end.eq(target_e))&t.metric.isin(INCOME)]
        if prev.empty:
            skipped.append({'ticker':ticker,'reason':f'no comparable period {target_s.date()}..{target_e.date()}'}); continue

        # Accounting identity gate.
        A,L,E=val(bal,'total_assets'),val(bal,'total_liabilities'),val(bal,'total_equity')
        if None in (A,L,E):
            skipped.append({'ticker':ticker,'reason':'missing assets/liabilities/equity'}); continue
        identity_gap=abs(A-(L+E))/max(abs(A),1.0)
        if identity_gap>0.01:
            skipped.append({'ticker':ticker,'reason':f'accounting identity failed: relative gap={identity_gap:.6f}'}); continue

        doc=inc.sort_values('publication_date').iloc[-1].document_id
        pub=inc.publication_date.max().date().isoformat()
        bun=unit_for(bal); fun=unit_for(cf)
        rev,gp,pbt,ni,nip=[val(inc,x) for x in ['revenue','gross_profit','profit_before_tax','net_income','net_income_parent']]
        prev_rev,prev_gp,prev_ni,prev_nip=[val(prev,x) for x in ['revenue','gross_profit','net_income','net_income_parent']]
        for name,x in [('revenue_growth',yoy(rev,prev_rev)),('gross_profit_growth',yoy(gp,prev_gp)),
                       ('net_income_growth',yoy(ni,prev_ni)),('parent_net_income_growth',yoy(nip,prev_nip))]:
            emit(rows,ticker,name,x,'ratio',s.date().isoformat(),e.date().isoformat(),'same_period_yoy',doc)
        gross_margin,net_margin,pre_tax_margin=div(gp,rev),div(ni,rev),div(pbt,rev)
        emit(rows,ticker,'gross_margin',gross_margin,'ratio',s.date().isoformat(),e.date().isoformat(),'period',doc)
        emit(rows,ticker,'net_margin',net_margin,'ratio',s.date().isoformat(),e.date().isoformat(),'period',doc)
        emit(rows,ticker,'pre_tax_margin',pre_tax_margin,'ratio',s.date().isoformat(),e.date().isoformat(),'period',doc)

        # ROA/ROE use same-date prior-year balance when available.
        pbal=t[(t.period_end.eq(target_e))&t.metric.isin(BALANCE)]
        pA=val(pbal,'total_assets')
        pParent=val(pbal,'total_equity_attributable_to_equity_owners_of_parent_entity')
        parent=val(bal,'total_equity_attributable_to_equity_owners_of_parent_entity')
        avgA=(A+pA)/2 if pA is not None else None
        avgEq=(parent+pParent)/2 if parent is not None and pParent is not None else None
        roa,roe=div(ni,avgA),div(nip,avgEq)
        months=max(1,(e.year-s.year)*12+e.month-s.month+1); factor=12/months
        emit(rows,ticker,'roa_period',roa,'ratio',s.date().isoformat(),e.date().isoformat(),'period',doc)
        emit(rows,ticker,'roe_period',roe,'ratio',s.date().isoformat(),e.date().isoformat(),'period',doc)
        emit(rows,ticker,'roa_annualized_estimate',None if roa is None else roa*factor,'ratio',s.date().isoformat(),e.date().isoformat(),'annualized_estimate',doc,note=f'{months}-month interim annualization')
        emit(rows,ticker,'roe_annualized_estimate',None if roe is None else roe*factor,'ratio',s.date().isoformat(),e.date().isoformat(),'annualized_estimate',doc,note=f'{months}-month interim annualization')

        ca,cl=val(bal,'total_current_assets'),val(bal,'total_current_liabilities')
        current_ratio=div(ca,cl)
        debt=sum(abs(val(bal,x) or 0) for x in DEBT if not pd.isna(val(bal,x) if val(bal,x) is not None else 0))
        debt_present=any(not bal[bal.metric.eq(x)].empty for x in DEBT)
        dte=div(debt,E) if debt_present else None
        emit(rows,ticker,'current_ratio',current_ratio,'ratio',e.date().isoformat(),e.date().isoformat(),'instant',doc)
        emit(rows,ticker,'liabilities_to_equity',div(L,E),'ratio',e.date().isoformat(),e.date().isoformat(),'instant',doc)
        emit(rows,ticker,'debt_to_equity',dte,'ratio',e.date().isoformat(),e.date().isoformat(),'identified_interest_bearing_debt',doc,
             note='Identified bank loans and finance lease liabilities divided by total equity')
        if debt_present: emit(rows,ticker,'interest_bearing_debt',debt,bun,e.date().isoformat(),e.date().isoformat(),'instant',doc)

        cfo,cfi,cff=[val(cf,x) for x in ['cash_from_operations','cash_from_investing','cash_from_financing']]
        for name,x in [('cfo',cfo),('cfi',cfi),('cff',cff)]:
            emit(rows,ticker,name,x,fun,s.date().isoformat(),e.date().isoformat(),'period',doc)
        capex=sum(abs(val(cf,x) or 0) for x in CAPEX)
        capex_present=any(not cf[cf.metric.eq(x)].empty for x in CAPEX)
        if capex_present:
            emit(rows,ticker,'capex',capex,fun,s.date().isoformat(),e.date().isoformat(),'cash_outflow_magnitude',doc,
                 note='Absolute magnitude of identified acquisition payments')
            emit(rows,ticker,'free_cash_flow',None if cfo is None else cfo-capex,fun,s.date().isoformat(),e.date().isoformat(),'CFO-capex',doc,
                 note='FCF = CFO - absolute identified capex payments')
        cfo_ni=div(cfo,ni)
        emit(rows,ticker,'cfo_to_net_income',cfo_ni,'ratio',s.date().isoformat(),e.date().isoformat(),'period',doc)

        nig=yoy(ni,prev_ni)
        growth_cls='UNKNOWN' if nig is None else ('STRONG' if nig>0.05 else ('WEAK' if nig<-0.05 else 'STABLE'))
        annroe=None if roe is None else roe*factor
        profit_cls='STRONG' if ((net_margin is not None and net_margin>0.10) or (annroe is not None and annroe>0.12)) else ('UNKNOWN' if net_margin is None and annroe is None else 'MIXED')
        if E<=0: balance_cls='RISK'
        elif dte is None: balance_cls='UNKNOWN'
        elif dte<1: balance_cls='HEALTHY'
        elif dte<2: balance_cls='WATCH'
        else: balance_cls='RISK'
        cash_cls='HEALTHY' if cfo_ni is not None and cfo_ni>=0.8 and (cfo or 0)>0 else ('CAUTION' if cfo_ni is not None and cfo_ni>=0.5 and (cfo or 0)>0 else 'WEAK')
        positives=sum(x in ('STRONG','HEALTHY') for x in [growth_cls,profit_cls,balance_cls,cash_cls])
        risks=sum(x in ('WEAK','RISK') for x in [growth_cls,profit_cls,balance_cls,cash_cls])
        overall='POSITIVE' if positives>=3 and risks==0 else ('WEAK' if risks>=2 else 'CAUTION')
        assessments.append(dict(ticker=ticker,period_start=s.date().isoformat(),period_end=e.date().isoformat(),
            publication_date=pub,growth=growth_cls,profitability=profit_cls,balance_sheet=balance_cls,cash_flow=cash_cls,
            overall_fundamental_view=overall,revenue_growth=yoy(rev,prev_rev),net_income_growth=nig,net_margin=net_margin,
            roe_annualized_estimate=annroe,debt_to_equity=dte,cfo_to_net_income=cfo_ni,pit_safe=True,
            accounting_identity_relative_gap=identity_gap,source_document_id=doc))

    metric_cols=['ticker','metric','value','unit','period_start','period_end','basis','source_document_id','confidence','note']
    assess_cols=['ticker','period_start','period_end','publication_date','growth','profitability','balance_sheet','cash_flow',
                 'overall_fundamental_view','revenue_growth','net_income_growth','net_margin','roe_annualized_estimate',
                 'debt_to_equity','cfo_to_net_income','pit_safe','accounting_identity_relative_gap','source_document_id']
    m=pd.DataFrame(rows,columns=metric_cols); a=pd.DataFrame(assessments,columns=assess_cols)
    Path(args.metrics_output).parent.mkdir(parents=True,exist_ok=True); m.to_csv(args.metrics_output,index=False); a.to_csv(args.assessment_output,index=False)
    status={'status':'FUNDAMENTAL_METRICS_BUILT','analysis_date':args.analysis_date,'tickers_processed':len(a),
            'metric_rows':len(m),'assessment_rows':len(a),'tickers':sorted(a.ticker.tolist()) if len(a) else [],
            'skipped_tickers':skipped,
            'notes':['Comparable growth requires exact prior-year period.','Accounting identity gate tolerance is 1%.',
                     'FCF uses CFO minus absolute identified capex payments.','Debt-to-equity uses identified debt divided by total equity.']}
    Path(args.status_output).write_text(json.dumps(status,indent=2),encoding='utf-8'); print(json.dumps(status,indent=2))
if __name__=='__main__': main()
