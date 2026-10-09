#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / 'data/fundamental/financial_statements_pit.csv'
DEFAULT_METRICS = ROOT / 'data/fundamental/fundamental_metrics.csv'
DEFAULT_ASSESS = ROOT / 'data/fundamental/fundamental_assessment.csv'
DEFAULT_STATUS = ROOT / 'data/fundamental/fundamental_metrics_status.json'

REQ = [
    'ticker',
    'metric',
    'value',
    'currency',
    'period_start',
    'period_end',
    'publication_date',
    'period_type',
    'is_cumulative',
    'report_date',
    'document_id',
    'confidence'
]

ALIASES = {
 'revenue':'revenue', 'gross_profit':'gross_profit', 'profit_before_tax':'profit_before_tax',
 'net_income':'net_income', 'net_income_attributable_parent':'net_income_parent',
 'eps_basic':'basic_earnings_loss_per_share_from_continuing_operations',
 'eps_diluted':'diluted_earnings_loss_per_share_from_continuing_operations',
 'total_assets':'total_assets','total_current_assets':'total_current_assets',
 'total_liabilities':'total_liabilities','total_current_liabilities':'total_current_liabilities',
 'total_equity':'total_equity','parent_equity':'total_equity_attributable_to_equity_owners_of_parent_entity',
 'cash':'cash','cfo':'cash_from_operations','cfi':'cash_from_investing','cff':'cash_from_financing',
 'ppe_capex':'payments_for_acquisition_of_property_plant_and_equipment',
 'mining_capex':'payments_for_acquisition_of_mining_properties',
 'intangibles_capex':'payments_for_acquisition_of_intangible_assets',
 'current_bank_debt':'current_maturities_of_bank_loans',
 'long_term_bank_debt':'long_term_bank_loans',
 'current_lease_debt':'current_maturities_of_finance_lease_liabilities',
 'long_term_lease_debt':'long_term_finance_lease_liabilities',
}

def pct(x): return None if x is None or pd.isna(x) else float(x*100)
def safe_div(a,b):
    if a is None or b is None or pd.isna(a) or pd.isna(b) or abs(float(b)) < 1e-12: return None
    return float(a)/float(b)
def growth(cur, prev): return None if cur is None or prev is None or abs(prev)<1e-12 else (cur/prev)-1

def pick(df, ticker, statement, key, period_end=None, period_start=None):
    q=df[(df.ticker==ticker)&(df.statement==statement)&(df.metric_key==ALIASES[key])].copy()
    if period_end is not None: q=q[q.period_end==period_end]
    if period_start is not None: q=q[q.period_start==period_start]
    if q.empty: return None
    # prefer highest confidence and most recent publication, deterministic
    q['confidence_rank']=q.confidence.map({'HIGH':3,'MEDIUM':2,'LOW':1}).fillna(0)
    q=q.sort_values(['confidence_rank','publication_date','document_id'], ascending=[False,False,True])
    return float(q.iloc[0].value)

def available_periods(df, ticker, statement, cumulative=True):
    q=df[(df.ticker==ticker)&(df.statement==statement)].copy()
    if cumulative: q=q[q.is_cumulative==True]
    return q[['period_start','period_end','publication_date','document_id']].drop_duplicates().sort_values('period_end')

def select_current_income(df,ticker):
    q=available_periods(df,ticker,'profit_and_loss',True)
    if q.empty: return None
    # latest period end, then latest publication
    r=q.sort_values(['period_end','publication_date'],ascending=[False,False]).iloc[0]
    return str(r.period_start),str(r.period_end),str(r.publication_date),str(r.document_id)

def select_prior_same_period_income(df,ticker,current):
    start,end,_,_=current
    s=pd.Timestamp(start); e=pd.Timestamp(end)
    target_start=(s-pd.DateOffset(years=1)).date().isoformat()
    target_end=(e-pd.DateOffset(years=1)).date().isoformat()
    q=available_periods(df,ticker,'profit_and_loss',True)
    q=q[(q.period_start==target_start)&(q.period_end==target_end)]
    if q.empty: return None
    r=q.sort_values('publication_date',ascending=False).iloc[0]
    return str(r.period_start),str(r.period_end),str(r.publication_date),str(r.document_id)

def select_balance_current(df,ticker):
    q=available_periods(df,ticker,'statement_of_financial_position',False)
    if q.empty:return None
    r=q.sort_values(['period_end','publication_date'],ascending=[False,False]).iloc[0]
    return str(r.period_end),str(r.publication_date),str(r.document_id)

def select_balance_prior(df,ticker,current_end):
    e=pd.Timestamp(current_end)
    target=(e-pd.DateOffset(years=1)).date().isoformat()
    q=available_periods(df,ticker,'statement_of_financial_position',False)
    # Prefer exact year-end; otherwise latest balance before current with prior-year vicinity.
    exact=q[q.period_end==target]
    if not exact.empty:
        r=exact.sort_values('publication_date',ascending=False).iloc[0]
    else:
        before=q[pd.to_datetime(q.period_end)<e]
        if before.empty:return None
        r=before.sort_values(['period_end','publication_date'],ascending=[False,False]).iloc[0]
    return str(r.period_end),str(r.publication_date),str(r.document_id)

def emit_metric(rows, ticker, name, value, unit, period_start, period_end, basis, source_doc, confidence='HIGH', note=''):
    if value is None or pd.isna(value): return
    rows.append({'ticker':ticker,'metric':name,'value':float(value),'unit':unit,'period_start':period_start,'period_end':period_end,'basis':basis,'source_document_id':source_doc,'confidence':confidence,'note':note})

def classify(growth, margin, roe, leverage, cfo_quality):
    growth_ok = growth is not None and growth > 0.05
    growth_bad = growth is not None and growth < -0.05
    margin_ok = margin is not None and margin > 0.10
    roe_ok = roe is not None and roe > 0.12
    leverage_ok = leverage is not None and leverage < 1.0
    if cfo_quality == 'HEALTHY': cash='HEALTHY'
    elif cfo_quality == 'CAUTION': cash='CAUTION'
    else: cash='UNKNOWN'
    if growth_ok: g='STRONG'
    elif growth_bad: g='WEAK'
    elif growth is None: g='UNKNOWN'
    else: g='STABLE'
    if margin_ok or roe_ok: p='STRONG'
    elif margin is None and roe is None: p='UNKNOWN'
    else: p='MIXED'
    if leverage is None: b='UNKNOWN'
    elif leverage_ok: b='HEALTHY'
    elif leverage < 2: b='WATCH'
    else: b='RISK'
    positives=sum(x=='STRONG' or x=='HEALTHY' for x in [g,p,b,cash])
    risks=sum(x in ('WEAK','RISK') for x in [g,p,b,cash])
    overall='POSITIVE' if positives>=3 and risks==0 else ('CAUTION' if risks<=1 else 'WEAK')
    return g,p,b,cash,overall

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=str(DEFAULT_INPUT))
    ap.add_argument('--metrics-output', default=str(DEFAULT_METRICS))
    ap.add_argument('--assessment-output', default=str(DEFAULT_ASSESS))
    ap.add_argument('--status-output', default=str(DEFAULT_STATUS))
    ap.add_argument('--analysis-date', required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    missing = [c for c in REQ if c not in df.columns]
    if missing:
        raise SystemExit(f'Missing columns: {missing}')
    # ------------------------------------------------------------------
    # PIT schema adapter
    # financial_statements_pit.csv uses "metric" as the canonical field.
    # Internal aliases below preserve compatibility with the metrics engine.
    # ------------------------------------------------------------------

    df['metric_key'] = df['metric'].astype(str).str.strip()

    INCOME_METRICS = {
        'revenue',
        'gross_profit',
        'profit_before_tax',
        'net_income',
        'net_income_parent',
        'basic_earnings_loss_per_share_from_continuing_operations',
        'diluted_earnings_loss_per_share_from_continuing_operations',
    }

    BALANCE_METRICS = {
        'total_assets',
        'total_current_assets',
        'total_liabilities',
        'total_current_liabilities',
        'total_equity',
        'total_equity_attributable_to_equity_owners_of_parent_entity',
        'cash',
        'current_maturities_of_bank_loans',
        'long_term_bank_loans',
        'current_maturities_of_finance_lease_liabilities',
        'long_term_finance_lease_liabilities',
    }

    CASH_FLOW_METRICS = {
        'cash_from_operations',
        'cash_from_investing',
        'cash_from_financing',
        'payments_for_acquisition_of_property_plant_and_equipment',
        'payments_for_acquisition_of_mining_properties',
        'payments_for_acquisition_of_intangible_assets',
    }

    def infer_statement(metric):
        if metric in INCOME_METRICS:
            return 'profit_and_loss'
        if metric in BALANCE_METRICS:
            return 'statement_of_financial_position'
        if metric in CASH_FLOW_METRICS:
            return 'cash_flow'
        return 'unknown'

    df['statement'] = df['metric_key'].map(infer_statement)
    # ------------------------------------------------------------------
    # Normalize PIT period metadata
    # ------------------------------------------------------------------

    df['period_type'] = (
        df['period_type']
        .fillna('UNKNOWN')
        .astype(str)
        .str.strip()
        .str.upper()
    )

    def normalize_cumulative(value):
        if pd.isna(value):
            return False

        v = str(value).strip().lower()

        if v in {'true', '1', 'yes', 'y'}:
            return True

        if v in {'false', '0', 'no', 'n'}:
            return False

        return False

    df['is_cumulative'] = df['is_cumulative'].map(normalize_cumulative)

    # Infer cumulative status only when source metadata is missing.
    # Balance-sheet metrics are instant values.
    # Income-statement and cash-flow metrics are treated as cumulative
    # when their reporting period starts near the beginning of the year.

    period_start_dt = pd.to_datetime(df['period_start'], errors='coerce')
    period_end_dt = pd.to_datetime(df['period_end'], errors='coerce')

    flow_mask = df['statement'].isin([
        'profit_and_loss',
        'cash_flow'
    ])

    same_year = period_start_dt.dt.year.eq(period_end_dt.dt.year)

    starts_near_year_begin = period_start_dt.dt.month.le(2)

    infer_cumulative_mask = (
        flow_mask
        & same_year
        & starts_near_year_begin
        & df['period_type'].eq('UNKNOWN')
    )

    df.loc[
        infer_cumulative_mask,
        'is_cumulative'
    ] = True

    df['period_end'] = pd.to_datetime(
        df.period_end,
        errors='coerce'
    ).dt.strftime('%Y-%m-%d')

    df['period_start'] = pd.to_datetime(
        df.period_start,
        errors='coerce'
    ).dt.strftime('%Y-%m-%d')

    df['publication_date'] = pd.to_datetime(
        df.publication_date,
        errors='coerce'
    ).dt.strftime('%Y-%m-%d')

    # Point-in-Time cutoff:
    # Only information publicly available by analysis_date may be used.
    analysis_date = pd.Timestamp(args.analysis_date)

    publication_dt = pd.to_datetime(
        df['publication_date'],
        errors='coerce'
    )

    df = df[
        publication_dt.notna() &
        (publication_dt <= analysis_date)
    ].copy()

    rows = []
    assessments = []
    for ticker in sorted(df.ticker.dropna().unique()):
        inc=select_current_income(df,ticker)
        if not inc: continue
        prev_inc=select_prior_same_period_income(df,ticker,inc)
        bal=select_balance_current(df,ticker)
        prev_bal=select_balance_prior(df,ticker,inc[1]) if bal else None
        if not bal: continue
        s,e,pub,doc=inc
        be,pub_b,doc_b=bal
        prev_e=prev_inc[1] if prev_inc else None
        # income metrics
        vals={k:pick(df,ticker,'profit_and_loss',k,period_end=e,period_start=s) for k in ['revenue','gross_profit','profit_before_tax','net_income','net_income_attributable_parent','eps_basic']}
        prevvals={k:pick(df,ticker,'profit_and_loss',k,period_end=prev_e,period_start=prev_inc[0]) for k in vals} if prev_inc else {}
        # balance
        bvals={k:pick(df,ticker,'statement_of_financial_position',k,period_end=be) for k in ['total_assets','total_current_assets','total_liabilities','total_current_liabilities','total_equity','parent_equity','cash','current_bank_debt','long_term_bank_debt','current_lease_debt','long_term_lease_debt']}
        pbvals={k:pick(df,ticker,'statement_of_financial_position',k,period_end=prev_bal[0]) for k in bvals} if prev_bal else {}
        cvals={k:pick(df,ticker,'cash_flow',k,period_end=e,period_start=s) for k in ['cfo','cfi','cff','ppe_capex','mining_capex','intangibles_capex']}
        # Growth
        for k,label in [('revenue','revenue_growth'),('gross_profit','gross_profit_growth'),('net_income','net_income_growth'),('net_income_attributable_parent','parent_net_income_growth'),('eps_basic','eps_growth')]:
            emit_metric(rows,ticker,label,growth(vals.get(k),prevvals.get(k)),'ratio',s,e,'same_period_yoy',doc)
        # Margins
        emit_metric(rows,ticker,'gross_margin',safe_div(vals.get('gross_profit'),vals.get('revenue')),'ratio',s,e,'period',doc)
        emit_metric(rows,ticker,'net_margin',safe_div(vals.get('net_income'),vals.get('revenue')),'ratio',s,e,'period',doc)
        emit_metric(rows,ticker,'pre_tax_margin',safe_div(vals.get('profit_before_tax'),vals.get('revenue')),'ratio',s,e,'period',doc)
        # ROA/ROE: both period and annualized estimate for interim
        avg_assets=(bvals.get('total_assets') + pbvals.get('total_assets'))/2 if bvals.get('total_assets') is not None and pbvals.get('total_assets') is not None else None
        avg_eq=(bvals.get('parent_equity') + pbvals.get('parent_equity'))/2 if bvals.get('parent_equity') is not None and pbvals.get('parent_equity') is not None else None
        roa=safe_div(vals.get('net_income'),avg_assets); roe=safe_div(vals.get('net_income_attributable_parent'),avg_eq)
        months=max(1,(pd.Timestamp(e).year-pd.Timestamp(s).year)*12+pd.Timestamp(e).month-pd.Timestamp(s).month+1)
        factor=12/months if months>0 else 1
        emit_metric(rows,ticker,'roa_period',roa,'ratio',s,e,'period',doc)
        emit_metric(rows,ticker,'roe_period',roe,'ratio',s,e,'period',doc)
        emit_metric(rows,ticker,'roa_annualized_estimate',roa*factor if roa is not None else None,'ratio',s,e,'annualized_estimate',doc,note=f'{months}-month interim annualization')
        emit_metric(rows,ticker,'roe_annualized_estimate',roe*factor if roe is not None else None,'ratio',s,e,'annualized_estimate',doc,note=f'{months}-month interim annualization')
        # Balance ratios
        emit_metric(rows,ticker,'debt_to_equity',safe_div((bvals.get('current_bank_debt') or 0)+(bvals.get('long_term_bank_debt') or 0)+(bvals.get('current_lease_debt') or 0)+(bvals.get('long_term_lease_debt') or 0),bvals.get('parent_equity') or bvals.get('total_equity')),'ratio',be,be,'instant',doc_b)
        emit_metric(rows,ticker,'liabilities_to_equity',safe_div(bvals.get('total_liabilities'),bvals.get('total_equity')),'ratio',be,be,'instant',doc_b)
        emit_metric(rows,ticker,'current_ratio',safe_div(bvals.get('total_current_assets'),bvals.get('total_current_liabilities')),'ratio',be,be,'instant',doc_b)
        debt=sum((bvals.get(k) or 0) for k in ['current_bank_debt','long_term_bank_debt','current_lease_debt','long_term_lease_debt'])
        net_debt=debt-(bvals.get('cash') or 0)
        emit_metric(rows,ticker,'interest_bearing_debt',debt,'reporting_currency_thousand',be,be,'instant',doc_b)
        emit_metric(rows,ticker,'net_debt',net_debt,'reporting_currency_thousand',be,be,'instant',doc_b)
        # Cash flow
        for k in ['cfo','cfi','cff']: emit_metric(rows,ticker,k,cvals.get(k),'reporting_currency_thousand',s,e,'period',doc)
        capex=sum((cvals.get(k) or 0) for k in ['ppe_capex','mining_capex','intangibles_capex'])
        fcf=(cvals.get('cfo') or 0)-capex
        emit_metric(rows,ticker,'free_cash_flow',fcf,'reporting_currency_thousand',s,e,'CFO+investment_capex',doc,note='Capex defined as PPE + mining properties + intangible acquisitions; source cash-flow signs retained')
        cfo_ni=safe_div(cvals.get('cfo'),vals.get('net_income'))
        emit_metric(rows,ticker,'cfo_to_net_income',cfo_ni,'ratio',s,e,'period',doc)
        # assessment
        growth_y=growth(vals.get('net_income'),prevvals.get('net_income'))
        margin=safe_div(vals.get('net_income'),vals.get('revenue'))
        leverage=safe_div((bvals.get('current_bank_debt') or 0)+(bvals.get('long_term_bank_debt') or 0)+(bvals.get('current_lease_debt') or 0)+(bvals.get('long_term_lease_debt') or 0),bvals.get('parent_equity') or bvals.get('total_equity'))
        cash_quality='HEALTHY' if cfo_ni is not None and cfo_ni>=0.8 and (cvals.get('cfo') or 0)>0 else ('CAUTION' if cfo_ni is not None and cfo_ni>=0.5 else 'WEAK')
        g,p,b,c,o=classify(growth_y,margin,roe*factor if roe is not None else None,leverage,cash_quality)
        assessments.append({'ticker':ticker,'period_start':s,'period_end':e,'publication_date':pub,'growth':g,'profitability':p,'balance_sheet':b,'cash_flow':c,'overall_fundamental_view':o,'revenue_growth':growth(vals.get('revenue'),prevvals.get('revenue')),'net_income_growth':growth_y,'net_margin':margin,'roe_annualized_estimate':roe*factor if roe is not None else None,'debt_to_equity':leverage,'cfo_to_net_income':cfo_ni,'pit_safe':True,'source_document_id':doc})
    out=pd.DataFrame(rows); ass=pd.DataFrame(assessments)
    Path(args.metrics_output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.metrics_output,index=False)
    ass.to_csv(args.assessment_output,index=False)
    status={'status':'FUNDAMENTAL_METRICS_BUILT','engine_changed':False,'tickers_processed':int(len(ass)),'metric_rows':int(len(out)),'assessment_rows':int(len(ass)),'notes':['Descriptive metrics only; no investment score or trade decision.','Interim ROA/ROE annualized estimates are explicitly labeled.','Debt-to-equity uses identified bank loans and finance leases only; other debt may require taxonomy expansion.','PIT safety inherits publication_date from the normalized financial statements.']}
    Path(args.status_output).write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status,indent=2))

if __name__=='__main__': main()
