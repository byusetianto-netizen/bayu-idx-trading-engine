#!/usr/bin/env python3
"""V2.2 Complete Valuation Engine — timestamp-authoritative PIT edition.

Research-only. Builds current valuation, PIT historical valuation evidence,
peer-relative valuation, GARP diagnostics and value-trap flags.
Never changes actual_engine.py and never emits BUY/SELL/probability.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path('.')
PRICE = ROOT/'data/idx_stock_prices.csv'
PRICE_EXPANSION = ROOT/'data/idx_stock_prices_expansion.csv'
PRICE_SUPPLEMENT = ROOT/'data/valuation/valuation_price_supplement.csv'
FUND_DIR = ROOT/'data/fundamental'
METRICS = FUND_DIR/'fundamental_metrics.csv'
FS_PIT = FUND_DIR/'financial_statements_pit.csv'
MANIFEST = FUND_DIR/'financial_source_manifest.csv'
RATIO_SNAP = FUND_DIR/'idx_financial_ratio_snapshots.csv'
SECTOR = ROOT/'data/sector/sector_map.csv'
OUTDIR = ROOT/'data/valuation'; OUTDIR.mkdir(parents=True, exist_ok=True)

MIN_HIST_OBS = 120
MIN_HIST_PERIODS = 2
MIN_PEERS = 3
TZ = 'Asia/Jakarta'

ALIASES = {
    'eps': ['eps','eps_basic','basic_eps','basic_earnings_loss_per_share_from_continuing_operations'],
    'bvps': ['bvps','book_value_per_share','book value per share'],
    'net_income': ['net_income','net_income_attributable_parent'],
    'equity': ['total_equity','parent_equity','equity'],
    'shares': ['shares_outstanding','weighted_average_shares','weighted_average_number_of_shares','ordinary_shares'],
}

def num(x):
    try:
        if pd.isna(x): return np.nan
        return float(x)
    except Exception:
        return np.nan

def read_csv(path):
    if not path.exists(): return pd.DataFrame()
    try: return pd.read_csv(path)
    except Exception: return pd.DataFrame()

def normalize_cols(df):
    if df.empty: return df
    df=df.copy()
    df.columns=[str(c).strip().lower() for c in df.columns]
    return df

def strict_bool(s):
    return s.astype(str).str.strip().str.lower().isin({'true','1','yes','y'})

def utc_series(s):
    return pd.to_datetime(s, errors='coerce', utc=True, format='mixed')

def analysis_clock(latest_date):
    d=pd.to_datetime(latest_date, errors='coerce')
    if pd.isna(d): return pd.NaT, pd.NaT
    local=pd.Timestamp(d.date()).tz_localize(TZ)+pd.Timedelta(hours=23,minutes=59,seconds=59)
    return local, local.tz_convert('UTC')

def price_clock_series(s):
    d=pd.to_datetime(s, errors='coerce')
    local=d.dt.normalize().dt.tz_localize(TZ)+pd.Timedelta(hours=23,minutes=59,seconds=59)
    return local.dt.tz_convert('UTC')

def load_prices(pit_tickers=None):
    frames=[]
    primary=normalize_cols(read_csv(PRICE)); frames.append(primary)
    needed=set(str(x).upper().strip() for x in (pit_tickers or []))
    for extra_path in (PRICE_EXPANSION, PRICE_SUPPLEMENT):
        if not extra_path.exists(): continue
        extra=normalize_cols(read_csv(extra_path))
        if not extra.empty and 'ticker' in extra.columns and needed:
            extra['ticker']=extra['ticker'].astype(str).str.upper().str.strip()
            extra=extra[extra['ticker'].isin(needed)].copy()
        frames.append(extra)
    p=pd.concat([x for x in frames if not x.empty],ignore_index=True) if any(not x.empty for x in frames) else pd.DataFrame()
    if not {'ticker','date','close'}.issubset(p.columns): return pd.DataFrame()
    p['ticker']=p['ticker'].astype(str).str.upper().str.strip()
    p['date']=pd.to_datetime(p['date'],errors='coerce').dt.tz_localize(None)
    p['close']=pd.to_numeric(p['close'],errors='coerce')
    p=p.dropna(subset=['ticker','date','close']).drop_duplicates(['ticker','date'],keep='first')
    return p.sort_values(['ticker','date'])

def load_fundamentals():
    fs=normalize_cols(read_csv(FS_PIT))
    m=normalize_cols(read_csv(METRICS))
    man=normalize_cols(read_csv(MANIFEST))
    ratio=normalize_cols(read_csv(RATIO_SNAP))
    for d in (fs,m,man,ratio):
        if d.empty: continue
        if 'ticker' in d.columns: d['ticker']=d['ticker'].astype(str).str.upper().str.strip()
        for c in ('period_start','period_end','publication_date','report_date'):
            if c in d.columns: d[c]=pd.to_datetime(d[c],errors='coerce',utc=True)
        if 'publication_timestamp' in d.columns:
            d['publication_timestamp']=utc_series(d['publication_timestamp'])
        if 'pit_ready' in d.columns:
            d['pit_ready']=strict_bool(d['pit_ready'])
    return fs,m,man,ratio

def load_sector():
    s=normalize_cols(read_csv(SECTOR))
    if 'ticker' in s.columns: s['ticker']=s['ticker'].astype(str).str.upper().str.strip()
    return s

def fs_schema_ok(fs):
    return (not fs.empty and
            {'ticker','period_end','publication_timestamp','pit_ready'}.issubset(fs.columns))

def pit_evidence_for_ticker(fs,ticker,analysis_ts_utc):
    result={'pit_status':'PIT_UNKNOWN','pit_period_end':pd.NaT,
            'pit_publication_timestamp':pd.NaT,'pit_publication_date':pd.NaT,'pit_rows':0}
    if not fs_schema_ok(fs) or pd.isna(analysis_ts_utc): return result
    q=fs[fs['ticker'].eq(str(ticker).upper().strip())].copy()
    if q.empty: return result
    q=q[q['pit_ready'] & q['period_end'].notna() & q['publication_timestamp'].notna()
        & (q['publication_timestamp']<=analysis_ts_utc)].copy()
    if q.empty: return result
    latest_period=q['period_end'].max()
    z=q[q['period_end'].eq(latest_period)].sort_values('publication_timestamp')
    if z.empty: return result
    result['pit_status']='PIT_VERIFIED'
    result['pit_period_end']=latest_period
    result['pit_publication_timestamp']=z['publication_timestamp'].max()
    if 'publication_date' in z.columns and z['publication_date'].notna().any():
        result['pit_publication_date']=z['publication_date'].dropna().max()
    result['pit_rows']=int(len(z))
    return result

def timestamp_safe_aux(aux, ticker, analysis_ts_utc, period_end=None):
    """Fail closed: auxiliary data is usable only with its own valid timestamp + pit_ready."""
    if aux.empty or not {'ticker','publication_timestamp','pit_ready'}.issubset(aux.columns):
        return pd.DataFrame()
    q=aux[aux['ticker'].eq(str(ticker).upper().strip())].copy()
    q=q[q['pit_ready'] & q['publication_timestamp'].notna()
        & (q['publication_timestamp']<=analysis_ts_utc)].copy()
    if period_end is not None and 'period_end' in q.columns:
        q=q[q['period_end'].eq(period_end)]
    return q

def extract_periodic(fs,ticker,analysis_ts_utc,ratio_snap=None):
    if not fs_schema_ok(fs) or 'value' not in fs.columns: return pd.DataFrame()
    q=fs[fs['ticker'].eq(str(ticker).upper().strip())].copy()
    q['value']=pd.to_numeric(q['value'],errors='coerce')
    q=q[q['pit_ready'] & q['value'].notna() & q['period_end'].notna()
        & q['publication_timestamp'].notna()
        & (q['publication_timestamp']<=analysis_ts_utc)].copy()
    if q.empty: return pd.DataFrame()

    ratio=ratio_snap.copy() if isinstance(ratio_snap,pd.DataFrame) else pd.DataFrame()
    ratio=timestamp_safe_aux(ratio,ticker,analysis_ts_utc)

    def ratio_value(g, aliases, direct_aliases=()):
        if g.empty: return np.nan
        cols={str(c).lower():c for c in g.columns}
        for name in direct_aliases:
            c=cols.get(name.lower())
            if c:
                vals=pd.to_numeric(g[c],errors='coerce').dropna()
                if not vals.empty: return num(vals.iloc[-1])
        metric_col=cols.get('metric') or cols.get('metric_key'); value_col=cols.get('value')
        if metric_col and value_col:
            z=g[g[metric_col].astype(str).str.lower().isin({x.lower() for x in aliases})]
            vals=pd.to_numeric(z[value_col],errors='coerce').dropna()
            if not vals.empty: return num(vals.iloc[-1])
        return np.nan

    out=[]
    for pe,g in q.groupby('period_end',dropna=True):
        pub_ts=g['publication_timestamp'].max()
        if pd.isna(pub_ts): continue
        def pick(names):
            aliases={x.lower() for x in names}
            col='metric_key' if 'metric_key' in g.columns else ('metric' if 'metric' in g.columns else None)
            if not col: return np.nan
            z=g[g[col].astype(str).str.lower().isin(aliases)]
            return num(z.iloc[0]['value']) if not z.empty else np.nan
        eps=pick(ALIASES['eps']); bvps=pick(ALIASES['bvps'])
        if not ratio.empty and 'period_end' in ratio.columns:
            rg=ratio[ratio['period_end'].eq(pe)].copy()
            if not rg.empty:
                eps_r=ratio_value(rg,ALIASES['eps'],('eps','eps_basic'))
                bvps_r=ratio_value(rg,ALIASES['bvps'],('bvps','book_value_per_share'))
                if not np.isfinite(eps) and np.isfinite(eps_r): eps=eps_r
                if not np.isfinite(bvps) and np.isfinite(bvps_r): bvps=bvps_r
        if not np.isfinite(bvps):
            eq=pick(ALIASES['equity']); sh=pick(ALIASES['shares'])
            if np.isfinite(eq) and np.isfinite(sh) and sh>0:
                unit_eq=' '.join(g['unit'].astype(str).tolist()).lower() if 'unit' in g.columns else ''
                if 'share' in unit_eq or 'per share' in unit_eq: bvps=eq/sh
        out.append({'ticker':ticker,'period_end':pe,'publication_timestamp':pub_ts,'eps':eps,'bvps':bvps})
    return pd.DataFrame(out)

def current_metric_map(m,ticker,analysis_ts_utc):
    q=timestamp_safe_aux(m,ticker,analysis_ts_utc)
    if q.empty or 'metric' not in q.columns or 'value' not in q.columns: return {}
    q['value']=pd.to_numeric(q['value'],errors='coerce')
    res={}
    names=['revenue_growth','revenue_growth_pct','net_income_growth','net_profit_growth',
           'net_profit_growth_pct','roe_annualized_estimate','roe_annualized','net_margin']
    for name in names:
        z=q[q.metric.astype(str).str.lower().eq(name.lower())].sort_values(
            [c for c in ['period_end','publication_timestamp'] if c in q.columns])
        if not z.empty: res[name]=num(z.iloc[-1].value)
    return res

def valuation_obs_for_ticker(prices,fund_hist):
    if prices.empty or fund_hist.empty: return pd.DataFrame()
    pp=prices[['date','close']].sort_values('date').copy()
    pp['analysis_timestamp_utc']=price_clock_series(pp['date'])
    ff=fund_hist[['publication_timestamp','period_end','eps','bvps']].dropna(
        subset=['publication_timestamp']).sort_values('publication_timestamp').copy()
    z=pd.merge_asof(pp.sort_values('analysis_timestamp_utc'),ff,
                    left_on='analysis_timestamp_utc',right_on='publication_timestamp',
                    direction='backward')
    z['per']=np.where((z.eps>0)&np.isfinite(z.eps),z.close/z.eps,np.nan)
    z['pbv']=np.where((z.bvps>0)&np.isfinite(z.bvps),z.close/z.bvps,np.nan)
    return z.dropna(subset=['date'])

def hist_stats(obs,current_date):
    result={'historical_status':'UNKNOWN','historical_obs':0,'historical_periods':0,
            'per_hist_median':np.nan,'per_hist_p10':np.nan,'per_hist_p90':np.nan,'per_percentile':np.nan,
            'pbv_hist_median':np.nan,'pbv_hist_p10':np.nan,'pbv_hist_p90':np.nan,'pbv_percentile':np.nan}
    if obs.empty: return result
    result['historical_obs']=int(len(obs)); result['historical_periods']=int(obs['period_end'].nunique())
    if result['historical_periods']>=MIN_HIST_PERIODS and len(obs)>=MIN_HIST_OBS:
        result['historical_status']='VERIFIED_PIT_RANGE'
        for metric in ['per','pbv']:
            a=pd.to_numeric(obs[metric],errors='coerce').dropna()
            if len(a)>=MIN_HIST_OBS:
                cur=num(a.iloc[-1]); result[f'{metric}_hist_median']=float(a.median())
                result[f'{metric}_hist_p10']=float(a.quantile(.10)); result[f'{metric}_hist_p90']=float(a.quantile(.90))
                if np.isfinite(cur): result[f'{metric}_percentile']=float((a<=cur).mean()*100)
    else:
        result['historical_status']='INSUFFICIENT_PIT_HISTORY'
    return result

def peer_stats(snapshot):
    if snapshot.empty or 'sector' not in snapshot.columns: return snapshot
    out=snapshot.copy()
    out['peer_count']=0; out['peer_per_median']=np.nan; out['peer_pbv_median']=np.nan
    out['peer_growth_median']=np.nan; out['peer_roe_median']=np.nan; out['peer_status']='UNKNOWN'
    for idx,r in out.iterrows():
        sec=str(r.get('sector','')).strip()
        if not sec or sec.lower() in {'nan','none'}: continue
        peers=out[(out['sector'].astype(str).str.lower()==sec.lower())&(out['ticker']!=r['ticker'])]
        peers=peers[peers['pit_current_status'].eq('PIT_VERIFIED')]
        out.at[idx,'peer_count']=len(peers)
        if len(peers)<MIN_PEERS: continue
        for c in ['per','pbv','net_profit_growth_pct','roe_pct']:
            vals=pd.to_numeric(peers[c],errors='coerce').dropna()
            if len(vals)>=MIN_PEERS:
                target={'per':'peer_per_median','pbv':'peer_pbv_median',
                        'net_profit_growth_pct':'peer_growth_median','roe_pct':'peer_roe_median'}[c]
                out.at[idx,target]=float(vals.median())
        out.at[idx,'peer_status']='PEER_SET_AVAILABLE'
    return out

def classify(r):
    if r['pit_current_status']!='PIT_VERIFIED': return 'INSUFFICIENT_PIT_DATA'
    if r['value_trap_risk']: return 'VALUE_TRAP_RISK'
    per,growth=r['per'],r['net_profit_growth_pct']
    hist_low=np.isfinite(r['per_percentile']) and r['per_percentile']<=30
    peer_low=np.isfinite(r['peer_per_median']) and np.isfinite(per) and per<r['peer_per_median']*.80
    garp=np.isfinite(r['peg_diagnostic']) and r['peg_diagnostic']<=1 and growth>0
    if garp and (hist_low or peer_low): return 'GARP_UNDERVALUATION_CANDIDATE'
    if hist_low or peer_low or (np.isfinite(per) and per<=10): return 'POTENTIAL_VALUE'
    if np.isfinite(per) and per>=25: return 'EXPENSIVE_MULTIPLE'
    if r['historical_status']=='INSUFFICIENT_PIT_HISTORY' and r['peer_status']=='UNKNOWN':
        return 'CURRENT_ONLY_CONTEXT'
    return 'FAIR_OR_CONTEXT_DEPENDENT'

def write_status(obj):
    (OUTDIR/'valuation_status.json').write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
    print(json.dumps(obj,indent=2,default=str))

def main():
    fs,m,man,ratio=load_fundamentals()
    if not fs_schema_ok(fs):
        write_status({'status':'INSUFFICIENT_PIT_DATA','engine_changed':False,
                      'reason':'Canonical financial_statements_pit.csv missing/empty or lacks publication_timestamp/pit_ready; raw fallback disabled.',
                      'publication_date_fallback':False,'raw_financial_fallback':False})
        return
    pit_tickers=sorted(set(fs['ticker']))
    p=load_prices(pit_tickers); sec=load_sector()
    if p.empty:
        write_status({'status':'INSUFFICIENT_DATA','engine_changed':False,'reason':'Price data unavailable'})
        return
    latest_date=p.date.max()
    analysis_ts_local,analysis_ts_utc=analysis_clock(latest_date)
    sector_lookup={}
    if not sec.empty and 'ticker' in sec.columns:
        for _,r in sec.drop_duplicates('ticker').iterrows():
            sector_lookup[str(r.ticker).upper()]=str(r.get('sector_name','') or '')
    rows=[]; hist_rows=[]
    for t,pg in p.groupby('ticker'):
        t=str(t).upper(); pg=pg.sort_values('date').copy()
        price_candidates=pg[pg['date']<=latest_date]
        if price_candidates.empty: continue
        price_row=price_candidates.iloc[-1]; price=num(price_row.close); price_date=pd.to_datetime(price_row.date)
        pit_ev=pit_evidence_for_ticker(fs,t,analysis_ts_utc)
        f_hist=extract_periodic(fs,t,analysis_ts_utc,ratio)
        obs=valuation_obs_for_ticker(pg,f_hist)
        hs=hist_stats(obs,latest_date)
        cm=current_metric_map(m,t,analysis_ts_utc)
        eps=num(obs.iloc[-1].eps) if not obs.empty and pd.notna(obs.iloc[-1].eps) else np.nan
        bvps=num(obs.iloc[-1].bvps) if not obs.empty and pd.notna(obs.iloc[-1].bvps) else np.nan
        per=price/eps if np.isfinite(eps) and eps>0 else np.nan
        pbv=price/bvps if np.isfinite(bvps) and bvps>0 else np.nan
        ey=1/per if np.isfinite(per) and per>0 else np.nan
        growth=next((num(cm[k]) for k in ['net_income_growth','net_profit_growth','net_profit_growth_pct']
                     if k in cm and np.isfinite(num(cm[k]))),np.nan)
        roe=next((num(cm[k]) for k in ['roe_annualized_estimate','roe_annualized']
                  if k in cm and np.isfinite(num(cm[k]))),np.nan)
        npm=num(cm.get('net_margin',np.nan)); revg=num(cm.get('revenue_growth',cm.get('revenue_growth_pct',np.nan)))
        peg=per/growth if np.isfinite(per) and np.isfinite(growth) and growth>0 else np.nan
        flags=[]
        if np.isfinite(eps) and eps<=0: flags.append('NEGATIVE_EPS')
        if np.isfinite(bvps) and bvps<=0: flags.append('NEGATIVE_BOOK_VALUE')
        if np.isfinite(growth) and growth<0: flags.append('EARNINGS_DECLINE')
        if np.isfinite(revg) and revg<0: flags.append('REVENUE_DECLINE')
        if np.isfinite(npm) and npm<0: flags.append('NEGATIVE_MARGIN')
        if np.isfinite(roe) and roe<0: flags.append('NEGATIVE_ROE')
        trap=(('NEGATIVE_EPS' in flags or 'NEGATIVE_BOOK_VALUE' in flags)
              and ('EARNINGS_DECLINE' in flags or 'REVENUE_DECLINE' in flags))
        if trap: flags.append('VALUE_TRAP_RISK')
        pit_period=pit_ev['pit_period_end']; pit_ts=pit_ev['pit_publication_timestamp']; pit_date=pit_ev['pit_publication_date']
        r={'analysis_date':latest_date.date().isoformat(),
           'analysis_timestamp':analysis_ts_local.isoformat(),
           'price_date':price_date.date().isoformat(),'ticker':t,'close':price,'eps':eps,'bvps':bvps,
           'per':per,'pbv':pbv,'earnings_yield':ey,'revenue_growth_pct':revg,
           'net_profit_growth_pct':growth,'roe_pct':roe,'net_margin_pct':npm,'peg_diagnostic':peg,
           'sector':sector_lookup.get(t,''),
           'pit_period_end':pit_period.date().isoformat() if pd.notna(pit_period) else '',
           'pit_publication_date':pit_date.date().isoformat() if pd.notna(pit_date) else '',
           'pit_publication_timestamp':pit_ts.isoformat() if pd.notna(pit_ts) else '',
           'pit_evidence_rows':int(pit_ev['pit_rows']),'pit_current_status':pit_ev['pit_status'],
           'value_trap_risk':bool(trap),'flags':';'.join(flags),**hs}
        r['peer_count']=0; r['peer_per_median']=np.nan; r['peer_pbv_median']=np.nan
        r['peer_growth_median']=np.nan; r['peer_roe_median']=np.nan; r['peer_status']='UNKNOWN'
        rows.append(r)
        if not obs.empty:
            o=obs.copy(); o['ticker']=t; hist_rows.append(o)
    snap=pd.DataFrame(rows)
    if snap.empty:
        write_status({'status':'INSUFFICIENT_DATA','engine_changed':False,'reason':'No latest-price rows'}); return
    snap=peer_stats(snap); snap['classification']=snap.apply(classify,axis=1)
    snap['historical_discount_vs_median_pct']=np.where(np.isfinite(snap.per)&np.isfinite(snap.per_hist_median),(1-snap.per/snap.per_hist_median)*100,np.nan)
    snap['peer_discount_vs_median_pct']=np.where(np.isfinite(snap.per)&np.isfinite(snap.peer_per_median),(1-snap.per/snap.peer_per_median)*100,np.nan)
    snap.to_csv(OUTDIR/'valuation_snapshot.csv',index=False)
    if hist_rows:
        pd.concat(hist_rows,ignore_index=True).to_csv(OUTDIR/'pit_valuation_observations.csv',index=False)
    else:
        pd.DataFrame(columns=['ticker','date','close','analysis_timestamp_utc','publication_timestamp','period_end','eps','bvps','per','pbv']).to_csv(OUTDIR/'pit_valuation_observations.csv',index=False)
    counts=snap.classification.value_counts(dropna=False).to_dict()
    verified=set(snap.loc[snap.pit_current_status.eq('PIT_VERIFIED'),'ticker'].astype(str).str.upper())
    ratio_eligible=0
    if not ratio.empty and {'publication_timestamp','pit_ready'}.issubset(ratio.columns):
        ratio_eligible=int((ratio['pit_ready'] & ratio['publication_timestamp'].notna()
                            & (ratio['publication_timestamp']<=analysis_ts_utc)).sum())
    status={'status':'BUILT','engine_changed':False,'analysis_date':latest_date.date().isoformat(),
            'analysis_timestamp':analysis_ts_local.isoformat(),'analysis_timezone':'Asia/Jakarta (+07:00)',
            'tickers':int(len(snap)),'financial_source':'financial_statements_pit.csv',
            'raw_financial_fallback':False,'publication_date_fallback':False,
            'ratio_snapshot_rows':int(len(ratio)),'ratio_timestamp_eligible_rows':ratio_eligible,
            'pit_verified':int((snap.pit_current_status=='PIT_VERIFIED').sum()),
            'pit_verified_tickers':sorted(verified),
            'historical_verified':int((snap.historical_status=='VERIFIED_PIT_RANGE').sum()),
            'peer_sets_available':int((snap.peer_status=='PEER_SET_AVAILABLE').sum()),
            'classification_counts':{str(k):int(v) for k,v in counts.items()},
            'notes':['publication_timestamp is authoritative; publication_date is metadata only and never a fallback.',
                     'PIT availability requires pit_ready=true and publication_timestamp <= analysis_timestamp.',
                     'Daily price dates are evaluated at 23:59:59 WIB for end-of-day valuation.',
                     'Raw financial-statements fallback is disabled; missing canonical PIT data fails closed.',
                     'Auxiliary metrics/ratio snapshots are usable only with their own valid publication_timestamp and pit_ready=true.',
                     'Valuation classification is research evidence, not a BUY/SELL signal or probability.']}
    write_status(status)

if __name__=='__main__': main()
