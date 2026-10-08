#!/usr/bin/env python3
"""V2.2 Complete Valuation Engine.

Research-only. Builds current valuation, PIT historical valuation evidence,
peer-relative valuation, GARP diagnostics and value-trap flags.
Never changes actual_engine.py and never emits BUY/SELL/probability.
"""
from pathlib import Path
import json, math
import numpy as np
import pandas as pd

ROOT = Path('.')
PRICE = ROOT/'data/idx_stock_prices.csv'
FUND_DIR = ROOT/'data/fundamental'
METRICS = FUND_DIR/'fundamental_metrics.csv'
FS = FUND_DIR/'financial_statements.csv'
MANIFEST = FUND_DIR/'financial_source_manifest.csv'
SECTOR = ROOT/'data/sector/sector_map.csv'
OUTDIR = ROOT/'data/valuation'; OUTDIR.mkdir(parents=True, exist_ok=True)

ANALYSIS_DATE = None
MIN_HIST_OBS = 120
MIN_HIST_PERIODS = 2
MIN_PEERS = 3

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
    except Exception: return np.nan

def read_csv(path):
    if not path.exists(): return pd.DataFrame()
    try: return pd.read_csv(path)
    except Exception: return pd.DataFrame()

def normalize_cols(df):
    if df.empty: return df
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def load_prices():
    p = normalize_cols(read_csv(PRICE))
    need={'ticker','date','close'}
    if not need.issubset(p.columns): return pd.DataFrame()
    p['ticker']=p['ticker'].astype(str).str.upper().str.strip()
    p['date']=pd.to_datetime(p['date'], errors='coerce').dt.tz_localize(None)
    p['close']=pd.to_numeric(p['close'], errors='coerce')
    return p.dropna(subset=['ticker','date','close']).sort_values(['ticker','date'])

def load_fundamentals():
    fs=normalize_cols(read_csv(FS)); m=normalize_cols(read_csv(METRICS)); man=normalize_cols(read_csv(MANIFEST))
    for d in (fs,m,man):
        if not d.empty:
            for c in ('period_start','period_end','publication_date','publication_timestamp','report_date'):
                if c in d.columns: d[c]=pd.to_datetime(d[c], errors='coerce').dt.tz_localize(None)
            if 'ticker' in d.columns: d['ticker']=d['ticker'].astype(str).str.upper().str.strip()
    return fs,m,man

def load_sector():
    s=normalize_cols(read_csv(SECTOR))
    if 'ticker' in s.columns: s['ticker']=s['ticker'].astype(str).str.upper().str.strip()
    return s

def attach_publication(m, fs, man):
    """Attach publication evidence. Missing publication => NOT PIT usable."""
    m=m.copy()
    if m.empty: return m
    if 'publication_date' not in m.columns: m['publication_date']=pd.NaT
    if 'period_end' not in m.columns: m['period_end']=pd.NaT
    # 1) exact manifest evidence by ticker/period_end
    if not man.empty and {'ticker','period_end'}.issubset(man.columns):
        mm=man[['ticker','period_end']+[c for c in ['publication_timestamp','publication_date','publication_confidence','source_status'] if c in man.columns]].copy()
        if 'publication_timestamp' in mm.columns: mm['pub_from_manifest']=mm['publication_timestamp']
        elif 'publication_date' in mm.columns: mm['pub_from_manifest']=mm['publication_date']
        else: mm['pub_from_manifest']=pd.NaT
        mm=mm.sort_values('pub_from_manifest').drop_duplicates(['ticker','period_end'], keep='first')
        m=m.merge(mm[['ticker','period_end','pub_from_manifest','publication_confidence','source_status']], on=['ticker','period_end'], how='left')
        m['publication_date']=m['publication_date'].fillna(m['pub_from_manifest'])
        m=m.drop(columns=['pub_from_manifest'], errors='ignore')
    # 2) financial_statements publication date by exact ticker/period_end/document when possible
    if not fs.empty and {'ticker','period_end','publication_date'}.issubset(fs.columns):
        cols=['ticker','period_end','publication_date']
        if 'document_id' in fs.columns: cols.append('document_id')
        ff=fs[cols].dropna(subset=['publication_date']).copy()
        ff=ff.sort_values('publication_date').drop_duplicates(['ticker','period_end'], keep='first')
        ff=ff.rename(columns={'publication_date':'pub_from_fs'})
        m=m.merge(ff[['ticker','period_end','pub_from_fs']], on=['ticker','period_end'], how='left')
        m['publication_date']=m['publication_date'].fillna(m['pub_from_fs'])
        m=m.drop(columns=['pub_from_fs'], errors='ignore')
    return m

def metric_rows(fs, ticker, names):
    if fs.empty or 'ticker' not in fs.columns: return fs.iloc[0:0]
    q=fs[fs['ticker'].eq(ticker)].copy()
    if 'metric_key' in q.columns:
        keys=q['metric_key'].astype(str).str.lower()
        aliases={x.lower() for x in names}
        q=q[keys.isin(aliases)]
    elif 'metric' in q.columns:
        keys=q['metric'].astype(str).str.lower()
        aliases={x.lower() for x in names}
        q=q[keys.isin(aliases)]
    return q

def extract_periodic(fs, ticker):
    """Return PIT-friendly fundamental observations. Prefer direct EPS/BVPS.
    If BVPS absent, derive only when explicit equity and share count exist in the
    same reporting period; otherwise leave BVPS unknown rather than inventing it.
    """
    if fs.empty: return pd.DataFrame()
    q=fs[fs['ticker'].eq(ticker)].copy()
    if 'value' not in q.columns or 'period_end' not in q.columns: return pd.DataFrame()
    q['value']=pd.to_numeric(q['value'], errors='coerce')
    q=q.dropna(subset=['value','period_end'])
    out=[]
    groups=q.groupby(['period_end'], dropna=True)
    for pe,g in groups:
        pub=pd.to_datetime(g.get('publication_date', pd.Series(dtype='datetime64[ns]')), errors='coerce').min() if 'publication_date' in g else pd.NaT
        if pd.isna(pub): continue
        def pick(names):
            if 'metric_key' in g.columns:
                z=g[g.metric_key.astype(str).str.lower().isin({x.lower() for x in names})]
            else:
                z=g[g.metric.astype(str).str.lower().isin({x.lower() for x in names})] if 'metric' in g else g.iloc[0:0]
            if z.empty: return np.nan
            return num(z.iloc[0]['value'])
        eps=pick(ALIASES['eps'])
        bvps=pick(ALIASES['bvps'])
        # Explicit derivation, only if units are plausibly compatible.
        if not np.isfinite(bvps):
            eq=pick(ALIASES['equity']); sh=pick(ALIASES['shares'])
            if np.isfinite(eq) and np.isfinite(sh) and sh>0:
                # Most IDX normalized financial values are in reporting currency thousands
                # while shares are absolute. We cannot safely assume unit compatibility.
                # Only derive when source unit explicitly indicates per-share or matching units.
                unit_eq=' '.join(g[g['metric_key'].astype(str).str.lower().isin({x.lower() for x in ALIASES['equity']})]['unit'].astype(str).tolist()).lower() if 'metric_key' in g.columns and 'unit' in g.columns else ''
                if 'share' in unit_eq or 'per share' in unit_eq:
                    bvps=eq/sh
        out.append({'ticker':ticker,'period_end':pe,'publication_date':pub,'eps':eps,'bvps':bvps})
    return pd.DataFrame(out)

def current_metric_map(m, ticker, analysis_date):
    if m.empty or 'ticker' not in m.columns or 'metric' not in m.columns or 'value' not in m.columns: return {}
    q=m[(m.ticker==ticker) & (m.publication_date.notna()) & (m.publication_date<=analysis_date)].copy()
    if q.empty: return {}
    q['value']=pd.to_numeric(q['value'], errors='coerce')
    res={}
    for name in ['revenue_growth','revenue_growth_pct','net_income_growth','net_profit_growth','net_profit_growth_pct','roe_annualized_estimate','roe_annualized','net_margin']:
        z=q[q.metric.astype(str).str.lower()==name.lower()].sort_values(['period_end','publication_date'])
        if not z.empty: res[name]=num(z.iloc[-1].value)
    return res

def valuation_obs_for_ticker(prices, fund_hist):
    if prices.empty or fund_hist.empty: return pd.DataFrame()
    pp=prices[['date','close']].sort_values('date').copy()
    ff=fund_hist[['publication_date','period_end','eps','bvps']].dropna(subset=['publication_date']).sort_values('publication_date').copy()
    # A valuation on date D may only use a filing published on/before D.
    z=pd.merge_asof(pp, ff, left_on='date', right_on='publication_date', direction='backward')
    z['per']=np.where((z.eps>0)&np.isfinite(z.eps), z.close/z.eps, np.nan)
    z['pbv']=np.where((z.bvps>0)&np.isfinite(z.bvps), z.close/z.bvps, np.nan)
    z=z.dropna(subset=['date'])
    return z

def hist_stats(obs, current_date):
    result={
        'historical_status':'UNKNOWN','historical_obs':0,'historical_periods':0,
        'per_hist_median':np.nan,'per_hist_p10':np.nan,'per_hist_p90':np.nan,'per_percentile':np.nan,
        'pbv_hist_median':np.nan,'pbv_hist_p10':np.nan,'pbv_hist_p90':np.nan,'pbv_percentile':np.nan,
    }
    if obs.empty: return result
    result['historical_obs']=int(len(obs)); result['historical_periods']=int(obs['period_end'].nunique())
    valid_periods=result['historical_periods']>=MIN_HIST_PERIODS
    if valid_periods and len(obs)>=MIN_HIST_OBS:
        result['historical_status']='VERIFIED_PIT_RANGE'
        for metric in ['per','pbv']:
            a=pd.to_numeric(obs[metric], errors='coerce').dropna()
            if len(a)>=MIN_HIST_OBS:
                cur=num(a.iloc[-1]); result[f'{metric}_hist_median']=float(a.median()); result[f'{metric}_hist_p10']=float(a.quantile(.10)); result[f'{metric}_hist_p90']=float(a.quantile(.90))
                if np.isfinite(cur): result[f'{metric}_percentile']=float((a<=cur).mean()*100)
    else:
        result['historical_status']='INSUFFICIENT_PIT_HISTORY'
    return result

def peer_stats(snapshot):
    if snapshot.empty or 'sector' not in snapshot.columns: return snapshot
    out=snapshot.copy()
    out['peer_count']=0; out['peer_per_median']=np.nan; out['peer_pbv_median']=np.nan; out['peer_growth_median']=np.nan; out['peer_roe_median']=np.nan; out['peer_status']='UNKNOWN'
    for idx,r in out.iterrows():
        sec=str(r.get('sector','')).strip()
        if not sec or sec.lower() in {'nan','none'}: continue
        peers=out[(out['sector'].astype(str).str.lower()==sec.lower()) & (out['ticker']!=r['ticker'])]
        peers=peers[(peers['pit_current_status']=='PIT_VERIFIED')]
        out.at[idx,'peer_count']=len(peers)
        if len(peers)<MIN_PEERS: continue
        for c in ['per','pbv','net_profit_growth_pct','roe_pct']:
            vals=pd.to_numeric(peers[c],errors='coerce').dropna()
            if len(vals)>=MIN_PEERS:
                target={'per':'peer_per_median','pbv':'peer_pbv_median','net_profit_growth_pct':'peer_growth_median','roe_pct':'peer_roe_median'}[c]
                out.at[idx,target]=float(vals.median())
        out.at[idx,'peer_status']='PEER_SET_AVAILABLE'
    return out

def classify(r):
    # Hard data gates first.
    if r['pit_current_status']!='PIT_VERIFIED': return 'INSUFFICIENT_PIT_DATA'
    if r['value_trap_risk']: return 'VALUE_TRAP_RISK'
    per,pbv,growth= r['per'],r['pbv'],r['net_profit_growth_pct']
    hp=r['historical_status']; ps=r['peer_status']
    hist_low = np.isfinite(r['per_percentile']) and r['per_percentile']<=30
    peer_low = np.isfinite(r['peer_per_median']) and np.isfinite(per) and per < r['peer_per_median']*0.80
    garp = np.isfinite(r['peg_diagnostic']) and r['peg_diagnostic']<=1 and growth>0
    if garp and (hist_low or peer_low): return 'GARP_UNDERVALUATION_CANDIDATE'
    if hist_low or peer_low or (np.isfinite(per) and per<=10): return 'POTENTIAL_VALUE'
    if np.isfinite(per) and per>=25: return 'EXPENSIVE_MULTIPLE'
    if hp=='INSUFFICIENT_PIT_HISTORY' and ps=='UNKNOWN': return 'CURRENT_ONLY_CONTEXT'
    return 'FAIR_OR_CONTEXT_DEPENDENT'

def main():
    p=load_prices(); fs,m,man=load_fundamentals(); sec=load_sector()
    if p.empty:
        status={'status':'INSUFFICIENT_DATA','engine_changed':False,'reason':'Price data unavailable'}
        (OUTDIR/'valuation_status.json').write_text(json.dumps(status,indent=2)); return
    latest_date=p.date.max(); global ANALYSIS_DATE; ANALYSIS_DATE=latest_date
    m=attach_publication(m,fs,man)
    # sector map
    sector_lookup={}
    if not sec.empty and 'ticker' in sec.columns:
        for _,r in sec.drop_duplicates('ticker').iterrows(): sector_lookup[str(r.ticker).upper()]=str(r.get('sector_name','') or '')
    rows=[]; hist_rows=[]
    for t,pg in p.groupby('ticker'):
        t=str(t).upper(); price_row=pg[pg.date==latest_date]
        if price_row.empty: continue
        price=num(price_row.iloc[-1].close)
        f_hist=extract_periodic(fs,t)
        # If direct EPS/BVPS are unavailable in raw statements, use derived metrics only for current snapshot.
        obs=valuation_obs_for_ticker(pg,f_hist)
        hs=hist_stats(obs,latest_date)
        cm=current_metric_map(m,t,latest_date)
        eps=num(obs.iloc[-1].eps) if not obs.empty and pd.notna(obs.iloc[-1].eps) else np.nan
        bvps=num(obs.iloc[-1].bvps) if not obs.empty and pd.notna(obs.iloc[-1].bvps) else np.nan
        # Match the latest PIT financial period explicitly.
        pit_pub=pd.NaT; pit_period=pd.NaT
        if not f_hist.empty:
            q=f_hist[f_hist.publication_date<=latest_date].sort_values(['period_end','publication_date'])
            if not q.empty:
                rr=q.iloc[-1]; pit_pub=rr.publication_date; pit_period=rr.period_end
        per=price/eps if np.isfinite(eps) and eps>0 else np.nan
        pbv=price/bvps if np.isfinite(bvps) and bvps>0 else np.nan
        ey=1/per if np.isfinite(per) and per>0 else np.nan
        growth=next((num(cm[k]) for k in ['net_income_growth','net_profit_growth','net_profit_growth_pct'] if k in cm and np.isfinite(num(cm[k]))),np.nan)
        roe=next((num(cm[k]) for k in ['roe_annualized_estimate','roe_annualized'] if k in cm and np.isfinite(num(cm[k]))),np.nan)
        npm=num(cm.get('net_margin',np.nan))
        revg=num(cm.get('revenue_growth',cm.get('revenue_growth_pct',np.nan)))
        peg=per/growth if np.isfinite(per) and np.isfinite(growth) and growth>0 else np.nan
        flags=[]
        if np.isfinite(eps) and eps<=0: flags.append('NEGATIVE_EPS')
        if np.isfinite(bvps) and bvps<=0: flags.append('NEGATIVE_BOOK_VALUE')
        if np.isfinite(growth) and growth<0: flags.append('EARNINGS_DECLINE')
        if np.isfinite(revg) and revg<0: flags.append('REVENUE_DECLINE')
        if np.isfinite(npm) and npm<0: flags.append('NEGATIVE_MARGIN')
        if np.isfinite(roe) and roe<0: flags.append('NEGATIVE_ROE')
        trap=(('NEGATIVE_EPS' in flags or 'NEGATIVE_BOOK_VALUE' in flags) and (('EARNINGS_DECLINE' in flags) or ('REVENUE_DECLINE' in flags)))
        if trap: flags.append('VALUE_TRAP_RISK')
        pit_status='PIT_VERIFIED' if pd.notna(pit_pub) else 'PIT_UNKNOWN'
        r={'analysis_date':latest_date.date().isoformat(),'ticker':t,'close':price,'eps':eps,'bvps':bvps,'per':per,'pbv':pbv,'earnings_yield':ey,
           'revenue_growth_pct':revg,'net_profit_growth_pct':growth,'roe_pct':roe,'net_margin_pct':npm,'peg_diagnostic':peg,
           'sector':sector_lookup.get(t,''),'pit_period_end':pit_period.date().isoformat() if pd.notna(pit_period) else '',
           'pit_publication_date':pit_pub.date().isoformat() if pd.notna(pit_pub) else '','pit_current_status':pit_status,
           'value_trap_risk':bool(trap),'flags':';'.join(flags),**hs}
        r['peer_count']=0; r['peer_per_median']=np.nan; r['peer_pbv_median']=np.nan; r['peer_growth_median']=np.nan; r['peer_roe_median']=np.nan; r['peer_status']='UNKNOWN'
        rows.append(r)
        if not obs.empty:
            o=obs.copy(); o['ticker']=t; hist_rows.append(o)
    snap=pd.DataFrame(rows)
    if snap.empty:
        status={'status':'INSUFFICIENT_DATA','engine_changed':False,'reason':'No latest-price rows'}
        (OUTDIR/'valuation_status.json').write_text(json.dumps(status,indent=2)); return
    snap=peer_stats(snap)
    snap['classification']=snap.apply(classify,axis=1)
    snap['historical_discount_vs_median_pct']=np.where(np.isfinite(snap.per)&np.isfinite(snap.per_hist_median), (1-snap.per/snap.per_hist_median)*100, np.nan)
    snap['peer_discount_vs_median_pct']=np.where(np.isfinite(snap.per)&np.isfinite(snap.peer_per_median), (1-snap.per/snap.peer_per_median)*100, np.nan)
    snap.to_csv(OUTDIR/'valuation_snapshot.csv',index=False)
    if hist_rows:
        pd.concat(hist_rows,ignore_index=True).to_csv(OUTDIR/'pit_valuation_observations.csv',index=False)
    else:
        pd.DataFrame(columns=['ticker','date','close','publication_date','period_end','eps','bvps','per','pbv']).to_csv(OUTDIR/'pit_valuation_observations.csv',index=False)
    counts=snap.classification.value_counts(dropna=False).to_dict()
    status={'status':'BUILT','engine_changed':False,'analysis_date':latest_date.date().isoformat(),'tickers':int(len(snap)),
            'pit_verified':int((snap.pit_current_status=='PIT_VERIFIED').sum()),
            'historical_verified':int((snap.historical_status=='VERIFIED_PIT_RANGE').sum()),
            'peer_sets_available':int((snap.peer_status=='PEER_SET_AVAILABLE').sum()),
            'classification_counts':{str(k):int(v) for k,v in counts.items()},
            'notes':['Current valuation uses only fundamental evidence published on/before analysis date.',
                     'Historical bands require at least 120 price observations and 2 distinct PIT financial periods.',
                     'Peer comparison requires at least 3 PIT-verified peers in the same supplied sector grouping.',
                     'Missing PIT evidence remains UNKNOWN; no current value is backfilled into historical dates.',
                     'Valuation classification is evidence, not a BUY/SELL signal or probability.']}
    (OUTDIR/'valuation_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status,indent=2))

if __name__=='__main__': main()
