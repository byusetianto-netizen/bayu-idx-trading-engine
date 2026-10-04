from pathlib import Path
import argparse, io, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; STOCKS=DATA/'idx_stock_prices.csv'; IHSG=DATA/'idx_ihsg_index.csv'; MANIFEST=DATA/'update_manifest.json'
FALLBACK_BASE='https://raw.githubusercontent.com/nofendian17/idx_dataset/main/data/stock_data_{date}.csv'
MIN_TICKER_COVERAGE=.85; MIN_ROW_COVERAGE=.70

def normalize_existing():
    df=pd.read_csv(STOCKS,parse_dates=['date']).drop_duplicates(['ticker','date']).sort_values(['ticker','date'])
    df['ticker']=df['ticker'].astype(str).str.strip()
    for c in ['open','high','low','close','volume']: df[c]=pd.to_numeric(df[c],errors='coerce')
    df.to_csv(STOCKS,index=False); return df

def fetch_one(ticker,start,end,retries=3):
    try: import yfinance as yf
    except Exception as e: return ticker,None,str(e)
    for a in range(retries):
        try:
            x=yf.download(f'{ticker}.JK',start=start,end=end,auto_adjust=True,progress=False,threads=False)
            if x is None or x.empty: raise ValueError('empty response')
            x=x.reset_index()
            if hasattr(x.columns,'levels'): x.columns=[c[0] if isinstance(c,tuple) else c for c in x.columns]
            x.columns=[str(c).lower().replace(' ','_') for c in x.columns]
            req=['date','open','high','low','close','volume']
            if not all(c in x.columns for c in req): raise ValueError('missing columns')
            x=x[req].copy(); x['ticker']=ticker; x['date']=pd.to_datetime(x['date'],errors='coerce').dt.tz_localize(None)
            for c in req[1:]: x[c]=pd.to_numeric(x[c],errors='coerce')
            x=x.dropna(subset=['date','close'])
            return ticker,x[['date','ticker','open','high','low','close','volume']],None
        except Exception as e:
            if a<retries-1: time.sleep(2**a)
            else: return ticker,None,str(e)

def fetch_yfinance(tickers,start,end,workers=4):
    chunks=[]; diag=[]
    with ThreadPoolExecutor(max_workers=max(1,min(workers,4))) as ex:
        fs={ex.submit(fetch_one,t,start,end):t for t in tickers}
        for f in as_completed(fs):
            t=fs[f]
            try: t,x,e=f.result()
            except Exception as z: x,e=None,str(z)
            if x is not None and not x.empty:
                chunks.append(x); diag.append({'ticker':t,'ok':True,'rows':len(x),'last_date':str(x.date.max().date()),'error':None})
            else: diag.append({'ticker':t,'ok':False,'rows':0,'last_date':None,'error':e})
    return (pd.concat(chunks,ignore_index=True) if chunks else None),sorted(diag,key=lambda z:z['ticker'])

def coverage(x,tickers,start,end):
    days=pd.bdate_range(start,end); expected=len(days)*len(tickers)
    if x is None or x.empty: return {'pass':False,'ticker_coverage':0,'row_coverage':0,'tickers_with_data':0,'expected_tickers':len(tickers),'rows_received':0,'expected_rows':expected,'latest_date':None}
    x=x.copy(); x['date']=pd.to_datetime(x['date'],errors='coerce'); x=x.dropna(subset=['date','ticker']).drop_duplicates(['ticker','date'])
    n=int(x.groupby('ticker').size().gt(0).sum()); rows=len(x); tc=n/len(tickers) if tickers else 0; rc=rows/expected if expected else 0
    return {'pass':tc>=MIN_TICKER_COVERAGE and rc>=MIN_ROW_COVERAGE,'ticker_coverage':round(tc,4),'row_coverage':round(rc,4),'tickers_with_data':n,'expected_tickers':len(tickers),'rows_received':rows,'expected_rows':expected,'latest_date':str(x.date.max().date())}

def fallback_day(d):
    try:
        r=requests.get(FALLBACK_BASE.format(date=d),timeout=30)
        if r.status_code!=200 or not r.text.strip(): return None
        x=pd.read_csv(io.StringIO(r.text)).rename(columns={'Date':'date','Stock Code':'ticker','Open Price':'open','High Price':'high','Low Price':'low','Last Price':'close','Volume':'volume'})
        req=['date','ticker','open','high','low','close','volume']
        if not all(c in x.columns for c in req): return None
        x=x[req].copy(); x['date']=pd.to_datetime(x.date,errors='coerce'); x['ticker']=x.ticker.astype(str).str.strip()
        for c in req[2:]: x[c]=pd.to_numeric(x[c],errors='coerce')
        return x.dropna(subset=['date','ticker','close'])
    except Exception: return None

def fallback(tickers,start,end,workers=8):
    days=pd.date_range(start,end,freq='B'); chunks=[]
    with ThreadPoolExecutor(max_workers=min(workers,8)) as ex:
        fs={ex.submit(fallback_day,d.strftime('%Y-%m-%d')):d for d in days}
        for f in as_completed(fs):
            try: x=f.result()
            except Exception: x=None
            if x is not None: chunks.append(x[x.ticker.isin(tickers)])
    return pd.concat(chunks,ignore_index=True) if chunks else None

def normalize_ihsg():
    ih=pd.read_csv(IHSG)
    if 'Price' in ih.columns:
        ih=ih[pd.to_datetime(ih.Price,errors='coerce').notna()].copy().rename(columns={'Price':'date','Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
    elif 'date' in ih.columns: ih=ih[pd.to_datetime(ih.date,errors='coerce').notna()].copy()
    else: raise ValueError("IHSG file has no recognized date column")
    ih['date']=pd.to_datetime(ih.date,errors='coerce'); ih['close']=pd.to_numeric(ih.close,errors='coerce')
    return ih.dropna(subset=['date','close']).drop_duplicates('date',keep='last').sort_values('date')

def update_ihsg(start,end):
    try:
        import yfinance as yf
        x=yf.download('^JKSE',start=start,end=end,auto_adjust=False,progress=False,threads=False).reset_index()
        if hasattr(x.columns,'levels'): x.columns=[c[0] if isinstance(c,tuple) else c for c in x.columns]
        x.columns=[str(c).lower().replace(' ','_') for c in x.columns]
        return pd.DataFrame({'date':pd.to_datetime(x.date),'open':x.open,'high':x.high,'low':x.low,'close':x.close,'volume':x.get('volume')}) if not x.empty else None
    except Exception: return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--start'); ap.add_argument('--end'); ap.add_argument('--workers',type=int,default=4); a=ap.parse_args()
    stocks=normalize_existing(); old=stocks.date.max().date(); start=a.start or (old+pd.Timedelta(days=1)).strftime('%Y-%m-%d'); end=a.end or pd.Timestamp.now(tz='Asia/Jakarta').date().strftime('%Y-%m-%d'); tickers=sorted(stocks.ticker.dropna().astype(str).unique())
    yf,diag=fetch_yfinance(tickers,start,end,a.workers); yc=coverage(yf,tickers,start,end); additions=yf; source='yfinance'; fallback_cov=None
    if not yc['pass']:
        fb=fallback(tickers,start,end,a.workers); fallback_cov=coverage(fb,tickers,start,end)
        if fb is not None and (yf is None or fallback_cov['row_coverage']>yc['row_coverage']): additions=fb; source='github-daily-snapshot'
        elif fb is not None and yf is not None: additions=pd.concat([yf,fb],ignore_index=True).drop_duplicates(['ticker','date']); source='yfinance+github-daily-snapshot'
        fc=coverage(additions,tickers,start,end)
        if not fc['pass']:
            result={'status':'FAILED_DATA_COVERAGE','source':source,'old_last_date':str(old),'requested_start':start,'requested_end':end,'ticker_count':len(tickers),'yfinance_coverage':yc,'fallback_coverage':fallback_cov,'final_coverage':fc,'message':'Update rejected; dataset was not overwritten.'}
            MANIFEST.write_text(json.dumps(result,indent=2)); raise RuntimeError(json.dumps(result,indent=2))
    if additions is None or additions.empty: raise RuntimeError('No validated stock data returned')
    combined=pd.concat([stocks,additions],ignore_index=True); combined['date']=pd.to_datetime(combined.date); combined=combined.drop_duplicates(['ticker','date'],keep='last').sort_values(['ticker','date']); combined.to_csv(STOCKS,index=False)
    ih=normalize_ihsg(); ihnew=update_ihsg(start,end)
    if ihnew is not None and not ihnew.empty: pd.concat([ih,ihnew],ignore_index=True).drop_duplicates('date',keep='last').sort_values('date').to_csv(IHSG,index=False)
    final=pd.read_csv(STOCKS,parse_dates=['date']); finalih=pd.read_csv(IHSG,parse_dates=['date']); latest=final.date.max().date(); latest_tickers=int(final.loc[final.date==pd.Timestamp(latest),'ticker'].nunique())
    result={'status':'SUCCESS','updated_at':pd.Timestamp.now(tz='Asia/Jakarta').isoformat(),'source':source,'old_last_date':str(old),'new_last_date':str(latest),'ihsg_last_date':str(finalih.date.max().date()),'requested_start':start,'requested_end':end,'ticker_count':len(tickers),'tickers_on_latest_stock_date':latest_tickers,'rows':len(final),'coverage':coverage(additions,tickers,start,end),'yfinance_diagnostics':diag}
    MANIFEST.write_text(json.dumps(result,indent=2,default=str)); print(json.dumps(result,indent=2,default=str))

if __name__=='__main__': main()
