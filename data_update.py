"""Incrementally update the research dataset.

Primary source: Yahoo Finance via yfinance for the existing 95-ticker universe.
Fallback source: daily all-stock snapshot published by nofendian17/idx_dataset.
The updater only appends dates after the local last date and validates the result.
Research use only; this is not an official IDX feed.
"""
from pathlib import Path
import argparse, io, json
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
STOCKS=DATA/'idx_stock_prices.csv'
IHSG=DATA/'idx_ihsg_index.csv'
MANIFEST=DATA/'update_manifest.json'
FALLBACK_BASE='https://raw.githubusercontent.com/nofendian17/idx_dataset/main/data/stock_data_{date}.csv'

def normalize_existing():
    df=pd.read_csv(STOCKS,parse_dates=['date'])
    df=df.drop_duplicates(['ticker','date']).sort_values(['ticker','date'])
    for c in ['open','high','low','close','volume']:
        df[c]=pd.to_numeric(df[c],errors='coerce')
    df.to_csv(STOCKS,index=False)
    return df

def fetch_yfinance(tickers,start,end):
    try:
        import yfinance as yf
    except Exception:
        return None
    symbols=[f'{t}.JK' for t in tickers]
    raw=yf.download(symbols,start=start,end=end,auto_adjust=True,progress=False,group_by='ticker',threads=True)
    if raw is None or raw.empty:
        return None
    rows=[]
    for t in tickers:
        sym=f'{t}.JK'
        try: x=raw[sym].copy()
        except Exception: continue
        x=x.reset_index()
        x.columns=[str(c).lower().replace(' ','_') for c in x.columns]
        if 'date' not in x.columns: continue
        x['ticker']=t
        keep={'date':'date','open':'open','high':'high','low':'low','close':'close','volume':'volume'}
        if not all(c in x.columns for c in keep): continue
        rows.append(x[list(keep.keys())].rename(columns=keep))
    return pd.concat(rows,ignore_index=True) if rows else None

def fetch_fallback_day(date):
    url=FALLBACK_BASE.format(date=date)
    r=requests.get(url,timeout=30)
    if r.status_code!=200 or not r.text.strip(): return None,r.status_code
    x=pd.read_csv(io.StringIO(r.text))
    x=x.rename(columns={'Date':'date','Stock Code':'ticker','Open Price':'open',
                       'High Price':'high','Low Price':'low','Last Price':'close','Volume':'volume'})
    x=x[['date','ticker','open','high','low','close','volume']]
    x['date']=pd.to_datetime(x['date'])
    return x,r.status_code

def update_fallback(tickers,start,end,workers=8):
    days=pd.date_range(start,end,freq='B')
    chunks=[]; found=[]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs={ex.submit(fetch_fallback_day,d.strftime('%Y-%m-%d')):d for d in days}
        for fut in as_completed(futs):
            d=futs[fut]
            try: x,status=fut.result()
            except Exception: x,status=None,None
            if x is not None:
                x=x[x.ticker.isin(tickers)]
                chunks.append(x); found.append(d.strftime('%Y-%m-%d'))
    return pd.concat(chunks,ignore_index=True) if chunks else None,sorted(found)

def update_ihsg(start,end):
    try:
        import yfinance as yf
        x=yf.download('^JKSE',start=start,end=end,auto_adjust=False,progress=False)
        if x is None or x.empty: return None
        x=x.reset_index()
        if hasattr(x.columns,'levels'):
            x.columns=[c[0] if isinstance(c,tuple) else c for c in x.columns]
        x.columns=[str(c).lower().replace(' ','_') for c in x.columns]
        return pd.DataFrame({
            'date':pd.to_datetime(x['date']),
            'open':x['open'],'high':x['high'],'low':x['low'],
            'close':x['close'],'volume':x.get('volume')
        })
    except Exception:
        return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--start',default=None)
    ap.add_argument('--end',default=None)
    ap.add_argument('--workers',type=int,default=8)
    args=ap.parse_args()

    stocks=normalize_existing()
    old_last=stocks.date.max().date()
    start=args.start or (old_last+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    end=args.end or pd.Timestamp.today(tz='Asia/Jakarta').date().strftime('%Y-%m-%d')
    tickers=sorted(stocks.ticker.dropna().unique())

    additions=fetch_yfinance(tickers,start,end)
    source='yfinance'
    if additions is None or additions.empty:
        additions,days=update_fallback(tickers,start,end,args.workers)
        source='github-daily-snapshot'
    else:
        days=sorted(pd.to_datetime(additions.date).dt.strftime('%Y-%m-%d').unique())

    if additions is not None and not additions.empty:
        combined=pd.concat([stocks,additions],ignore_index=True)
        combined['date']=pd.to_datetime(combined.date)
        combined=combined.drop_duplicates(['ticker','date'],keep='last').sort_values(['ticker','date'])
        combined.to_csv(STOCKS,index=False)
    else:
        days=[]

    ih=pd.read_csv(IHSG)
    if 'Price' in ih.columns:
        ih=ih[pd.to_datetime(ih['Price'],errors='coerce').notna()].copy()
        ih=ih.rename(columns={'Price':'date','Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
    elif 'date' in ih.columns:
        ih=ih[pd.to_datetime(ih['date'],errors='coerce').notna()].copy()
    else:
        raise ValueError("IHSG file has no recognized date column (expected 'Price' or 'date').")

    ih['date']=pd.to_datetime(ih['date'])
    ihnew=update_ihsg(start,end)
    if ihnew is not None and not ihnew.empty:
        ih=pd.concat([ih,ihnew],ignore_index=True).drop_duplicates('date',keep='last').sort_values('date')
        ih.to_csv(IHSG,index=False)

    final_stock=pd.read_csv(STOCKS,parse_dates=['date'])
    final_ihsg=pd.read_csv(IHSG,parse_dates=['date'])
    new_last_stock=final_stock.date.max().date()
    new_last_ihsg=final_ihsg.date.max().date()

    result={
        'updated_at':pd.Timestamp.now(tz='Asia/Jakarta').isoformat(),
        'source':source,
        'old_last_date':str(old_last),
        'new_last_date':str(new_last_stock),
        'ihsg_last_date':str(new_last_ihsg),
        'requested_start':start,
        'requested_end':end,
        'new_days':days,
        'ticker_count':len(tickers),
        'rows':int(len(final_stock))
    }
    MANIFEST.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
