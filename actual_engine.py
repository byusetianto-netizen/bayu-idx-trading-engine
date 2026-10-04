import pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'idx_stock_prices.csv'
IHSG=ROOT/'data'/'idx_ihsg_index.csv'
OUT=ROOT/'data'/'engine_output'
OUT.mkdir(exist_ok=True)

df=pd.read_csv(DATA, parse_dates=['date'])
df=df.drop_duplicates(['ticker','date']).sort_values(['ticker','date']).reset_index(drop=True)
for c in ['open','high','low','close','volume']:
    df[c]=pd.to_numeric(df[c],errors='coerce')

# Clean IHSG: accept both the original Yahoo-style columns
# (Price/Close/Open/High/Low) and the updater's normalized columns
# (date/close/open/high/low).
idx=pd.read_csv(IHSG)
if 'Price' in idx.columns:
    idx=idx[pd.to_datetime(idx['Price'],errors='coerce').notna()].copy()
    idx=idx.rename(columns={
        'Price':'date','Open':'open','High':'high','Low':'low',
        'Close':'close','Volume':'volume'
    })
elif 'date' in idx.columns:
    idx=idx[pd.to_datetime(idx['date'],errors='coerce').notna()].copy()
else:
    raise ValueError("IHSG file has no recognized date column (expected 'Price' or 'date').")

# Normalize case for OHLC fields when necessary
rename_lower={}
for c in idx.columns:
    lc=str(c).lower()
    if lc in {'open','high','low','close','volume','date'} and c != lc:
        rename_lower[c]=lc
idx=idx.rename(columns=rename_lower)

idx['date']=pd.to_datetime(idx['date'],errors='coerce')
idx['close']=pd.to_numeric(idx['close'],errors='coerce')
idx=idx[['date','close']].dropna().drop_duplicates('date').sort_values('date')

# market returns
ih=idx.set_index('date')['close']
df['ihsg_close']=df['date'].map(ih)

# Features
G=df.groupby('ticker',group_keys=False)
for n in [5,10,20,60]:
    df[f'ret{n}']=G['close'].pct_change(n)
for n in [20,50,200]:
    ma=G['close'].transform(lambda s:s.rolling(n,min_periods=n).mean())
    df[f'ma{n}_dist']=df['close']/ma-1

df['vol20']=G['volume'].transform(lambda s:s.rolling(20,min_periods=20).mean())
df['vol_ratio']=df['volume']/df['vol20']
df['volatility20']=G['ret5'].transform(lambda s:s.rolling(4,min_periods=4).std())

delta=G['close'].diff()
gain=delta.clip(lower=0)
loss=-delta.clip(upper=0)
avg_gain=gain.groupby(df['ticker']).transform(lambda s:s.rolling(14,min_periods=14).mean())
avg_loss=loss.groupby(df['ticker']).transform(lambda s:s.rolling(14,min_periods=14).mean())
rs=avg_gain/avg_loss.replace(0,np.nan)
df['rsi']=100-(100/(1+rs))

df['ihsg_ret20']=df['date'].map(ih.pct_change(20))
df['rs20']=df['ret20']-df['ihsg_ret20']
df['liq20']=G.apply(lambda x:(x['close']*x['volume']).rolling(20,min_periods=20).median()).reset_index(level=0,drop=True)

features=['ret5','ret10','ret20','ret60','ma20_dist','ma50_dist','ma200_dist','vol_ratio','rsi','volatility20','rs20']

for target in [0.03,0.05,0.08]:
    hi=pd.concat([G['high'].shift(-i) for i in range(1,6)],axis=1).max(axis=1)
    lo=pd.concat([G['low'].shift(-i) for i in range(1,6)],axis=1).min(axis=1)
    df[f'y{int(target*100)}']=((hi>=df['close']*(1+target)) & (lo>df['close']*0.97)).astype(float)

train=(df['date']<'2023-01-01')
valid=(df['date']>='2023-01-01')&(df['date']<'2025-01-01')
latest_date=df['date'].max()
latest=df['date'].eq(latest_date)

metrics=[]
for t in [3,5,8]:
    X_train=df.loc[train,features]
    y=df.loc[train,f'y{t}']
    mask=y.notna() & X_train.notna().all(axis=1)

    pipe=make_pipeline(
        SimpleImputer(strategy='median'),
        HistGradientBoostingClassifier(
            max_iter=180,max_leaf_nodes=15,
            learning_rate=.06,l2_regularization=1.0,
            random_state=42
        )
    )
    pipe.fit(X_train.loc[mask],y.loc[mask].astype(int))

    X_val=df.loc[valid,features]
    vm=X_val.notna().all(axis=1)
    yy=df.loc[valid,f'y{t}']
    pred=pipe.predict_proba(X_val.loc[vm])[:,1]
    auc=roc_auc_score(yy.loc[vm],pred) if yy.loc[vm].nunique()>1 else np.nan
    metrics.append({'target':t,'validation_auc':auc,'train_n':int(mask.sum())})

    X_latest=df.loc[latest,features]
    valid_latest=X_latest.notna().all(axis=1)
    p=np.full(len(X_latest),np.nan)
    if valid_latest.any():
        p[valid_latest]=pipe.predict_proba(X_latest.loc[valid_latest])[:,1]
    df.loc[latest,f'p{t}']=p

ihdf=idx.copy()
ihdf['ma50']=ihdf.close.rolling(50).mean()
ihdf['ma200']=ihdf.close.rolling(200).mean()
last=ihdf.iloc[-1]
if last.close>last.ma200 and last.close>last.ma50:
    regime='BULL'
elif last.close<last.ma200 and last.close<last.ma50:
    regime='BEAR'
else:
    regime='NEUTRAL'

L=df.loc[latest].copy()
trend=(np.clip((L.ma20_dist+0.05)/0.20,0,1)*.35+
       np.clip((L.ma50_dist+0.10)/0.30,0,1)*.35+
       np.clip((L.ret20+0.15)/0.30,0,1)*.30)
mom=np.clip((L.rsi-35)/40,0,1)*.5+np.clip((L.ret10+0.10)/0.25,0,1)*.5
vol=np.clip(L.vol_ratio/2,0,1)
rs=np.clip((L.rs20+0.15)/0.30,0,1)
prob=.2*L.p3+.45*L.p5+.35*L.p8
liq=np.clip(np.log10(L.liq20.clip(lower=1))/12,0,1)
score=100*(.25*trend+.15*mom+.10*vol+.10*rs+.30*prob+.10*liq)

L['opportunity_score']=score
L['regime']=regime
counts=df.groupby('ticker').size()
L['history_n']=L.ticker.map(counts)
L['eligible']=(L.history_n>=200)&(L.liq20>=1e9)&(L.close>=100)&L[['p3','p5','p8']].notna().all(axis=1)
L=L[L.eligible].sort_values(['opportunity_score','p5'],ascending=False)

L['stop']=L.close*.97
L['target3']=L.close*1.03
L['target5']=L.close*1.05
L['target8']=L.close*1.08
L['risk_reward_5']=(L.target5-L.close)/(L.close-L.stop)
L['decision']=np.where(
    (L.opportunity_score>=65)&(L.p5>=.45)&(L.risk_reward_5>=1.5),
    'WATCH','NO TRADE'
)

cols=['date','ticker','close','opportunity_score','p3','p5','p8','rsi',
      'ret20','ma20_dist','ma50_dist','ma200_dist','vol_ratio','rs20',
      'liq20','regime','risk_reward_5','stop','target3','target5',
      'target8','decision']

L[cols].head(20).to_csv(OUT/'latest_actual_radar.csv',index=False)
pd.DataFrame(metrics).to_csv(OUT/'model_validation_metrics.csv',index=False)

with open(OUT/'ACTUAL_ENGINE_REPORT.md','w') as f:
    f.write(
        f'# Actual Engine Run\n\n'
        f'Latest data date: {latest_date.date()}\n\n'
        f'Market regime: {regime}\n\n'
        f'Eligible candidates: {len(L)}\n\n'
        f'Top 10:\n\n'+L[cols].head(10).to_markdown(index=False)+
        '\n\nIMPORTANT: This is an actual data-connected research run, not a production live signal. '
        'The universe remains incomplete (95 tickers), so survivorship bias is unresolved.\n'
    )

print('latest',latest_date.date(),'regime',regime,'eligible',len(L))
print(L[cols].head(10).to_string(index=False))
print(pd.DataFrame(metrics).to_string(index=False))
