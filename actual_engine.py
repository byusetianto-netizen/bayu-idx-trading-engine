from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'idx_stock_prices.csv'
IHSG = ROOT / 'data' / 'idx_ihsg_index.csv'
OUT = ROOT / 'data' / 'engine_output'
OUT.mkdir(exist_ok=True)

TARGETS = [0.03, 0.05, 0.08]
MAX_HOLD = 5
STOP_ATR_MULT = 1.5
STRUCTURE_BUFFER = 0.005
MIN_HISTORY = 200
MIN_LIQ = 1e9
MIN_PRICE = 100


def clean_ihsg(path):
    idx = pd.read_csv(path)
    if 'Price' in idx.columns:
        idx['date'] = pd.to_datetime(idx['Price'], errors='coerce')
        idx['close'] = pd.to_numeric(idx['Close'], errors='coerce')
    elif 'date' in idx.columns:
        idx['date'] = pd.to_datetime(idx['date'], errors='coerce')
        idx['close'] = pd.to_numeric(idx['close'], errors='coerce')
    else:
        raise ValueError('IHSG file has no recognized date column')
    return idx[['date','close']].dropna().drop_duplicates('date').sort_values('date').reset_index(drop=True)


def dynamic_levels(entry, atr14, swing_low20):
    if not np.isfinite(entry) or not np.isfinite(atr14) or atr14 <= 0:
        return np.nan, np.nan, np.nan, np.nan
    atr_stop = entry - STOP_ATR_MULT * atr14
    structure_stop = swing_low20 * (1 - STRUCTURE_BUFFER) if np.isfinite(swing_low20) else np.nan
    # Use the higher (closer) valid stop to avoid an oversized risk distance.
    stop = max(atr_stop, structure_stop) if np.isfinite(structure_stop) else atr_stop
    stop = min(stop, entry * 0.995)
    risk = entry - stop
    if risk <= 0:
        return np.nan, np.nan, np.nan, np.nan
    atr_pct = atr14 / entry
    target_pct = float(np.clip(2.0 * risk / entry, 0.03, 0.08))
    target = entry * (1 + target_pct)
    rr = (target - entry) / risk
    return stop, target, rr, target_pct


df = pd.read_csv(DATA, parse_dates=['date'])
df = df.drop_duplicates(['ticker','date']).sort_values(['ticker','date']).reset_index(drop=True)
for c in ['open','high','low','close','volume']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

idx = clean_ihsg(IHSG)
ih = idx.set_index('date')['close']
df['ihsg_close'] = df['date'].map(ih)
G = df.groupby('ticker', group_keys=False)

for n in [5,10,20,60]:
    df[f'ret{n}'] = G['close'].pct_change(n)
for n in [20,50,200]:
    ma = G['close'].transform(lambda s: s.rolling(n, min_periods=n).mean())
    df[f'ma{n}_dist'] = df['close'] / ma - 1

df['vol20'] = G['volume'].transform(lambda s: s.rolling(20, min_periods=20).mean())
df['vol_ratio'] = df['volume'] / df['vol20']

prev_close = G['close'].shift(1)
tr = pd.concat([
    df['high'] - df['low'],
    (df['high'] - prev_close).abs(),
    (df['low'] - prev_close).abs()
], axis=1).max(axis=1)
df['atr14'] = tr.groupby(df['ticker']).transform(lambda s: s.rolling(14, min_periods=14).mean())
df['atr_pct'] = df['atr14'] / df['close']
df['swing_low20'] = G['low'].transform(lambda s: s.rolling(20, min_periods=20).min())
daily_ret = G['close'].pct_change()
df['volatility20'] = daily_ret.groupby(df['ticker']).transform(lambda s: s.rolling(20, min_periods=20).std())

delta = G['close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.groupby(df['ticker']).transform(lambda s: s.rolling(14, min_periods=14).mean())
avg_loss = loss.groupby(df['ticker']).transform(lambda s: s.rolling(14, min_periods=14).mean())
rs14 = avg_gain / avg_loss.replace(0, np.nan)
df['rsi'] = 100 - (100 / (1 + rs14))

df['ihsg_ret20'] = df['date'].map(ih.pct_change(20))
df['rs20'] = df['ret20'] - df['ihsg_ret20']
df['liq20'] = (df['close'] * df['volume']).groupby(df['ticker']).transform(lambda s: s.rolling(20, min_periods=20).median())

levels = [dynamic_levels(e,a,s) for e,a,s in zip(df['close'],df['atr14'],df['swing_low20'])]
lev = pd.DataFrame(levels, columns=['signal_stop','signal_target','signal_rr','signal_target_pct'], index=df.index)
df = pd.concat([df,lev],axis=1)

features = ['ret5','ret10','ret20','ret60','ma20_dist','ma50_dist','ma200_dist','vol_ratio','rsi','volatility20','atr_pct','rs20','signal_rr']

# Labels use the next trading-session OPEN as entry. No future information enters features.
for target in TARGETS:
    labels = np.full(len(df), np.nan)
    for ticker,g in df.groupby('ticker', sort=False):
        ix = g.index.to_numpy()
        opens = g['open'].to_numpy(float)
        highs = g['high'].to_numpy(float)
        lows = g['low'].to_numpy(float)
        stops = g['signal_stop'].to_numpy(float)
        for j in range(len(g)-MAX_HOLD):
            entry = opens[j+1]
            stop = stops[j]
            if not np.isfinite(entry) or not np.isfinite(stop) or stop >= entry:
                continue
            target_px = entry * (1+target)
            result = 0.0
            for k in range(j+1, min(j+1+MAX_HOLD,len(g))):
                hit_target = highs[k] >= target_px
                hit_stop = lows[k] <= stop
                if hit_target and hit_stop:
                    result = 0.0
                    break
                if hit_target:
                    result = 1.0
                    break
                if hit_stop:
                    result = 0.0
                    break
            labels[ix[j]] = result
    df[f'y{int(target*100)}'] = labels

train = df['date'] < '2023-01-01'
valid = (df['date'] >= '2023-01-01') & (df['date'] < '2025-01-01')
latest_date = df['date'].max()
latest = df['date'].eq(latest_date)
metrics=[]

for t in [3,5,8]:
    X_train=df.loc[train,features]
    y=df.loc[train,f'y{t}']
    mask=y.notna() & X_train.notna().all(axis=1)
    pipe=make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,learning_rate=.06,l2_regularization=1.0,random_state=42))
    pipe.fit(X_train.loc[mask],y.loc[mask].astype(int))
    X_val=df.loc[valid,features]; yy=df.loc[valid,f'y{t}']
    vm=yy.notna() & X_val.notna().all(axis=1)
    pred=pipe.predict_proba(X_val.loc[vm])[:,1]
    auc=roc_auc_score(yy.loc[vm],pred) if yy.loc[vm].nunique()>1 else np.nan
    metrics.append({'target':t,'validation_auc':auc,'train_n':int(mask.sum()),'validation_n':int(vm.sum())})
    lm=latest & df[features].notna().all(axis=1)
    df.loc[lm,f'p{t}']=pipe.predict_proba(df.loc[lm,features])[:,1]

ihdf=idx.copy(); ihdf['ma50']=ihdf.close.rolling(50).mean(); ihdf['ma200']=ihdf.close.rolling(200).mean(); last=ihdf.iloc[-1]
if last.close>last.ma200 and last.close>last.ma50: regime='BULL'
elif last.close<last.ma200 and last.close<last.ma50: regime='BEAR'
else: regime='NEUTRAL'

L=df.loc[latest].copy()
trend=np.clip((L.ma20_dist+.05)/.20,0,1)*.35+np.clip((L.ma50_dist+.10)/.30,0,1)*.35+np.clip((L.ret20+.15)/.30,0,1)*.30
mom=np.clip((L.rsi-35)/40,0,1)*.5+np.clip((L.ret10+.10)/.25,0,1)*.5
vol=np.clip(L.vol_ratio/2,0,1); rs=np.clip((L.rs20+.15)/.30,0,1); prob=.20*L.p3+.45*L.p5+.35*L.p8; liq=np.clip(np.log10(L.liq20.clip(lower=1))/12,0,1)
L['opportunity_score']=100*(.25*trend+.15*mom+.10*vol+.10*rs+.30*prob+.10*liq); L['regime']=regime
counts=df.groupby('ticker').size(); L['history_n']=L.ticker.map(counts)
L['eligible']=(L.history_n>=MIN_HISTORY)&(L.liq20>=MIN_LIQ)&(L.close>=MIN_PRICE)&L[['p3','p5','p8']].notna().all(axis=1)&L['signal_stop'].notna()&L['signal_target'].notna()
L=L[L.eligible].copy()
L['reference_entry']=L['close']; L['entry_status']='NEXT_SESSION_OPEN_PENDING'; L['stop']=L['signal_stop']; L['target']=L['signal_target']; L['target_pct']=L['signal_target_pct']; L['risk_pct']=(L.reference_entry-L.stop)/L.reference_entry; L['risk_reward']=L['signal_rr']
L['target3']=L.reference_entry*1.03; L['target5']=L.reference_entry*1.05; L['target8']=L.reference_entry*1.08
L['decision']=np.where((L.opportunity_score>=65)&(L.p5>=.45)&(L.risk_reward>=1.5),'WATCH','NO TRADE')
L=L.sort_values(['opportunity_score','p5'],ascending=False)

cols=['date','ticker','close','reference_entry','entry_status','opportunity_score','p3','p5','p8','rsi','ret20','ma20_dist','ma50_dist','ma200_dist','vol_ratio','atr14','atr_pct','swing_low20','rs20','liq20','regime','stop','target','target_pct','risk_pct','risk_reward','target3','target5','target8','decision']
L[cols].head(20).to_csv(OUT/'latest_actual_radar.csv',index=False)
pd.DataFrame(metrics).to_csv(OUT/'model_validation_metrics.csv',index=False)
report=f'''# Actual Engine V1.1 Run\n\nLatest data date: {latest_date.date()}\nMarket regime: {regime}\nEligible candidates: {len(L)}\n\n## Risk / execution model\n- Historical entry: next trading-session OPEN.\n- Latest radar: last close is only a reference entry; actual next open is pending.\n- Stop: dynamic using 1.5 ATR(14) and 20D structure.\n- Target: volatility-aware, 3%–8%.\n- R/R: dynamic; no constant 1.6667.\n- Same-bar target + stop: failure.\n\n## Validation\n{pd.DataFrame(metrics).to_markdown(index=False)}\n\nIMPORTANT: Research/paper trading only. Universe remains incomplete (95 tickers); survivorship/security-master bias is unresolved.\n'''
(OUT/'ACTUAL_ENGINE_REPORT.md').write_text(report,encoding='utf-8')
print('latest',latest_date.date(),'regime',regime,'eligible',len(L))
print(L[cols].head(10).to_string(index=False))
print(pd.DataFrame(metrics).to_string(index=False))
