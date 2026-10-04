from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path
import pandas as pd
import numpy as np
import subprocess, sys

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RADAR = DATA / 'engine_output' / 'latest_actual_radar.csv'
METRICS = DATA / 'engine_output' / 'model_validation_metrics.csv'
MANIFEST = DATA / 'update_manifest.json'
STOCKS = DATA / 'idx_stock_prices.csv'
IHSG = DATA / 'idx_ihsg_index.csv'

app = FastAPI(title='IDX Weekly Trading Engine', version='1.0.0')


def load_radar():
    df = pd.read_csv(RADAR)
    return df


def clean_num(x):
    return None if pd.isna(x) else float(x)


def market_summary():
    r = load_radar()
    latest = pd.to_datetime(r['date']).max()
    regime = r['regime'].iloc[0] if len(r) else 'UNKNOWN'
    return {
        'data_date': latest.strftime('%Y-%m-%d'),
        'regime': regime,
        'candidates': int(len(r)),
        'watch': int((r.decision == 'WATCH').sum()),
        'no_trade': int((r.decision == 'NO TRADE').sum()),
        'universe': int(pd.read_csv(STOCKS, usecols=['ticker'])['ticker'].nunique()) if STOCKS.exists() else 0,
        'universe_status': 'INCOMPLETE / RESEARCH ONLY',
        'update_manifest': pd.read_json(MANIFEST, typ='series').to_dict() if MANIFEST.exists() else None
    }

@app.get('/api/summary')
def summary():
    return market_summary()

@app.get('/api/validation')
def validation():
    df = pd.read_csv(METRICS)
    return df.to_dict(orient='records')

@app.get('/api/radar')
def radar(limit: int = 50):
    df = load_radar().head(max(1, min(limit, 100)))
    rows=[]
    for _,x in df.iterrows():
        d=x.to_dict()
        for k,v in list(d.items()):
            if isinstance(v, (np.floating, np.integer)): d[k]=float(v)
            elif pd.isna(v): d[k]=None
        rows.append(d)
    return rows

@app.get('/api/stock/{ticker}')
def stock(ticker: str):
    ticker=ticker.upper()
    r=load_radar()
    row=r[r.ticker.eq(ticker)]
    if row.empty: raise HTTPException(404, 'Ticker not found in current radar')
    x=row.iloc[0]
    result={k:(None if pd.isna(v) else (float(v) if isinstance(v,(np.floating,np.integer)) else v)) for k,v in x.to_dict().items()}
    # Last 90 observations for lightweight price chart
    if STOCKS.exists():
        s=pd.read_csv(STOCKS, usecols=['date','ticker','close'])
        s=s[s.ticker.eq(ticker)].tail(90)
        result['history']=[{'date':str(d),'close':float(c)} for d,c in zip(s.date,s.close)]
    return result

@app.post('/api/refresh')
def refresh():
    # Incrementally update data, then re-run the actual engine.
    try:
        u=subprocess.run([sys.executable, str(ROOT/'data_update.py')], cwd=ROOT, capture_output=True, text=True, timeout=300)
        if u.returncode != 0:
            raise HTTPException(500, u.stderr[-2000:])
        p=subprocess.run([sys.executable, str(ROOT/'actual_engine.py')], cwd=ROOT, capture_output=True, text=True, timeout=300)
        if p.returncode != 0:
            raise HTTPException(500, p.stderr[-2000:])
        return {'ok':True, 'message':'Data updated and engine refreshed', 'update_stdout':u.stdout[-4000:], 'engine_stdout':p.stdout[-4000:]}
    except subprocess.TimeoutExpired:
        raise HTTPException(504, 'Engine refresh timed out')

HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IDX Weekly Trading Engine</title>
<style>
:root{--bg:#0b1020;--panel:#11182b;--panel2:#151e34;--text:#e9edf7;--muted:#8f9ab3;--line:#26314a;--accent:#63a4ff;--green:#42d392;--red:#ff6b7a;--yellow:#f5c451}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(180deg,#0a0f1d,#0d1425);color:var(--text);font:14px/1.45 Inter,system-ui,Segoe UI,Arial,sans-serif}button{cursor:pointer;border:1px solid var(--line);background:var(--panel2);color:var(--text);padding:9px 13px;border-radius:9px}button:hover{border-color:var(--accent)}.wrap{max-width:1280px;margin:auto;padding:24px}.top{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:22px}.brand{font-size:22px;font-weight:800}.sub{color:var(--muted);font-size:12px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{background:rgba(17,24,43,.88);border:1px solid var(--line);border-radius:14px;padding:16px;box-shadow:0 12px 35px rgba(0,0,0,.16)}.label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}.big{font-size:27px;font-weight:800;margin-top:5px}.regime{color:var(--red)}.section{margin-top:18px}.section h2{font-size:16px;margin:0 0 10px}.notice{border-left:4px solid var(--yellow);background:#171a27;padding:12px 14px;border-radius:8px;color:#cbd2e3}.tablewrap{overflow:auto;border:1px solid var(--line);border-radius:12px}.table{width:100%;border-collapse:collapse;min-width:900px}.table th,.table td{padding:11px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}.table th{color:var(--muted);font-size:11px;text-transform:uppercase}.table th:first-child,.table td:first-child{text-align:left}.ticker{color:var(--accent);font-weight:800;cursor:pointer}.pill{padding:4px 8px;border-radius:99px;font-size:11px;font-weight:700}.pill.red{background:#3a1820;color:#ff8c98}.pill.green{background:#123326;color:#6be0aa}.meter{height:7px;background:#202a40;border-radius:99px;overflow:hidden}.meter i{display:block;height:100%;background:var(--accent)}.two{display:grid;grid-template-columns:2fr 1fr;gap:12px}.detail{display:none;position:fixed;inset:0;background:rgba(3,6,13,.72);backdrop-filter:blur(5px);z-index:5;padding:30px}.modal{max-width:900px;margin:30px auto;background:#10172a;border:1px solid var(--line);border-radius:16px;padding:20px;max-height:88vh;overflow:auto}.close{float:right}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:9px}.mini{background:#151e34;border:1px solid var(--line);border-radius:10px;padding:11px}.chart{height:190px;display:flex;align-items:end;gap:2px;border-bottom:1px solid var(--line);padding:10px}.bar{flex:1;background:#4e8fe7;min-width:2px}.foot{color:var(--muted);font-size:11px;margin-top:18px}@media(max-width:850px){.grid,.stats{grid-template-columns:repeat(2,1fr)}.two{grid-template-columns:1fr}.top{align-items:flex-start;flex-direction:column}}@media(max-width:500px){.grid,.stats{grid-template-columns:1fr}}
</style></head><body>
<div class="wrap"><div class="top"><div><div class="brand">IDX Weekly Trading Engine</div><div class="sub">Data → Algorithm → Validation → Decision → AI interpretation</div></div><button onclick="refreshEngine()">↻ Refresh Engine</button></div>
<div class="notice"><b>RESEARCH MODE.</b> Current dataset ends on <span id="noticeDate">—</span>. Universe is incomplete (95 tickers), so this system is not a live trading signal and survivorship bias remains unresolved.</div>
<div class="grid section"><div class="card"><div class="label">Market Regime</div><div class="big regime" id="regime">—</div></div><div class="card"><div class="label">Radar Candidates</div><div class="big" id="candidates">—</div></div><div class="card"><div class="label">Watch</div><div class="big" id="watch">—</div></div><div class="card"><div class="label">Last Data</div><div class="big" id="date">—</div></div></div>
<div class="section"><h2>Weekly Decision</h2><div class="card"><div class="big" id="decision">—</div><div class="sub" style="margin-top:5px">The engine is allowed to say <b>NO TRADE</b>. It never forces three picks.</div></div></div>
<div class="section"><h2>Weekly Radar</h2><div class="tablewrap"><table class="table"><thead><tr><th>Ticker</th><th>Score</th><th>P(+3%)</th><th>P(+5%)</th><th>P(+8%)</th><th>RSI</th><th>RS20</th><th>Liquidity</th><th>R/R</th><th>Decision</th></tr></thead><tbody id="radar"></tbody></table></div></div>
<div class="two section"><div class="card"><h2>Model Validation</h2><div id="validation"></div></div><div class="card"><h2>System Status</h2><p><b>Model:</b> HistGradientBoosting</p><p><b>Targets:</b> +3%, +5%, +8% within 5 sessions</p><p><b>Universe:</b> dynamic current local universe</p><p><b>Trading horizon:</b> 5 sessions</p><p><b>Status:</b> Research only</p></div></div>
<div class="foot">V1.1 data-aware web layer. The current web reads the existing research engine output; data/security-master improvements can be plugged in without redesigning the UI.</div></div>
<div class="detail" id="detail"><div class="modal"><button class="close" onclick="closeDetail()">Close</button><h1 id="dticker">—</h1><div id="dcontent"></div></div></div>
<script>
const fmt=(x,d=2)=>x==null?'—':Number(x).toLocaleString('en-US',{maximumFractionDigits:d});
const pct=x=>x==null?'—':(Number(x)*100).toFixed(1)+'%';
const money=x=>x==null?'—':'Rp '+Number(x).toLocaleString('id-ID',{maximumFractionDigits:0});
async function load(){let s=await (await fetch('/api/summary')).json(); document.getElementById('regime').textContent=s.regime;document.getElementById('candidates').textContent=s.candidates;document.getElementById('watch').textContent=s.watch;document.getElementById('date').textContent=s.data_date;document.getElementById('noticeDate').textContent=s.data_date;document.getElementById('decision').textContent=s.watch?'WATCHLIST ACTIVE':'NO TRADE';let r=await (await fetch('/api/radar?limit=50')).json();document.getElementById('radar').innerHTML=r.map(x=>`<tr><td><span class="ticker" onclick="openStock('${x.ticker}')">${x.ticker}</span></td><td>${fmt(x.opportunity_score,1)}</td><td>${pct(x.p3)}</td><td>${pct(x.p5)}</td><td>${pct(x.p8)}</td><td>${fmt(x.rsi,1)}</td><td>${pct(x.rs20)}</td><td>${money(x.liq20)}</td><td>${fmt(x.risk_reward_5,2)}</td><td><span class="pill ${x.decision==='WATCH'?'green':'red'}">${x.decision}</span></td></tr>`).join('');let v=await (await fetch('/api/validation')).json();document.getElementById('validation').innerHTML=v.map(x=>`<div style="margin:12px 0"><div style="display:flex;justify-content:space-between"><span>P(+${x.target}%)</span><b>${Number(x.validation_auc).toFixed(3)} AUC</b></div><div class="meter"><i style="width:${Math.max(0,Math.min(100,(x.validation_auc-.5)*500))}%"></i></div><div class="sub">Train observations: ${Number(x.train_n).toLocaleString()}</div></div>`).join('')}
async function openStock(t){let x=await (await fetch('/api/stock/'+t)).json();document.getElementById('dticker').textContent=x.ticker+' · '+money(x.close);let hist=x.history||[];let mn=Math.min(...hist.map(a=>a.close)),mx=Math.max(...hist.map(a=>a.close));let chart=hist.map(a=>`<div class="bar" title="${a.date}: ${money(a.close)}" style="height:${8+((a.close-mn)/(mx-mn||1))*92}%"></div>`).join('');document.getElementById('dcontent').innerHTML=`<div class="stats"><div class="mini"><div class="label">Score</div><b>${fmt(x.opportunity_score,1)}</b></div><div class="mini"><div class="label">P(+5%)</div><b>${pct(x.p5)}</b></div><div class="mini"><div class="label">RSI</div><b>${fmt(x.rsi,1)}</b></div><div class="mini"><div class="label">Decision</div><b>${x.decision}</b></div></div><div class="section card"><h2>Trade Plan (research)</h2><p>Entry reference: <b>${money(x.close)}</b> · Stop: <b>${money(x.stop)}</b> · Target +3%: <b>${money(x.target3)}</b> · Target +5%: <b>${money(x.target5)}</b> · Target +8%: <b>${money(x.target8)}</b></p><p>Risk/Reward +5%: <b>${fmt(x.risk_reward_5,2)}</b></p></div><div class="section card"><h2>Price History · 90 observations</h2><div class="chart">${chart}</div></div><div class="section card"><h2>Why this is / isn't a trade</h2><p>Market regime: <b>${x.regime}</b>. P(+3%) ${pct(x.p3)}, P(+5%) ${pct(x.p5)}, P(+8%) ${pct(x.p8)}. RSI ${fmt(x.rsi,1)}. The decision is generated by the deterministic engine, not by AI opinion.</p></div>`;document.getElementById('detail').style.display='block'}
function closeDetail(){document.getElementById('detail').style.display='none'}
async function refreshEngine(){let b=document.querySelector('button');b.disabled=true;b.textContent='Refreshing…';try{let r=await fetch('/api/refresh',{method:'POST'});if(!r.ok)throw new Error(await r.text());await load();alert('Engine refreshed.')}catch(e){alert('Refresh failed: '+e.message)}finally{b.disabled=false;b.textContent='↻ Refresh Engine'}}
load();
</script></body></html>'''

@app.get('/', response_class=HTMLResponse)
def home(): return HTMLResponse(HTML)
