import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RADAR = DATA / 'latest_actual_radar.csv'
METRICS = DATA / 'model_validation_metrics.csv'
MANIFEST = DATA / 'update_manifest.json'
STOCKS = DATA / 'idx_stock_prices.csv'
IHSG = DATA / 'idx_ihsg_index.csv'

st.set_page_config(page_title='IDX Weekly Trading Engine', page_icon='📈', layout='wide')
st.markdown('''<style>.block-container{padding-top:1.5rem;max-width:1400px}.small{color:#6b7280;font-size:.85rem}</style>''', unsafe_allow_html=True)

@st.cache_data
def load_data():
    radar = pd.read_csv(RADAR)
    metrics = pd.read_csv(METRICS) if METRICS.exists() else pd.DataFrame()
    stocks = pd.read_csv(STOCKS, usecols=['ticker','date','close']) if STOCKS.exists() else pd.DataFrame()
    ihsg = pd.read_csv(IHSG) if IHSG.exists() else pd.DataFrame()
    return radar, metrics, stocks, ihsg

radar, metrics, stocks, ihsg = load_data()
if len(radar):
    radar['date'] = pd.to_datetime(radar['date'])
latest = radar['date'].max() if len(radar) else None
regime = str(radar['regime'].iloc[0]) if len(radar) else 'UNKNOWN'
watch = radar[radar['decision'].eq('WATCH')].sort_values('opportunity_score', ascending=False) if len(radar) else radar

with st.sidebar:
    st.title('IDX Engine')
    st.caption('Research / Paper Trading')
    if st.button('↻ Refresh Engine', use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    page = st.radio('Navigate', ['Dashboard','Weekly Radar','Stock Analysis','Validation & Research','Data Status'])

st.title('IDX Weekly Trading Engine')
st.caption('Data → Algorithm → Validation → Decision → AI interpretation')

if page == 'Dashboard':
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Market Regime', regime)
    c2.metric('Latest Data', latest.strftime('%d %b %Y') if latest else '—')
    c3.metric('Radar Candidates', len(radar))
    c4.metric('WATCH', len(watch))
    st.divider()

    if len(watch):
        st.success(f'{len(watch)} candidate(s) currently pass the research threshold.')
        wcols = [c for c in ['ticker','close','opportunity_score','p5','risk_pct','risk_reward','stop','target','decision'] if c in watch.columns]
        st.dataframe(watch[wcols], use_container_width=True, hide_index=True)
    else:
        st.warning('NO TRADE — tidak ada saham yang memenuhi seluruh kriteria entry saat ini.')

    st.subheader('Top Radar')
    cols = [c for c in ['ticker','close','opportunity_score','p3','p5','p8','rsi','rs20','liq20','risk_pct','risk_reward','decision'] if c in radar.columns]
    st.dataframe(radar.sort_values('opportunity_score',ascending=False)[cols].head(10), use_container_width=True, hide_index=True)

    if len(radar) and 'entry_status' in radar.columns:
        st.info('Entry historis menggunakan OPEN sesi perdagangan berikutnya. Pada radar terbaru, harga terakhir hanya menjadi reference entry; OPEN berikutnya belum diketahui.')
    st.info('Universe saat ini belum lengkap. Sistem adalah research/paper only dan tidak boleh dianggap sebagai rekomendasi investasi.')

elif page == 'Weekly Radar':
    st.header('Weekly Radar')
    st.caption(f'Signal date: {latest.strftime("%d %b %Y") if latest else "—"}')
    q = st.text_input('Cari ticker', '').upper().strip()
    view = radar.copy()
    if q: view = view[view.ticker.str.contains(q, na=False)]
    st.dataframe(view.sort_values('opportunity_score',ascending=False), use_container_width=True, hide_index=True)

elif page == 'Stock Analysis':
    st.header('Stock Analysis')
    if not len(radar):
        st.warning('Radar belum tersedia.')
        st.stop()
    ticker = st.selectbox('Pilih saham', sorted(radar.ticker.unique()))
    row = radar[radar.ticker.eq(ticker)].sort_values('date').iloc[-1]
    st.subheader(f'{ticker} — Rp{row.close:,.0f}')

    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Opportunity Score', f'{row.opportunity_score:.1f}')
    c2.metric('P(+5%)', f'{row.p5:.1%}')
    c3.metric('RSI', f'{row.rsi:.1f}')
    rr = row.get('risk_reward', row.get('risk_reward_5', float('nan')))
    c4.metric('R/R', f'1 : {float(rr):.2f}' if pd.notna(rr) else '—')

    st.divider()
    a,b,c,d = st.columns(4)
    a.metric('Reference Entry', f'Rp{row.get("reference_entry", row.close):,.0f}')
    b.metric('Stop', f'Rp{row.stop:,.0f}')
    c.metric('Dynamic Target', f'Rp{row.target:,.0f}')
    d.metric('Risk', f'{row.risk_pct:.1%}' if 'risk_pct' in row else '—')

    if 'entry_status' in row.index:
        st.caption(f'Entry status: **{row.entry_status}**')

    st.subheader('Signal evidence')
    metrics_rows = [
        ('P(+3%)', f'{row.p3:.1%}'),
        ('P(+5%)', f'{row.p5:.1%}'),
        ('P(+8%)', f'{row.p8:.1%}'),
        ('RS20', f'{row.rs20:.1%}'),
        ('Volume Ratio', f'{row.vol_ratio:.2f}x'),
        ('ATR(14)', f'{row.atr14:,.2f}' if 'atr14' in row else '—'),
        ('ATR %', f'{row.atr_pct:.1%}' if 'atr_pct' in row else '—'),
        ('MA20 Distance', f'{row.ma20_dist:.1%}'),
        ('MA50 Distance', f'{row.ma50_dist:.1%}'),
        ('MA200 Distance', f'{row.ma200_dist:.1%}'),
        ('Liquidity 20D', f'Rp{row.liq20:,.0f}'),
        ('Risk / Reward', f'1 : {float(rr):.2f}' if pd.notna(rr) else '—'),
        ('Regime', row.regime),
        ('Decision', row.decision),
    ]
    st.dataframe(pd.DataFrame({'Metric':[x[0] for x in metrics_rows],'Value':[x[1] for x in metrics_rows]}), use_container_width=True, hide_index=True)
    if row.decision == 'NO TRADE': st.warning('NO TRADE: setup belum memenuhi seluruh syarat entry.')
    else: st.success('WATCH: kandidat memenuhi threshold riset; belum merupakan rekomendasi investasi.')

elif page == 'Validation & Research':
    st.header('Validation & Research')
    if len(metrics):
        st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.warning('Model belum terbukti memiliki edge yang robust pada final holdout. Jangan menganggap probability sebagai win-rate pasti.')
    st.markdown('**Prinsip:** score analitis ≠ probabilitas profit. Historical evidence dan out-of-sample validation harus dipisahkan.')
    st.markdown('**V1.1:** historical labels menggunakan next-session OPEN; stop/target dan R/R bersifat dinamis.')

else:
    st.header('Data Status')
    manifest = {}
    if MANIFEST.exists():
        try: manifest=json.loads(MANIFEST.read_text())
        except Exception: pass
    c1,c2,c3 = st.columns(3)
    c1.metric('Last engine date', latest.strftime('%Y-%m-%d') if latest else '—')
    c2.metric('Tickers', int(stocks.ticker.nunique()) if len(stocks) else 0)
    c3.metric('Stock rows', f'{len(stocks):,}')
    st.json(manifest)
    st.error('UNIVERSE-INCOMPLETE / RESEARCH ONLY')
    st.write('Data historis yang tersedia di paket ini masih terbatas; pipeline update dapat mengambil data baru ketika dijalankan pada environment yang memiliki internet.')
