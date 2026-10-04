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
radar['date'] = pd.to_datetime(radar['date'])
latest = radar['date'].max() if len(radar) else None
regime = str(radar['regime'].iloc[0]) if len(radar) else 'UNKNOWN'
watch = radar[radar['decision'].eq('WATCH')].sort_values('opportunity_score', ascending=False)

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
        st.dataframe(watch[['ticker','close','opportunity_score','p3','p5','p8','risk_reward_5','decision']], use_container_width=True, hide_index=True)
    else:
        st.warning('NO TRADE — tidak ada saham yang memenuhi seluruh kriteria entry saat ini.')
    st.subheader('Top Radar')
    cols=['ticker','close','opportunity_score','p3','p5','p8','rsi','rs20','liq20','risk_reward_5','decision']
    st.dataframe(radar.sort_values('opportunity_score',ascending=False)[cols].head(10), use_container_width=True, hide_index=True)
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
    ticker = st.selectbox('Pilih saham', sorted(radar.ticker.unique()))
    row = radar[radar.ticker.eq(ticker)].sort_values('date').iloc[-1]
    st.subheader(f'{ticker} — Rp{row.close:,.0f}')
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Opportunity Score', f'{row.opportunity_score:.1f}')
    c2.metric('P(+5%)', f'{row.p5:.1%}')
    c3.metric('RSI', f'{row.rsi:.1f}')
    c4.metric('R/R', f'1 : {row.risk_reward_5:.2f}')
    st.divider()
    a,b,c = st.columns(3)
    a.metric('Stop', f'Rp{row.stop:,.0f}')
    b.metric('Target +5%', f'Rp{row.target5:,.0f}')
    c.metric('Target +8%', f'Rp{row.target8:,.0f}')
    st.subheader('Signal evidence')
    st.dataframe(pd.DataFrame({
        'Metric':['P(+3%)','P(+5%)','P(+8%)','RS20','Volume Ratio','MA20 Distance','MA50 Distance','MA200 Distance','Liquidity 20D','Regime','Decision'],
        'Value':[f'{row.p3:.1%}',f'{row.p5:.1%}',f'{row.p8:.1%}',f'{row.rs20:.1%}',f'{row.vol_ratio:.2f}x',f'{row.ma20_dist:.1%}',f'{row.ma50_dist:.1%}',f'{row.ma200_dist:.1%}',f'Rp{row.liq20:,.0f}',row.regime,row.decision]
    }), use_container_width=True, hide_index=True)
    if row.decision == 'NO TRADE': st.warning('NO TRADE: setup belum memenuhi seluruh syarat entry.')
    else: st.success('WATCH: kandidat memenuhi threshold riset; belum merupakan rekomendasi investasi.')

elif page == 'Validation & Research':
    st.header('Validation & Research')
    if len(metrics):
        st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.warning('Model belum terbukti memiliki edge yang robust pada final holdout. Jangan menganggap probability sebagai win-rate pasti.')
    st.markdown('**Prinsip:** score analitis ≠ probabilitas profit. Historical evidence dan out-of-sample validation harus dipisahkan.')

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
