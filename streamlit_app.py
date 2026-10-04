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

st.set_page_config(page_title='IDX Opportunity Engine', page_icon='📈', layout='wide')
st.markdown('''<style>.block-container{padding-top:1.5rem;max-width:1450px}.small{color:#6b7280;font-size:.85rem}</style>''', unsafe_allow_html=True)

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
candidates = radar[radar['decision'].eq('TRADE CANDIDATE')].sort_values('opportunity_score', ascending=False) if len(radar) else radar

with st.sidebar:
    st.title('IDX Engine')
    st.caption('Explainable research / paper trading')
    if st.button('↻ Refresh Engine', use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    page = st.radio('Navigate', ['Dashboard','Opportunity Radar','Stock Analysis','Validation & Research','Belajar Istilah','Data Status'])

st.title('IDX Opportunity Trading Engine')
st.caption('Data → Algorithm → Validation → Risk/Reward → Explanation → Human decision')

if page == 'Dashboard':
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Market Regime', regime)
    c2.metric('Latest Data', latest.strftime('%d %b %Y') if latest else '—')
    c3.metric('Radar Candidates', len(radar))
    c4.metric('Trade Candidate', len(candidates))
    st.divider()

    if len(candidates):
        st.success(f'{len(candidates)} saham lolos filter riset saat ini. Ini bukan perintah beli.')
        cols = [c for c in ['ticker','close','opportunity_score','p_target','target_pct','risk_pct','risk_reward','expected_value','decision'] if c in candidates.columns]
        st.dataframe(candidates[cols], use_container_width=True, hide_index=True)
    else:
        st.warning('NO TRADE — belum ada saham yang secara bersamaan memenuhi standar peluang, risiko, risk/reward, dan expected value.')

    st.subheader('Top Opportunity Radar')
    cols = [c for c in ['ticker','close','opportunity_score','p3','p5','p8','p_target','rsi','rs20','risk_pct','risk_reward','expected_value','decision'] if c in radar.columns]
    st.dataframe(radar.sort_values('opportunity_score',ascending=False)[cols].head(10), use_container_width=True, hide_index=True)

    st.subheader('Cara membaca hasil')
    st.markdown('''
    **Trade Candidate** berarti setup lolos filter riset; bukan jaminan profit dan bukan instruksi beli.
    **NO TRADE** berarti ada minimal satu penghalang penting: risiko terlalu besar, peluang model belum cukup kuat, risk/reward rendah, expected value tidak positif, atau kondisi pasar terlalu lemah.
    ''')
    st.info('Horizon 60 sesi hanya dipakai sebagai batas penelitian agar statistik dapat dihitung. Ini BUKAN aturan harus menjual setelah 60 hari. Posisi dapat tetap dipertahankan selama alasan teknikal masih valid, dengan disiplin terhadap stop/invalidation.')
    st.info('Universe saat ini belum lengkap. Sistem adalah research/paper only dan tidak boleh dianggap sebagai rekomendasi investasi.')

elif page == 'Opportunity Radar':
    st.header('Opportunity Radar')
    st.caption(f'Signal date: {latest.strftime("%d %b %Y") if latest else "—"}')
    q = st.text_input('Cari ticker', '').upper().strip()
    view = radar.copy()
    if q: view = view[view.ticker.str.contains(q, na=False)]
    cols = [c for c in ['ticker','close','opportunity_score','p_target','target_pct','risk_pct','risk_reward','expected_value','regime','decision','decision_reason'] if c in view.columns]
    st.dataframe(view.sort_values('opportunity_score',ascending=False)[cols], use_container_width=True, hide_index=True)
    st.caption('P(+3%), P(+5%), dan P(+8%) adalah probabilitas model untuk mencapai target sebelum stop dalam maksimum 60 sesi penelitian. Bukan probabilitas profit pasti.')

elif page == 'Stock Analysis':
    st.header('Stock Analysis')
    if not len(radar):
        st.warning('Radar belum tersedia.')
        st.stop()
    ticker = st.selectbox('Pilih saham', sorted(radar.ticker.unique()))
    row = radar[radar.ticker.eq(ticker)].sort_values('date').iloc[-1]
    st.subheader(f'{ticker} — Rp{row.close:,.0f}')

    decision = str(row.decision)
    if decision == 'TRADE CANDIDATE': st.success('🟢 TRADE CANDIDATE — setup lolos filter riset. Ini bukan perintah beli.')
    else: st.warning('🔴 NO TRADE — setup belum memenuhi seluruh filter riset.')

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric('Opportunity Score', f'{row.opportunity_score:.1f}')
    c2.metric('P Target', f'{row.p_target:.1%}' if pd.notna(row.p_target) else '—')
    c3.metric('RSI', f'{row.rsi:.1f}')
    rr = row.get('risk_reward', float('nan'))
    c4.metric('Risk / Reward', f'1 : {float(rr):.2f}' if pd.notna(rr) else '—')
    c5.metric('Expected Value', f'{row.expected_value:.2%}' if 'expected_value' in row else '—')

    st.divider()
    a,b,c,d = st.columns(4)
    a.metric('Reference Entry', f'Rp{row.get("reference_entry", row.close):,.0f}')
    b.metric('Stop / Invalidation', f'Rp{row.stop:,.0f}')
    c.metric('Dynamic Target', f'Rp{row.target:,.0f}')
    d.metric('Planned Risk', f'{row.risk_pct:.1%}' if 'risk_pct' in row else '—')
    st.caption(f'Entry status: **{row.get("entry_status", "—")}**. Harga terakhir adalah reference entry; open sesi berikutnya belum diketahui.')

    st.subheader('🧠 Mengapa saham ini masuk radar?')
    st.write(row.get('why_positive','—'))

    st.subheader('⚠️ Mengapa keputusan ini?')
    st.write(row.get('decision_reason','—'))

    st.subheader('🔍 Apa risiko utamanya?')
    st.write(row.get('why_risk','—'))

    st.subheader('Angka yang mendasari keputusan')
    metrics_rows = [
        ('P(+3%)', f'{row.p3:.1%}', 'Peluang model mencapai +3% sebelum stop dalam maksimum 60 sesi penelitian.'),
        ('P(+5%)', f'{row.p5:.1%}', 'Peluang model mencapai +5% sebelum stop dalam maksimum 60 sesi penelitian.'),
        ('P(+8%)', f'{row.p8:.1%}', 'Peluang model mencapai +8% sebelum stop dalam maksimum 60 sesi penelitian.'),
        ('Target model', f'{row.target_pct:.1%}', 'Target dinamis berdasarkan risiko dan volatilitas saham.'),
        ('RSI', f'{row.rsi:.1f}', 'Indikator momentum; bukan sinyal beli/jual tunggal.'),
        ('RS20', f'{row.rs20:.1%}', 'Kekuatan saham dibanding IHSG selama sekitar 20 sesi.'),
        ('Volume Ratio', f'{row.vol_ratio:.2f}x', 'Volume hari ini dibanding rata-rata 20 sesi.'),
        ('ATR(14)', f'{row.atr14:,.2f}', 'Ukuran rata-rata pergerakan harga; dipakai membantu menentukan stop.'),
        ('ATR %', f'{row.atr_pct:.1%}', 'ATR dibanding harga; menunjukkan tingkat volatilitas relatif.'),
        ('MA20 Distance', f'{row.ma20_dist:.1%}', 'Jarak harga dari rata-rata 20 sesi.'),
        ('MA50 Distance', f'{row.ma50_dist:.1%}', 'Jarak harga dari rata-rata 50 sesi.'),
        ('MA200 Distance', f'{row.ma200_dist:.1%}', 'Jarak harga dari rata-rata 200 sesi.'),
        ('Risk / Reward', f'1 : {float(rr):.2f}', 'Berapa potensi reward dibanding 1 unit risiko.'),
        ('Expected Value', f'{row.expected_value:.2%}', 'Estimasi nilai harapan dari probabilitas model dan payoff/risk yang direncanakan; bukan jaminan profit.'),
        ('Market Regime', row.regime, 'Kondisi umum IHSG berdasarkan posisi terhadap MA50 dan MA200.'),
    ]
    st.dataframe(pd.DataFrame({'Istilah':[x[0] for x in metrics_rows],'Hasil':[x[1] for x in metrics_rows],'Arti sederhana':[x[2] for x in metrics_rows]}), use_container_width=True, hide_index=True)

    st.subheader('📌 Kesimpulan sederhana')
    if decision == 'TRADE CANDIDATE':
        st.write('Engine menilai setup ini cukup menarik secara statistik dan risk/reward untuk masuk daftar kandidat. Tetap tunggu harga entry aktual dan lakukan keputusan manusia sebelum transaksi.')
    else:
        st.write('Engine belum menemukan kombinasi peluang dan risiko yang cukup menarik. Tidak melakukan transaksi adalah keputusan yang valid.')

elif page == 'Validation & Research':
    st.header('Validation & Research')
    if len(metrics):
        st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.warning('AUC mengukur kemampuan model membedakan setup yang relatif lebih mungkin mencapai target; AUC bukan persentase keuntungan dan bukan win-rate.')
    st.markdown('**V1.2:** entry historis menggunakan next-session OPEN; stop/target dinamis; keputusan menggunakan risk/reward + target probability + expected value; horizon 60 sesi hanya untuk penelitian, bukan forced exit.')
    st.markdown('**Prinsip:** analytical score ≠ probability of profit ≠ guarantee. Historical evidence harus diuji pada out-of-sample data sebelum engine digunakan untuk keputusan nyata.')

elif page == 'Belajar Istilah':
    st.header('Belajar Istilah Trading')
    glossary = [
        ('Opportunity Score','Skor gabungan kekuatan setup. Semakin tinggi berarti semakin banyak faktor yang mendukung; bukan peluang profit 80% jika skornya 80.'),
        ('P(+3%) / P(+5%) / P(+8%)','Probabilitas model bahwa target tersebut tercapai sebelum stop dalam maksimum 60 sesi penelitian. Ini bukan jaminan harga akan naik.'),
        ('Risk / Reward (R/R)','Perbandingan potensi keuntungan dengan risiko. R/R 2 berarti setiap Rp1 risiko ditujukan untuk peluang reward sekitar Rp2.'),
        ('Expected Value (EV)','Perkiraan nilai rata-rata secara statistik dari kombinasi peluang, reward, dan risiko. EV positif belum menjamin setiap transaksi untung.'),
        ('Stop / Invalidation','Harga yang membuat alasan awal trade dianggap tidak lagi valid. Ini berbeda dari sekadar “batas rugi yang nyaman”.'),
        ('Target','Area harga yang digunakan untuk mengukur potensi reward. Target dinamis, bukan janji harga akan tercapai.'),
        ('ATR (Average True Range)','Ukuran volatilitas/pergerakan harga. ATR lebih tinggi berarti saham biasanya bergerak lebih besar.'),
        ('RSI','Indikator momentum 0–100. Nilai tinggi berarti momentum kuat/berpotensi panas; nilai rendah berarti momentum lemah. Tidak digunakan sendirian.'),
        ('MA20 / MA50 / MA200','Rata-rata harga 20/50/200 sesi. Dipakai untuk membaca posisi harga terhadap tren jangka pendek hingga panjang.'),
        ('RS20','Relative Strength 20 sesi terhadap IHSG. Positif berarti saham relatif lebih kuat daripada IHSG pada periode tersebut.'),
        ('Volume Ratio','Volume hari ini dibanding rata-rata volume 20 sesi. >1 berarti aktivitas perdagangan lebih tinggi dari biasanya.'),
        ('Market Regime','Kondisi umum IHSG: BULL, NEUTRAL, atau BEAR. Regime bearish membuat engine lebih selektif.'),
        ('Trade Candidate','Setup yang lolos filter riset. Bukan perintah beli.'),
        ('NO TRADE','Engine menilai setup belum cukup menarik atau risikonya tidak layak. Tidak trading adalah hasil yang valid.'),
        ('60-session research horizon','Batas waktu untuk menghitung bukti historis secara konsisten. Bukan aturan wajib memegang atau menjual setelah 60 sesi.'),
    ]
    st.dataframe(pd.DataFrame(glossary, columns=['Istilah','Penjelasan sederhana']), use_container_width=True, hide_index=True)

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
    st.write('Data historis masih terbatas pada universe yang tersedia. Pipeline update dapat mengambil data baru ketika dijalankan pada environment yang memiliki internet.')
