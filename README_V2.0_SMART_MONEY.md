# IDX Trading Engine V2.0 — Smart Money / Bandarmology Research Layer

## Tujuan

V2.0 adalah research layer terpisah dari `actual_engine.py`. Modul ini tidak mengubah keputusan production engine sebelum lolos validation.

Fokus:
1. Broker concentration
2. Broker persistence
3. Net broker flow
4. Estimated Big-Money Average Price
5. Foreign flow
6. Accumulation / distribution evidence
7. Phase detector: ACCUMULATION → MARK-UP → DISTRIBUTION → MARK-DOWN
8. Research validation terhadap future returns

## Prinsip

- "Bandar" tidak dianggap sebagai fakta yang bisa diidentifikasi dari kode broker.
- Istilah yang dipakai engine adalah `smart_money_flow` / `estimated_big_money`.
- Semua threshold adalah research parameters, bukan klaim bahwa threshold tersebut optimal.
- Tidak ada look-ahead: fitur tanggal T hanya boleh memakai data sampai T.
- Untuk label outcome, entry historis menggunakan next-session open.
- V2.0 tidak memaksa trade.
- V2.0 tidak mengubah `actual_engine.py`.

## Data provider

Collector mendukung Index Alpha sebagai provider adapter. Dokumentasi provider menyatakan broker summary per ticker tersedia sejak 2025-01-01, dengan data harian; rentang multi-hari bersifat agregat, sehingga collector harus meminta satu hari per request.

Batch endpoint menerima sampai 50 ticker, tetapi quota tetap dihitung per ticker. Karena itu `MAX_TICKERS_PER_DAY` default dibuat 5 agar kompatibel dengan free plan.

API key hanya boleh disimpan sebagai GitHub Actions Secret `INDEXALPHA_API_KEY`, bukan di CSV, source code, atau Streamlit frontend.

## File output

- `data/broker/broker_summary_daily.csv`
- `data/broker/foreign_flow_daily.csv`
- `data/broker/broker_flow_features.csv`
- `data/broker/flow_phase_features.csv`
- `data/broker/v20_collection_status.json`

## Phase model

### ACCUMULATION
Evidence yang dicari:
- net broker flow positif
- buyer concentration/persistence meningkat
- harga belum mengalami markup berlebihan
- volume mendukung tanpa price expansion ekstrem

### MARK-UP
Evidence:
- accumulation sebelumnya
- higher-high / higher-low
- breakout atau trend acceleration
- volume confirmation
- flow masih mendukung

### DISTRIBUTION
Evidence:
- harga relatif extended
- seller concentration/persistence meningkat
- net flow melemah/negatif
- volume tinggi tetapi price progress melemah / gagal breakout

### MARK-DOWN
Evidence:
- distribution sebelumnya
- support/trend breakdown
- negative flow
- lower-high / lower-low

Jika evidence bertentangan atau tidak cukup:
`INCONCLUSIVE` atau `TRANSITION`.

## Validation

Validation tidak mengoptimalkan threshold pada holdout. Minimal evaluasi:
- forward return 20-session
- forward return 60-session
- win rate
- average / median return
- profit factor untuk rule portfolio bila diperlukan
- bootstrap confidence interval
- split berdasarkan waktu
- regime breakdown
- ablation:
  - OHLCV baseline
  - OHLCV + foreign flow
  - OHLCV + broker flow
  - OHLCV + broker + foreign flow

V2.0 baru boleh menjadi input decision engine setelah ada bukti out-of-sample yang stabil.
