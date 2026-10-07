# V1.3C — Historical Universe / Survivorship Bias Probe

Tujuan:
- Membuat layer awal historical security master.
- Memisahkan fakta yang bersumber dari security master dari tanggal observasi harga.
- Menandai gap panjang sebagai REVIEW FLAG, bukan otomatis suspension.
- Tidak mengubah dataset utama.
- Tidak mengaktifkan historical PIT eligibility pada engine.

## Prinsip penting

`first_observed_price_date != official_listing_date`

Jika listing date belum bersumber dari IDX atau sumber historis yang dapat diverifikasi,
field listing date tetap UNKNOWN/LOW confidence.

Current-universe membership juga belum cukup untuk mengetahui apakah saham benar-benar
eligible pada setiap tanggal historis.

## Output

- `data/historical_security_master.csv`
- `data/historical_universe_gap_flags.csv`
- `data/historical_universe_summary.json`
- `data/historical_universe_status.json`

## Status

Probe ini AUDIT-ONLY. Tidak mengganti `idx_stock_prices.csv` atau
`idx_stock_prices_expansion.csv` dan tidak menjalankan ulang engine.

## Next step

Setelah probe:
1. kumpulkan listing date historis yang bersumber,
2. kumpulkan delisting/partial delisting,
3. kumpulkan suspension intervals,
4. identifikasi ticker/name changes,
5. baru bangun PIT security master yang bisa dipakai backtest.

Sumber resmi yang diprioritaskan:
- IDX Company Profiles
- IDX Stock New Listings
- IDX suspension/delisting announcements
- IDX corporate actions

Sumber sekunder hanya dipakai sebagai cross-check.
