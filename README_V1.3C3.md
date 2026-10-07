# V1.3C-3 — Evidence Reconciliation

Tujuan:
- Menggabungkan price universe, security master, dan event evidence.
- Menghasilkan status VERIFIED / INFERRED / UNKNOWN / REVIEW.
- Membuat review queue agar riset berikutnya fokus pada ticker yang benar-benar membutuhkan evidence.
- Tidak mengubah dataset harga.
- Tidak mengaktifkan PIT pada engine.

## Output

- `data/historical_universe_reconciliation.csv`
- `data/historical_universe_review_queue.csv`
- `data/historical_universe_reconciliation_summary.json`
- `data/historical_universe_reconciliation_status.json`

## Prinsip metodologi

First observed price date TIDAK sama dengan official listing date.

Jika listing/delisting evidence belum tersedia:
- status tetap UNKNOWN;
- first price date hanya dicatat sebagai observasi;
- candidate interval tidak boleh dipakai sebagai PIT truth.

`pit_safe_for_backtest` selalu FALSE pada tahap ini.

## Next step

Isi evidence yang terverifikasi untuk high-priority queue, lalu bangun V1.3C-4 PIT Security Master.
