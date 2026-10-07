# V1.3C-2 — Historical Event Evidence Layer

Tujuan:
- Menyediakan struktur evidence untuk listing, suspension, delisting, dan ticker change.
- Mengutamakan bukti resmi IDX.
- Tidak menganggap first price date sebagai listing date.
- Tidak menganggap price gap sebagai suspension.
- Tidak mengubah dataset harga utama.
- Tidak mengubah engine trading.

## Folder input

`data/historical_events/`

Template:
- `listing_events.csv`
- `suspension_events.csv`
- `delisting_events.csv`
- `ticker_change_events.csv`

Kolom:
`event_type,ticker,company_name,event_date,end_date,old_ticker,new_ticker,sector,board,source,source_date,source_url,confidence,notes`

## Source priority

1. IDX official reports/pages
2. IDX-derived structured datasets
3. Secondary sources only as cross-check

Contoh resmi IDX:
- Stock New Listings menyediakan Code, Company Name, dan Listing Date.
- Suspension Over 6 Months menyediakan Code, Company Name, Listing Date, Suspension Date, Sector dan Board.
- IDX Corporate Actions memiliki kategori termasuk Partial Delisting, IPO, Company Listing, dan lainnya.

## Penting

Pada first run, template masih kosong. Itu NORMAL.

Setelah evidence resmi dikumpulkan, isi CSV tersebut dan jalankan workflow lagi.

Jangan memasukkan event yang hanya diduga dari gap harga sebagai suspension.
Jangan mengubah `idx_stock_prices.csv` atau `idx_stock_prices_expansion.csv`.

Output:
- `data/historical_event_evidence.csv`
- `data/historical_security_master_v13c2.csv`
- `data/historical_event_evidence_summary.json`
- `data/historical_event_evidence_status.json`
