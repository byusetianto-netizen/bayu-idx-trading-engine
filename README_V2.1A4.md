# V2.1A4 — Automatic IDX Publication Evidence Collector

## Tujuan
Menghilangkan kebutuhan screenshot manual untuk memperoleh `publication_date` laporan keuangan.

Collector mencoba mengambil disclosure index resmi IDX menggunakan browser Chromium (Playwright), karena endpoint `/primary/...` IDX dapat menolak request HTTP biasa dengan 403/anti-bot. IDX sendiri menyediakan disclosure/financial-report pages dan XBRL financial reporting. 

## Prinsip PIT
- `period_end` adalah tanggal periode akuntansi, bukan tanggal tersedia.
- `publication_date` hanya boleh berasal dari evidence publikasi IDX.
- Jika IDX tidak dapat diakses atau matching tidak jelas, collector **tidak menebak** tanggal.
- Evidence yang sudah diverifikasi manual (AADI/AALI/ABBA Q1 2026) dipertahankan sebagai seed dan dapat diaudit.
- Collector hanya merge evidence baru; tidak mengubah nilai fundamental.

## Sumber
Primary source: IDX disclosure / listed-company financial-report infrastructure.

## Output
- `data/fundamental/publication_evidence.csv`
- `data/fundamental/publication_collection_status.json`
- `data/fundamental/publication_collection_raw.json` (metadata only; no financial values)
- `data/fundamental/pit_publication_validation.json`

## Matching
Collector mencari disclosure yang mengandung kata kunci laporan keuangan dan mencocokkan:
1. ticker
2. tahun
3. periode (TW1/TW2/TW3/audit)
4. publication timestamp

Jika beberapa kandidat sama-sama mungkin, record ditandai `REVIEW` dan tidak dipromosikan sebagai PIT-safe.

## Schedule
Weekdays 20:45 WIB (`45 13 * * 1-5`).

## Important
Ini adalah data acquisition / PIT evidence layer saja. Tidak mengubah `actual_engine.py`, tidak membuat investment score, dan tidak membuat trade decision.
