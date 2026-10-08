# V2.1G — Simple Manual Publication Evidence

Tujuan: membuat input tanggal publikasi laporan keuangan sesederhana mungkin.

## Yang perlu diedit

Hanya file:

`data/fundamental/publication_input.csv`

Satu baris = satu emiten + satu periode laporan.

Kolom utama:
- `ticker` — contoh AADI
- `period_end` — contoh 2026-06-30
- `publication_date` — WAJIB, contoh 2026-08-28
- `publication_time` — OPSIONAL, contoh 16:13:53
- `source_url` — OPSIONAL
- `notes` — OPSIONAL

### Jika hanya tahu tanggal

Isi:

`AALI,2026-06-30,2026-08-28,,,,`

Engine akan menggunakan timestamp konservatif `23:59:59 WIB` pada tanggal tersebut. Ini sengaja dibuat konservatif untuk PIT: kita tidak menganggap informasi tersedia lebih awal pada hari yang sama.

### Jika tahu tanggal + jam

Isi:

`AADI,2026-06-30,2026-08-28,16:13:53,...`

Akan diberi confidence HIGH.

## Setelah edit

GitHub → Actions → `V2.1G Manual Publication Evidence` → Run workflow.

Workflow otomatis membangun:
- `publication_evidence.csv`
- `publication_manual_status.json`
- `publication_manual_validation.json`

Tidak ada perubahan ke `actual_engine.py` pada tahap ini.

## Prinsip

- Tidak ada financial value yang diisi otomatis.
- Tidak ada publication date yang ditebak dari period_end.
- Tidak perlu screenshot di repository.
- Tidak perlu scraping IDX.
- Input manual hanya metadata PIT yang memang kita ketahui.
