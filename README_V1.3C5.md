# V1.3C-5 — Universe Classification Layer

## Tujuan
Menambahkan lapisan klasifikasi universe untuk membedakan:
- Blue-chip / large-cap liquid proxy
- Conglomerate / group affiliation
- BUMN-affiliated research bucket
- Quantitative liquidity/volatility research bucket

## Prinsip penting
Klasifikasi ini **belum** digunakan untuk menentukan BUY/SELL dan **tidak mengubah actual_engine.py**.

IDX menjelaskan IDX30, LQ45, dan IDX80 sebagai indeks yang berfokus pada saham dengan likuiditas tinggi dan kapitalisasi pasar besar; LQ45 juga mempertimbangkan fundamental. Keanggotaan indeks dievaluasi berkala, sehingga snapshot hari ini tidak boleh dipakai untuk mengklaim bahwa suatu saham merupakan anggota indeks pada tanggal historis tertentu.

Karena historical membership evidence belum tersedia, semua:
- `historical_membership_verified = False`
- `pit_safe_for_backtest = False`

## File
- `universe_classification.py`
- `.github/workflows/v13c5_universe_classification.yml`
- `data/universe_classification_seed.csv`

Output:
- `data/universe_classification.csv`
- `data/universe_classification_review_queue.csv`
- `data/universe_classification_summary.json`
- `data/universe_classification_status.json`

## Cara menjalankan
1. Copy `universe_classification.py` ke root repository.
2. Buat folder `.github/workflows/` bila belum ada.
3. Copy workflow.
4. Copy seed ke `data/`.
5. Commit & Push.
6. GitHub → Actions → **V1.3C-5 Universe Classification** → Run workflow.

## Interpretasi
`BLUE_CHIP` = current/recent index snapshot proxy, bukan historical PIT membership.

`CONGLOMERATE` = current/recent group-affiliation research label, bukan legal ownership master.

`quant_research_bucket` = klasifikasi berbasis likuiditas dan volatilitas 20 hari; ini hanya bucket penelitian, bukan rekomendasi.

**Jangan mengubah actual_engine menjadi blue-chip-only sebelum V1.3C historical evidence layer selesai.**
