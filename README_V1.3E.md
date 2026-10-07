# V1.3E — Research Universe Backtest Audit

## Tujuan

Menguji apakah penyaringan universe 903 -> 297 mengubah kualitas distribusi forward returns.

Perbandingan menggunakan **formula yang sama**:
- Entry = next trading session OPEN
- Return 5D = close pada sesi ke-5 / entry - 1
- Return 20D = close pada sesi ke-20 / entry - 1

Tidak ada threshold yang dioptimalkan berdasarkan hasil audit.

## Penting

Ini BUKAN pengganti backtest actual_engine.py.

Audit ini menjawab:
> "Apakah universe yang lebih bersih memiliki distribusi return yang berbeda?"

Bukan:
> "Apakah strategy ini profitable?"

Untuk klaim strategi tetap diperlukan walk-forward/out-of-sample validation, transaction cost, slippage, survivorship/PIT security master, dan validasi terhadap actual engine.

## Output

- `data/universe_backtest_comparison.csv`
- `data/universe_backtest_comparison_summary.json`
- `data/universe_backtest_by_class.csv`
- `data/universe_backtest_audit_status.json`

## Interpretasi

Jangan langsung memilih universe berdasarkan satu metrik.

Kita cari konsistensi:
- average return
- median return
- win rate
- profit factor
- drawdown
- jumlah observasi
- per-class behavior

IDX menyediakan statistik perdagangan dengan Volume, Value, Frequency, serta laporan aktivitas bulanan; ini mendukung penggunaan liquidity/trading-value sebagai salah satu data-quality gate, tetapi audit ini tetap menggunakan dataset lokal yang sedang kita bangun. citeturn0search0turn0search1turn0search2
