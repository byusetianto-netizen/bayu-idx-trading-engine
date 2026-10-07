# V1.3E-1 — Robust Universe Validation

## Tujuan

Menguji apakah hasil positif V1.3E (Research 297 vs Full 903) konsisten,
bukan hanya berasal dari satu periode.

Breakdown:
- overall
- calendar year
- market regime proxy
- primary classification

## Regime

Regime adalah proxy sederhana dari IHSG:
- BULL: trailing 20-session return > +2%
- BEAR: trailing 20-session return < -2%
- SIDEWAYS: lainnya

Ini bukan regime logic dari actual_engine.py. Ini hanya alat diagnostik.

## Bootstrap

95% CI untuk average 5D return:
- 2,000 resamples
- fixed random seed = 42

Jika CI masih melintasi 0, jangan menyebut average return sebagai robust positive evidence.

## Important

- No threshold optimization.
- No actual_engine.py change.
- No PIT-safe claim.
- No strategy profitability claim.
- No sequential overlapping-event drawdown metric.

IDX menyediakan data trading dan statistik seperti volume, value, frequency, trading days, listed issuers, serta statistik LQ45, yang menjadi konteks valid untuk pengembangan liquidity/regime layers. citeturn0search0turn0search1turn0search2

## Output

- `universe_robust_validation_stats.csv`
- `universe_robust_validation_comparison.csv`
- `universe_robust_validation_summary.json`
- `universe_robust_validation_status.json`
