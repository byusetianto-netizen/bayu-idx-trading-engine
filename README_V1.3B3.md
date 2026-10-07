# V1.3B-3 — OHLCV Integrity Audit

Purpose: validate the actual OHLCV rows in `data/idx_stock_prices_expansion.csv`
before any candidate dataset is considered for the trading engine.

Checks:
- invalid/missing dates
- missing OHLCV
- non-positive OHLC prices
- negative volume
- non-finite numeric values
- OHLC logical violations
- duplicate ticker/date observations
- exact duplicate rows
- long calendar gaps

Integrity grades are independent from V1.3B-2 coverage grades:
- A: no hard integrity issue, >=500 rows, no gap >45 calendar days, no ticker/date duplicates
- B: no hard integrity issue, but short history, long gap, or duplicate ticker/date requires review
- C: one or more hard integrity issues

Important:
- This audit does NOT modify `idx_stock_prices.csv`.
- It does NOT replace the expansion dataset.
- Passing the audit does NOT solve point-in-time membership, suspensions, delistings, or survivorship bias.
