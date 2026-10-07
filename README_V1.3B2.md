# V1.3B-2 — Price Data Quality Audit

## Purpose
Audit `data/idx_stock_prices_expansion.csv` before any consideration of replacing or expanding the main engine dataset.

## Safety
This version is **audit-only**. It does not modify `data/idx_stock_prices.csv` and does not feed the candidate dataset into the trading engine.

## Checks
- row/ticker counts
- duplicate `(ticker,date)` keys
- missing OHLCV
- non-positive prices
- invalid OHLC relationships
- negative volume
- first/last dates
- lag from latest IHSG date
- coverage against IHSG trading calendar
- maximum calendar gap
- maximum missing trading sessions

## Quality grades
- **A / RESEARCH_READY**: >=500 rows, latest lag <=10 days, no missing/invalid/duplicate data, and <=45 missing trading sessions in the observed range.
- **B / RESEARCH_WITH_REVIEW**: >=200 rows, latest lag <=30 days, no missing/invalid/duplicate data. Requires review before research use.
- **C / DO_NOT_USE_YET**: fails one or more minimum conditions.

Grade A is a **data-quality** classification only. It does not mean the stock was historically eligible/tradable throughout the sample. Point-in-time security master, suspensions, delistings, and survivorship bias remain separate problems.

## Outputs
- `data/price_data_quality_audit.csv`
- `data/price_data_quality_summary.json`
- `data/price_data_quality_status.json`
