# V2.1A-2 — IDX Financial XLSX/XBRL Local Import + PIT Normalizer

Purpose: prove a deterministic local acquisition/import path using an official IDX Financial Statement XLSX, without bypassing IDX access controls and without changing `actual_engine.py`.

## Golden sample
- Ticker: AADI
- File: `FinancialStatement-2026-II-AADI.xlsx`
- Reporting period: 2026-01-01 to 2026-06-30
- Publication date: 2026-08-28 (from IDX disclosure page)
- Report/review date: 2026-08-26
- Presentation currency: USD
- Rounding: In Thousand
- Report type: Limited Review

## Run locally
```bash
python scripts/import_idx_financial_statement.py --xlsx data/fundamental/inbox/FinancialStatement-2026-II-AADI.xlsx --publication-date 2026-08-28
python scripts/validate_pit_financials.py --analysis-date 2026-09-01
```

## PIT rule
A record is available to a backtest/analysis date only when:
`publication_date <= analysis_date`

`period_end` is not the availability date.

## Output
- `data/fundamental/financial_statements.csv`
- `data/fundamental/financial_import_metadata.json`
- `data/fundamental/acquisition_status.json`
- `data/fundamental/pit_validation_status.json`

Raw IDX documents should remain local unless redistribution/licensing permits committing them to the public repository.
