# V2.1E — Batch Financial Statement Importer

Purpose:
- Import IDX financial statement XLSX files placed in `data/fundamental/inbox/`.
- Normalize key financial metrics into the existing PIT-oriented financial statement layer.
- Use publication evidence from `data/fundamental/publication_evidence.csv` when available.
- Never infer publication date from period_end.
- Never fabricate missing values.
- Does not modify `actual_engine.py`.

## Expected input

Put one or more IDX FinancialStatement XLSX files in:

`data/fundamental/inbox/`

Recommended filename:
`FinancialStatement-2026-II-AALI.xlsx`

The importer detects ticker from the filename when possible, otherwise from the General Information sheet.

For PIT readiness, publication evidence should exist in:
`data/fundamental/publication_evidence.csv`

Expected evidence columns include:
`ticker, period_end, publication_date, publication_timestamp, source, confidence`

## Outputs

- `data/fundamental/financial_statements.csv`
- `data/fundamental/fundamental_import_audit.csv`
- `data/fundamental/fundamental_import_status.json`

The importer appends/merges records and de-duplicates by ticker, metric, period_end, publication_date.

## Important

A successful import is not the same as PIT verification.
Rows without publication evidence remain non-PIT-ready.

This layer is research-only and does not generate BUY/SELL signals.
