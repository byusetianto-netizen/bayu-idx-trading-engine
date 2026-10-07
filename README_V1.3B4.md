# V1.3B-4 — Session Coverage Diagnostic

Purpose:
Diagnose why some long-history tickers have stock-session counts above the
number of sessions in the IHSG dataset over the same date range.

This is diagnostic only.

It compares:
- stock observation dates
- IHSG observation dates
- common dates
- stock-only dates
- IHSG-only dates

It intentionally does NOT assume stock-only dates are invalid.

Outputs:
- `data/session_coverage_diagnostic.csv`
- `data/session_coverage_examples.csv`
- `data/session_coverage_summary.json`

No price dataset is overwritten and no trading-engine eligibility is changed.

This does not solve point-in-time security-master membership or survivorship bias.
