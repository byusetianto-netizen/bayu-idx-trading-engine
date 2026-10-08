# V2.2 PIT Diagnostic

Diagnostic only. This workflow does **not** modify `actual_engine.py`, `valuation_snapshot.csv`, or the V2.2 valuation logic.

It checks the exact current `scripts/valuation_engine.py` functions against:

- `data/fundamental/financial_statements_pit.csv`
- `load_fundamentals()`
- `pit_evidence_for_ticker()`

It tests AADI, AALI, and ABBA using the latest price date as the analysis date.

Expected rule:

`publication_date <= analysis_date`

Expected result with the current PIT evidence:

- AADI → PIT_VERIFIED
- AALI → PIT_VERIFIED
- ABBA → PIT_VERIFIED

The workflow is manual-only.
