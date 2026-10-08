# V2.1F — PIT Financial Reconciliation

Links normalized financial statement rows to explicit publication evidence.

Strict rule:
- Match by ticker + period_end.
- Never infer publication date from period_end.
- Never fall back to ticker-only evidence.
- No changes to actual_engine.py.

Outputs:
- financial_statements_pit.csv
- pit_reconciliation_audit.csv
- pit_reconciliation_status.json
- pit_reconciliation_validation_status.json
