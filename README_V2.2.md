# V2.2 Valuation Engine

Research-only valuation layer. No trade decision and no modification to actual_engine.py.

Goals:
- absolute valuation: PER/PBV when valid
- earnings yield
- growth-adjusted valuation (GARP diagnostic)
- peer-relative valuation when >=3 comparable peers exist
- historical valuation bands only when historical PIT valuation snapshots exist
- value-trap flags
- explicit data sufficiency / UNKNOWN states
- PIT rule: fundamental record publication_date <= analysis_date

Important:
- Low PER is NOT automatically undervalued.
- Negative EPS makes PER not meaningful.
- Negative equity makes PBV not meaningful.
- Interim EPS/ROE annualization must be explicitly labeled.
- Historical valuation requires point-in-time price + point-in-time financial snapshots; current valuation must never be backfilled into the past.
- This layer produces valuation evidence/classification only; it does not produce BUY/SELL/probability.

Inputs:
data/fundamental/fundamental_metrics.csv
data/fundamental/financial_statements.csv
data/fundamental/pit_validation_status.json
data/idx_stock_prices.csv
data/sector/sector_map.csv (optional)

Outputs:
data/valuation/valuation_snapshot.csv
data/valuation/valuation_status.json
