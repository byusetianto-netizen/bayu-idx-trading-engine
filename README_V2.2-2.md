# V2.2-2 PIT Verification Fix

Root cause: V2.2 was coupling PIT verification to the valuation metric
extraction path. EPS/BVPS can be unavailable even when publication evidence
is valid.

Fix:
- Prefer `data/fundamental/financial_statements_pit.csv`.
- Determine PIT verification directly from ticker + period_end +
  publication_date.
- Require publication_date <= analysis_date.
- Do not require EPS/BVPS for PIT verification.

Expected with the current PIT file:
AADI = PIT_VERIFIED
AALI = PIT_VERIFIED
ABBA = PIT_VERIFIED

No EPS/BVPS is invented. Historical valuation remains subject to its existing
minimum-period and observation requirements. `actual_engine.py` is unchanged.
