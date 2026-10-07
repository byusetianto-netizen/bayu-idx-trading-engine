# V1.3C-4 — PIT Security Master Audit

Builds a confidence-aware point-in-time security master.

STRICT policy:
- Official/verified listing evidence is required for a verified start.
- Verified delisting evidence closes an interval.
- Suspension is excluded only when an explicit suspension event exists.
- Price gaps never create suspension events.
- First observed price date is never relabeled as official listing date.

Outputs:
- `data/pit_security_master_v13c4.csv`
- `data/pit_daily_eligibility_sample.csv`
- `data/pit_security_master_review_queue.csv`
- `data/pit_security_master_summary.json`
- `data/pit_security_master_status.json`

IMPORTANT:
This workflow does NOT modify the price dataset and does NOT modify `actual_engine.py`.
PIT is not enabled for production/backtesting yet.

IDX evidence sources:
- Stock New Listings: official listing date field.
- Suspension Over 6 Months: official listing/suspension information.
- Corporate Actions: IPO, Company Listing, Partial Delisting and related events.

Next:
1. Populate verified event evidence.
2. Re-run V1.3C-3 reconciliation.
3. Re-run this PIT master.
4. Only then design V1.3D bias-aware backtest.
