# V1.3A — BEI Universe & Data Integrity Audit

This patch is the audit-first foundation for expanding beyond the original 95-ticker research universe.

## Sources

1. **IDX Company Profiles** — authoritative reference for the current listed-company universe, when the official page can be parsed successfully.
2. **Daily all-stock snapshot** — secondary discovery/cross-check source derived from IDX data and maintained by `nofendian17/idx_dataset`.
3. **Local `idx_stock_prices.csv`** — current historical price dataset used by the engine.

The IDX Company Profiles page currently exposes a current-company table with code, name and listing date. The daily snapshot repository documents that it fetches all listed stocks for a date and stores daily CSV snapshots.

## Outputs

- `data/security_master.csv`
- `data/universe_audit.csv`
- `data/universe_status.json`
- `data/universe_audit_summary.json`

## Important methodological rules

- This phase does **not** replace the current price dataset.
- This phase does **not** expand `actual_engine.py` yet.
- Missing listing dates are `UNKNOWN`; they are never inferred from the first observed price date.
- Current listed universe is not the same as historical tradable universe.
- Historical point-in-time membership, suspension intervals, delistings and survivorship bias are not solved yet.
- A successful workflow is not evidence that every IDX company has historical price data.

## Workflow

Run locally:

```bash
python universe_update.py
python universe_audit.py
```

Only after reviewing the audit should we build the price-data expansion layer.
