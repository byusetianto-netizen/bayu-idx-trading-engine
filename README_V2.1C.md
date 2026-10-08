# V2.1C — Fundamental Data Reliability & Coverage Layer

Purpose:
- Preserve the project's original objective: fundamental analysis must be based on auditable, point-in-time-safe data.
- Do not depend on screenshots.
- Do not depend on a single IDX endpoint.
- Separate raw financial statements from derived metrics.
- Use official IDX Financial Data and Ratio as a cross-check/coverage source, not as a substitute for detailed statements.
- Do not modify actual_engine.py or produce investment/trade decisions.

Source hierarchy:
1. Official issuer/IDX financial statement XLSX or XBRL with identifiable publication evidence — PRIMARY.
2. Official IDX Financial Data and Ratio — SECONDARY CROSS-CHECK / COVERAGE.
3. Company IR copy of the same filing — VERIFICATION.
4. Third-party provider — DISCOVERY ONLY until reconciled.

PIT rule:
publication_timestamp <= analysis_timestamp.
period_end is never used as publication date.
If publication timestamp is unavailable, PIT status = UNKNOWN/REVIEW.

This patch creates:
- financial_source_manifest.csv
- fundamental_coverage.csv
- scripts/validate_fundamental_reliability.py
- scripts/build_fundamental_coverage.py
- workflow v21c

Important:
- This patch does NOT scrape/bypass protected IDX endpoints.
- It does NOT attempt to defeat HTTP 403/WAF.
- It is designed so a future licensed/approved acquisition source can be plugged in without changing the fundamental engine.
- It does not change actual_engine.py.
