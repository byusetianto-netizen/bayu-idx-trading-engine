# V2.1A — Financial Data Acquisition & PIT Foundation

Purpose: establish a safe acquisition layer for IDX financial statements before calculating fundamentals.

## Source hierarchy
1. IDX official XBRL / Financial Report / Disclosure.
2. Licensed IDX Data Services Financial Statement / historical products.
3. Issuer investor-relations reports for verification.
4. Third-party data only as discovery/cross-check; never silently promoted to PIT-authoritative.

## Non-negotiable PIT fields
- ticker
- period_start
- period_end
- period_type
- publication_date
- report_date
- source
- source_url
- document_id
- version
- confidence

`period_end` tells us what the accounting period covers. `publication_date` tells us when the market could know it. Backtests must use `publication_date <= analysis_date`.

## What this patch does
- creates a raw normalized financial-statement schema
- creates source registry and acquisition status
- validates PIT-critical fields
- provides a safe import utility for CSV/XLSX/JSON files supplied from an authoritative source
- does NOT scrape undocumented IDX endpoints
- does NOT use current Yahoo fundamentals as historical facts
- does NOT modify actual_engine.py
- does NOT generate BUY/SELL decisions

## What it intentionally does not claim
This patch does not magically obtain the licensed historical IDX Data Services feed. If an official source requires authentication/license access, the acquisition status records that limitation rather than substituting unsafe data.

## Next step
V2.1B = normalization of actual authoritative financial-report files into `financial_statements.csv`.
