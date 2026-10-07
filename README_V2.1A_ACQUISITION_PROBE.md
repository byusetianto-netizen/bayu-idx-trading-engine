# V2.1A-1 — IDX Financial Statement Acquisition Probe

## Purpose

This probe tests whether GitHub Actions can retrieve **financial-report metadata from the official IDX endpoint** and, for a small sample, reach the referenced XBRL/XLSX attachment.

It does NOT populate the production fundamental dataset yet.

## Official endpoint

`https://www.idx.co.id/primary/ListedCompany/GetFinancialReport`

Parameters used by the probe:
- `year`
- `reportType=rdf`
- `periode` = `audit`, `tw1`, `tw2`, or `tw3`
- `kodeEmiten`

The endpoint is documented by IDX-facing open-source clients and returns attachments such as XBRL ZIP, inline XBRL ZIP, XLSX and PDF. The official IDX XBRL page confirms that financial statements use the IDX XBRL taxonomy.

## Safety rules

1. Metadata only by default; no production data overwrite.
2. Small probe: default 3 tickers.
3. No current fundamental values are inferred.
4. Publication/report dates are preserved when available.
5. Attachment SHA-256 is recorded only when an attachment is downloaded.
6. The probe fails closed if IDX blocks access.
7. `curl_cffi` browser impersonation is used because ordinary HTTP clients can receive anti-bot/403 responses from IDX infrastructure.

## Output

- `data/fundamental/acquisition_probe.csv`
- `data/fundamental/acquisition_probe_status.json`

## Next step

If metadata and one XBRL attachment are accessible, build V2.1A-2 XBRL attachment downloader/parser. If blocked, do not bypass access controls; use an approved IDX export/data-service route instead.
