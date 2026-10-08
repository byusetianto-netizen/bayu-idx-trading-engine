# V2.1B — Fundamental Metrics Engine

Purpose: transform PIT-normalized IDX financial statement observations into transparent fundamental metrics. This version does **not** create a trading score and does not modify `actual_engine.py`.

## Outputs
- `data/fundamental/fundamental_metrics.csv`
- `data/fundamental/fundamental_assessment.csv`
- `data/fundamental/fundamental_metrics_status.json`
- `data/fundamental/fundamental_metrics_validation.json`

## Metric groups
- Growth: revenue, gross profit, net income, parent net income, EPS YoY using the same reporting period from the prior year when available.
- Profitability: gross margin, net margin, pre-tax margin, ROA/ROE.
- Balance sheet: debt/equity, liabilities/equity, current ratio, identified interest-bearing debt, net debt.
- Cash flow: CFO, CFI, CFF, free cash flow, CFO/net income.
- Earnings quality: descriptive cash conversion classification.

## Important definitions
- Interim ROA/ROE are reported both for the actual period and as explicitly labeled annualized estimates. They are not treated as trailing twelve-month metrics.
- FCF = CFO - reported cash payments for acquisition of PPE, mining properties, and intangible assets. The source XLSX represents these payment lines as positive amounts, so the engine subtracts them.
- Debt/equity currently covers identified bank loans and finance leases. It is intentionally conservative; taxonomy expansion may add other borrowings later.
- No valuation, catalyst, governance, technical, Smart Money, or trade decision is produced here.

## PIT
The metrics inherit `publication_date` from the normalized source records. The validator rejects assessment records published after the selected analysis date.

## Research principle
This layer describes financial quality. It does not claim that a strong fundamental profile implies a profitable trade.
