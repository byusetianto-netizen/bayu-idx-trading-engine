# V2.2 Complete — Valuation Engine

This replaces the earlier V2.2 foundation with one integrated research layer.
No change to `actual_engine.py`.

## Included modules

1. Current valuation
   - PER
   - PBV
   - Earnings yield
   - growth-adjusted valuation / PEG diagnostic

2. Point-in-time historical valuation
   - daily PER/PBV observations only when the relevant filing was already published
   - historical median / P10 / P90
   - current percentile and discount vs historical median
   - strict minimum: 120 price observations and 2 distinct published financial periods

3. Peer-relative valuation
   - uses supplied `data/sector/sector_map.csv`
   - minimum 3 PIT-verified peers
   - peer median PER/PBV/growth/ROE
   - current discount vs peer median
   - if peer evidence is missing, status remains UNKNOWN

4. Value-trap diagnostics
   - negative EPS / book value
   - revenue or earnings decline
   - negative margin / ROE
   - combined value-trap flag

5. Conservative classification
   - GARP_UNDERVALUATION_CANDIDATE
   - POTENTIAL_VALUE
   - FAIR_OR_CONTEXT_DEPENDENT
   - EXPENSIVE_MULTIPLE
   - VALUE_TRAP_RISK
   - CURRENT_ONLY_CONTEXT
   - INSUFFICIENT_PIT_DATA

## PIT rules

- `publication_timestamp/publication_date <= analysis_date` is required.
- `period_end` is never treated as availability date.
- Missing publication evidence is UNKNOWN, not assumed available.
- Current valuation is not backfilled into history.
- Historical bands require multiple distinct published financial periods.

## Outputs

- `data/valuation/valuation_snapshot.csv`
- `data/valuation/pit_valuation_observations.csv`
- `data/valuation/valuation_status.json`
- `data/valuation/valuation_validation_status.json`

## Important limitation

The project currently has authoritative detailed financial evidence for only a small number of tickers. Therefore the complete engine is designed to scale safely, but it may report `INSUFFICIENT_PIT_DATA`, `INSUFFICIENT_PIT_HISTORY`, or `UNKNOWN` for most stocks. This is intentional and preferable to fabricated valuation history or peer comparisons.

## Not yet integrated into trade decisions

V2.2 is evidence/classification only. It does not modify `actual_engine.py`, does not generate BUY/SELL decisions, and does not represent probability or expected return.
