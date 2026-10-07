# V1.3D — Research Universe Selector + Liquidity Gate

## Tujuan

Membuat "filter penelitian" sebelum saham masuk ke Opportunity Engine.

V1.3D TIDAK:
- mengubah `actual_engine.py`
- mengubah `idx_stock_prices_expansion.csv`
- membuat BUY/SELL signal
- mengklaim historical PIT membership
- menganggap blue-chip/konglomerasi sebagai sinyal beli

## Gate

A stock is `research_universe_eligible=True` when:
1. history >= 200 observations
2. latest data lag <= 10 calendar days
3. median trading value 20D >= Rp1bn/day
4. median trading value 60D >= Rp1bn/day

Volatility >250% is a risk flag, NOT an automatic exclusion.

Trading value currently uses `close * volume` from the research price dataset.

IDX's official Ringkasan Saham exposes volume, value, frequency, listed shares and tradeable shares. The official feed is useful for a future upgrade to an official liquidity/free-float layer. Current V1.3D does not pretend that local OHLCV-derived value is equivalent to a point-in-time IDX liquidity dataset.

## Output

`data/research_universe.csv`
- one row per ticker
- gate results
- liquidity bucket
- classification context
- research priority

`data/research_universe_review_queue.csv`
- stocks failing one or more research gates

`data/research_universe_summary.json`
- counts and thresholds

`data/research_universe_status.json`
- explicit status and limitations

## Next step

After this audit succeeds, the next phase is to connect the research universe to the Opportunity Radar as a selectable universe/filter, while preserving:
- NO TRADE
- score/risk thresholds
- PIT safety rules
- no forced trades
- no classification-based BUY bias.
