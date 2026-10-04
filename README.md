# IDX Weekly Trading Engine — Web V1

Web dashboard connected to the actual-data research engine.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000

## What is included
- Dashboard / weekly decision
- Radar table with ranking and probabilities
- Clickable stock detail
- 90-observation price view
- Model validation panel
- Manual engine refresh endpoint
- Current research-data status warning

## Important
This is **research/paper trading software**, not financial advice and not a production live signal. The supplied universe is still only 95 tickers and the latest data date is 2026-07-01. The actual engine also currently uses close-at-signal for target labeling; this should be corrected to next-session-open before production backtesting.

## Data Update V1.1

The web now includes `data_update.py` for incremental updates. It uses:

1. **Yahoo Finance / yfinance** as the primary updater for the existing ticker universe.
2. **GitHub daily snapshot fallback** (`nofendian17/idx_dataset`) when yfinance is unavailable.
3. Local validation: duplicate `(ticker,date)` removal, numeric coercion, and update manifest.
4. IHSG is refreshed separately through yfinance.

Run manually:

```bash
python data_update.py --end 2026-10-02
python actual_engine.py
uvicorn app:app --reload
```

Or use **Refresh Engine** from the web UI; it runs the updater first and then the engine.

### Current package limitation

The execution environment used to build this package cannot make outbound HTTP requests from Python, so the bundled CSV remains at **2026-07-01**. The latest market date was independently verified on the web as **2026-10-02**; IHSG closed at **6,036.89 (+0.46%)**. Run the updater on an internet-connected machine to append the missing sessions and regenerate the radar. This is intentionally surfaced in `data/update_manifest.json` rather than fabricating data.

The system remains research/paper only. The historical universe is still incomplete and the updater does not solve point-in-time security-master or survivorship-bias issues.
