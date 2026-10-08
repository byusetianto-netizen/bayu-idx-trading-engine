#!/usr/bin/env python3
"""V2.2 PIT Diagnostic v2.

Read-only diagnostic. Does not modify the valuation engine or trading outputs.
"""
from pathlib import Path
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import valuation_engine as engine  # noqa: E402


def safe_value(x):
    if pd.isna(x):
        return None
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x


def main():
    prices = engine.load_prices()
    fs, metrics, manifest = engine.load_fundamentals()

    if prices.empty:
        raise RuntimeError("Price data is empty.")

    latest_date = prices["date"].max()

    result = {
        "status": "DIAGNOSTIC_COMPLETE",
        "analysis_date": safe_value(latest_date),
        "financial_source_expected": str(engine.FS_PIT),
        "financial_source_exists": bool(engine.FS_PIT.exists()),
        "financial_rows_loaded": int(len(fs)),
        "financial_columns": list(fs.columns),
        "tickers_checked": {},
    }

    for ticker in ["AADI", "AALI", "ABBA"]:
        q = fs[fs["ticker"].astype(str).str.upper().str.strip() == ticker].copy()
        pit = engine.pit_evidence_for_ticker(fs, ticker, latest_date)

        sample_cols = [
            c for c in
            ["ticker", "metric", "period_end", "publication_date",
             "publication_timestamp", "pit_ready", "source"]
            if c in q.columns
        ]

        result["tickers_checked"][ticker] = {
            "rows_loaded": int(len(q)),
            "sample": [
                {k: safe_value(v) for k, v in row.items()}
                for row in q[sample_cols].head(20).to_dict(orient="records")
            ],
            "pit_result": {k: safe_value(v) for k, v in pit.items()},
        }

    out = ROOT / "data" / "valuation" / "v22_pit_diagnostic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
