#!/usr/bin/env python3
"""V2.2 PIT diagnostic only.

Reads the existing valuation engine and PIT financial file. Does not modify
valuation outputs or actual_engine.py.
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PIT = ROOT / "data" / "fundamental" / "financial_statements_pit.csv"
OUT = ROOT / "data" / "valuation"
OUT.mkdir(parents=True, exist_ok=True)


def norm(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def main():
    analysis_date = None
    report = {
        "status": "STARTED",
        "engine_changed": False,
        "pit_file_exists": PIT.exists(),
        "pit_file_rows_raw": 0,
        "pit_columns": [],
        "tests": [],
        "notes": []
    }

    if not PIT.exists():
        report["status"] = "FAIL_PIT_FILE_MISSING"
        (OUT / "v22_pit_diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    raw = norm(pd.read_csv(PIT))
    report["pit_file_rows_raw"] = int(len(raw))
    report["pit_columns"] = list(raw.columns)

    required = {"ticker", "period_end", "publication_date"}
    report["required_columns_present"] = required.issubset(raw.columns)

    # Use the exact engine functions that the production V2.2 workflow uses.
    import scripts.valuation_engine as engine

    prices = engine.load_prices()
    fs, metrics, manifest = engine.load_fundamentals()
    if prices.empty:
        report["status"] = "FAIL_NO_PRICES"
        report["notes"].append("Engine load_prices() returned empty.")
        (OUT / "v22_pit_diagnostic.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(json.dumps(report, indent=2, default=str))
        return

    latest = pd.to_datetime(prices["date"], errors="coerce").max()
    analysis_date = latest
    report["analysis_date"] = latest.date().isoformat() if pd.notna(latest) else None
    report["engine_fs_rows"] = int(len(fs))
    report["engine_fs_columns"] = list(fs.columns)

    for ticker, expected_period in [("AADI", "2026-06-30"), ("AALI", "2026-06-30"), ("ABBA", "2026-03-31")]:
        raw_t = raw[raw["ticker"].astype(str).str.upper().str.strip() == ticker].copy()
        fs_t = fs[fs["ticker"].astype(str).str.upper().str.strip() == ticker].copy() if not fs.empty and "ticker" in fs.columns else pd.DataFrame()
        result = engine.pit_evidence_for_ticker(fs, ticker, latest)

        test = {
            "ticker": ticker,
            "expected_period_end": expected_period,
            "raw_rows": int(len(raw_t)),
            "engine_loaded_rows": int(len(fs_t)),
            "raw_period_end_values": sorted(raw_t["period_end"].astype(str).unique().tolist()) if "period_end" in raw_t.columns else [],
            "raw_publication_date_values": sorted(raw_t["publication_date"].astype(str).unique().tolist()) if "publication_date" in raw_t.columns else [],
            "engine_result": {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in result.items()},
            "status": "PASS" if result.get("pit_status") == "PIT_VERIFIED" else "FAIL_PIT_NOT_VERIFIED"
        }
        report["tests"].append(test)

    verified = sum(t["status"] == "PASS" for t in report["tests"])
    report["pit_verified_test_count"] = verified
    report["status"] = "PASS" if verified == 3 else "DIAGNOSTIC_REQUIRES_REVIEW"
    report["notes"] = [
        "Diagnostic only: no actual_engine.py changes and no valuation snapshot changes.",
        "The diagnostic calls the exact load_fundamentals() and pit_evidence_for_ticker() functions from the current valuation_engine.py.",
        "Expected PIT rule: publication_date <= analysis_date."
    ]

    (OUT / "v22_pit_diagnostic.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))

if __name__ == "__main__":
    main()
