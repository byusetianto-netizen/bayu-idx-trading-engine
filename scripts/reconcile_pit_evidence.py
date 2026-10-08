from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FUND = ROOT / "data/fundamental"
STATEMENTS = FUND / "financial_statements.csv"
EVIDENCE = FUND / "publication_evidence.csv"
OUT = FUND / "financial_statements_pit.csv"
AUDIT = FUND / "pit_reconciliation_audit.csv"
STATUS = FUND / "pit_reconciliation_status.json"

def norm_date(s):
    return pd.to_datetime(s, errors="coerce", dayfirst=False, format="mixed").dt.date.astype("string")

def find_col(df, candidates):
    low = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

def main():
    if not STATEMENTS.exists():
        raise SystemExit("financial_statements.csv not found")

    fs = pd.read_csv(STATEMENTS)
    if fs.empty:
        raise SystemExit("financial_statements.csv is empty")

    fs["ticker"] = fs["ticker"].fillna("").astype(str).str.upper().str.strip()
    fs["period_end"] = norm_date(fs["period_end"])

    if not EVIDENCE.exists():
        ev = pd.DataFrame()
    else:
        ev = pd.read_csv(EVIDENCE)

    pub_col = find_col(ev, ["publication_timestamp","publication_date","published_at","date"])
    ev_ticker = find_col(ev, ["ticker","stock_code","code"])
    ev_period = find_col(ev, ["period_end","period_end_date","report_period_end"])
    ev_source = find_col(ev, ["source","source_status"])
    ev_conf = find_col(ev, ["confidence","publication_confidence"])

    fs["pit_publication_date"] = ""
    fs["pit_publication_timestamp"] = ""
    fs["pit_source"] = ""
    fs["pit_confidence"] = ""
    fs["pit_match_status"] = "NO_EVIDENCE"
    fs["pit_ready"] = False

    audits = []

    if not ev.empty and ev_ticker and pub_col:
        ev = ev.copy()
        ev["__ticker"] = ev[ev_ticker].fillna("").astype(str).str.upper().str.strip()
        ev["__pub"] = pd.to_datetime(ev[pub_col], errors="coerce", dayfirst=False, format="mixed")
        if ev_period:
            ev["__period"] = norm_date(ev[ev_period])
        else:
            ev["__period"] = pd.Series(pd.NA, index=ev.index, dtype="string")

        for idx, row in fs.iterrows():
            ticker = row["ticker"]
            period = row["period_end"]
            candidates = ev[ev["__ticker"].eq(ticker)].copy()

            # Strong match: ticker + period_end.
            if period is not pd.NA and pd.notna(period) and "__period" in candidates:
                exact = candidates[candidates["__period"].eq(period) & candidates["__pub"].notna()]
            else:
                exact = pd.DataFrame()

            if len(exact) == 1:
                e = exact.iloc[0]
                fs.at[idx, "pit_publication_date"] = e["__pub"].date().isoformat()
                fs.at[idx, "pit_publication_timestamp"] = e["__pub"].isoformat()
                fs.at[idx, "pit_source"] = str(e[ev_source]) if ev_source else "publication_evidence"
                fs.at[idx, "pit_confidence"] = str(e[ev_conf]) if ev_conf else "HIGH"
                fs.at[idx, "pit_match_status"] = "MATCHED_TICKER_PERIOD"
                fs.at[idx, "pit_ready"] = True
                audits.append({
                    "ticker": ticker, "period_end": str(period),
                    "status": "MATCHED_TICKER_PERIOD",
                    "publication_timestamp": e["__pub"].isoformat()
                })
            elif len(exact) > 1:
                fs.at[idx, "pit_match_status"] = "AMBIGUOUS_MULTIPLE_EVIDENCE"
                audits.append({
                    "ticker": ticker, "period_end": str(period),
                    "status": "AMBIGUOUS_MULTIPLE_EVIDENCE",
                    "publication_timestamp": ""
                })
            else:
                # Do NOT fall back to ticker-only evidence; that could attach
                # the wrong reporting period.
                audits.append({
                    "ticker": ticker, "period_end": str(period),
                    "status": "NO_EXACT_PERIOD_EVIDENCE",
                    "publication_timestamp": ""
                })
    else:
        for _, row in fs.iterrows():
            audits.append({
                "ticker": row["ticker"], "period_end": str(row["period_end"]),
                "status": "EVIDENCE_SCHEMA_UNAVAILABLE",
                "publication_timestamp": ""
            })

    fs.to_csv(OUT, index=False)
    pd.DataFrame(audits).to_csv(AUDIT, index=False)

    matched = int(fs["pit_ready"].sum())
    status = {
        "status": "PASS" if matched > 0 else "REVIEW",
        "engine_changed": False,
        "financial_rows": int(len(fs)),
        "tickers": int(fs["ticker"].nunique()),
        "pit_ready_rows": matched,
        "pit_ready_tickers": int(fs.loc[fs["pit_ready"], "ticker"].nunique()) if matched else 0,
        "evidence_rows": int(len(ev)),
        "notes": [
            "Publication evidence is matched by ticker + period_end only.",
            "Ticker-only fallback is intentionally prohibited to avoid wrong-period PIT attribution.",
            "Publication date is never inferred from period_end.",
            "This layer does not modify actual_engine.py."
        ]
    }
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
