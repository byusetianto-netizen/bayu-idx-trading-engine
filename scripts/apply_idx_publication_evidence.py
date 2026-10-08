from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]
FIN = ROOT / "data" / "fundamental" / "financial_statements.csv"
EVID = ROOT / "data" / "fundamental" / "publication_evidence.csv"
STATUS = ROOT / "data" / "fundamental" / "pit_publication_status.json"

def norm_date(s):
    return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%d")

def main():
    if not FIN.exists():
        raise SystemExit(f"Missing {FIN}")
    if not EVID.exists():
        raise SystemExit(f"Missing {EVID}")

    fin = pd.read_csv(FIN)
    ev = pd.read_csv(EVID)

    if "ticker" not in fin.columns:
        raise SystemExit("financial_statements.csv must contain ticker")
    if "period_end" not in fin.columns:
        raise SystemExit("financial_statements.csv must contain period_end")

    fin["_ticker_key"] = fin["ticker"].astype(str).str.upper().str.strip()
    fin["_period_key"] = norm_date(fin["period_end"])

    ev["_ticker_key"] = ev["ticker"].astype(str).str.upper().str.strip()
    ev["_period_key"] = norm_date(ev["period_end"])

    pub_map = ev.set_index(["_ticker_key", "_period_key"])["publication_date"].to_dict()
    source_map = ev.set_index(["_ticker_key", "_period_key"])["publication_source"].to_dict()
    conf_map = ev.set_index(["_ticker_key", "_period_key"])["confidence"].to_dict()

    if "publication_date" not in fin.columns:
        fin["publication_date"] = pd.NA
    if "source" not in fin.columns:
        fin["source"] = pd.NA
    if "confidence" not in fin.columns:
        fin["confidence"] = pd.NA

    applied = 0
    for idx, row in fin.iterrows():
        key = (row["_ticker_key"], row["_period_key"])
        if key in pub_map:
            fin.at[idx, "publication_date"] = pub_map[key]
            if pd.isna(fin.at[idx, "source"]) or str(fin.at[idx, "source"]).strip() == "":
                fin.at[idx, "source"] = source_map[key]
            if pd.isna(fin.at[idx, "confidence"]) or str(fin.at[idx, "confidence"]).strip() == "":
                fin.at[idx, "confidence"] = conf_map[key]
            applied += 1

    fin = fin.drop(columns=["_ticker_key", "_period_key"])
    fin.to_csv(FIN, index=False)

    status = {
        "status": "PIT_PUBLICATION_EVIDENCE_APPLIED",
        "engine_changed": False,
        "evidence_rows": int(len(ev)),
        "financial_rows": int(len(fin)),
        "rows_with_evidence_applied": int(applied),
        "tickers_with_evidence": sorted(ev["ticker"].astype(str).str.upper().unique().tolist()),
        "notes": [
            "Publication timestamp is treated as the availability timestamp for PIT filtering.",
            "period_end is not used as publication_date.",
            "Evidence source is IDX and was captured from the IDX disclosure page screenshot supplied by the user.",
            "This patch does not create investment scores or trade decisions."
        ]
    }
    STATUS.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
