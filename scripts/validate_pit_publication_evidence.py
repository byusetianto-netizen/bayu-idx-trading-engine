from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]
FIN = ROOT / "data" / "fundamental" / "financial_statements.csv"
EVID = ROOT / "data" / "fundamental" / "publication_evidence.csv"
OUT = ROOT / "data" / "fundamental" / "pit_publication_validation.json"

def main():
    fin = pd.read_csv(FIN)
    ev = pd.read_csv(EVID)

    fin["ticker"] = fin["ticker"].astype(str).str.upper().str.strip()
    ev["ticker"] = ev["ticker"].astype(str).str.upper().str.strip()
    fin["period_end"] = pd.to_datetime(fin["period_end"], errors="coerce")
    ev["period_end"] = pd.to_datetime(ev["period_end"], errors="coerce")
    fin["publication_date"] = pd.to_datetime(fin["publication_date"], errors="coerce")

    target = fin.merge(
        ev[["ticker","period_end","publication_date"]].rename(
            columns={"publication_date":"evidence_publication_date"}
        ),
        on=["ticker","period_end"],
        how="inner"
    )
    target["match"] = (
        target["publication_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        == pd.to_datetime(target["evidence_publication_date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    checks = {
        "publication_date_present": int(target["publication_date"].notna().sum()),
        "publication_date_missing": int(target["publication_date"].isna().sum()),
        "evidence_rows_matched": int(len(target)),
        "exact_timestamp_matches": int(target["match"].sum()),
        "timestamp_mismatches": int((~target["match"]).sum()),
    }

    # PIT examples: a record is available only when publication timestamp <= analysis timestamp.
    examples = []
    for analysis in ["2026-04-28 10:00:00","2026-04-28 18:00:00","2026-04-29 10:00:00","2026-05-01 10:00:00"]:
        a = pd.Timestamp(analysis)
        available = target[target["publication_date"] <= a]["ticker"].drop_duplicates().tolist()
        examples.append({"analysis_date": analysis, "available_tickers": available})

    status = "PASS" if checks["timestamp_mismatches"] == 0 and checks["evidence_rows_matched"] > 0 else "REVIEW"
    result = {
        "status": status,
        "engine_changed": False,
        "checks": checks,
        "pit_examples": examples,
        "notes": [
            "PIT rule: publication_date <= analysis_date.",
            "A report's accounting period end does not determine availability.",
            "This validates publication evidence only; it does not validate the accounting metrics themselves."
        ]
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

if __name__ == "__main__":
    main()
