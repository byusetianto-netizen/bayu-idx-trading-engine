from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FUND = ROOT / "data/fundamental"
F = FUND / "financial_statements_pit.csv"
OUT = FUND / "pit_reconciliation_validation_status.json"

status = {"status":"PASS","engine_changed":False,"checks":{}}

if not F.exists():
    status["status"]="REVIEW"
    status["checks"]={"pit_file_present":False}
else:
    df = pd.read_csv(F)
    dup = int(df.duplicated(["ticker","metric","period_end","document_id"]).sum()) if len(df) else 0
    ready = int(df["pit_ready"].fillna(False).astype(bool).sum()) if "pit_ready" in df else 0
    bad_pub = 0
    if ready:
        pub = pd.to_datetime(df.loc[df["pit_ready"], "pit_publication_timestamp"], errors="coerce")
        period = pd.to_datetime(df.loc[df["pit_ready"], "period_end"], errors="coerce")
        bad_pub = int((pub.isna() | (pub < period)).sum())

    status["checks"] = {
        "pit_file_present": True,
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()) if len(df) else 0,
        "duplicates": dup,
        "pit_ready_rows": ready,
        "publication_before_period_end": bad_pub
    }
    if dup or bad_pub:
        status["status"]="REVIEW"

status["notes"]=[
    "PASS is structural/PIT-reconciliation validation only.",
    "A PIT-ready row requires explicit publication evidence matched to ticker and period_end.",
    "No investment decision is produced."
]
OUT.write_text(json.dumps(status, indent=2), encoding="utf-8")
print(json.dumps(status, indent=2))
