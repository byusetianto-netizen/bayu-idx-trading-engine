from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "data/fundamental/financial_statements.csv"
OUT = ROOT / "data/fundamental/fundamental_import_validation_status.json"

status = {"status":"PASS","engine_changed":False,"checks":{}}

if not F.exists():
    status["status"]="REVIEW"
    status["checks"]={"financial_statement_file":False}
else:
    df = pd.read_csv(F)
    dup = int(df.duplicated(["ticker","metric","period_end","document_id"]).sum()) if len(df) else 0
    missing_metric = int(df["metric"].isna().sum()) if "metric" in df else len(df)
    missing_period = int(df["period_end"].fillna("").eq("").sum()) if "period_end" in df else len(df)
    status["checks"] = {
        "financial_statement_file": True,
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()) if len(df) else 0,
        "duplicates": dup,
        "missing_metric": missing_metric,
        "missing_period_end": missing_period,
        "pit_ready_rows": int(df["publication_date"].fillna("").astype(str).str.strip().ne("").sum()) if "publication_date" in df else 0,
    }
    if dup or missing_metric or missing_period:
        status["status"]="REVIEW"
status["notes"]=[
    "PASS means structural validation only.",
    "PIT readiness requires publication evidence and publication_date <= analysis_date.",
    "Do not treat imported values as investment conclusions."
]
OUT.write_text(json.dumps(status, indent=2), encoding="utf-8")
print(json.dumps(status, indent=2))
