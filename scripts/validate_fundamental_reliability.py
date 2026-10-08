import csv, json, os
from datetime import datetime

ROOT="data/fundamental"
MANIFEST=os.path.join(ROOT,"financial_source_manifest.csv")
OUT=os.path.join(ROOT,"fundamental_reliability_status.json")

def parse_dt(s):
    return datetime.fromisoformat(s) if s else None

rows=list(csv.DictReader(open(MANIFEST,encoding="utf-8")))
checks=[]
for r in rows:
    pub=parse_dt(r["publication_timestamp"])
    checks.append({
        "ticker":r["ticker"],
        "period_end":r["period_end"],
        "publication_timestamp_present":bool(pub),
        "publication_confidence":r["publication_confidence"],
        "source_status":r["source_status"],
        "pit_rule_ready": bool(pub) and r["publication_confidence"]=="HIGH",
    })

status="PASS" if rows and all(x["pit_rule_ready"] for x in checks) else "REVIEW"
result={
    "status":status,
    "engine_changed":False,
    "tickers":len(rows),
    "pit_ready":sum(x["pit_rule_ready"] for x in checks),
    "checks":checks,
    "notes":[
        "Publication timestamp is treated as availability evidence; period_end is never used as availability date.",
        "Official IDX Financial Data and Ratio is a secondary cross-check and does not replace detailed statements.",
        "No scraping bypass or investment decision is performed by this layer."
    ]
}
json.dump(result,open(OUT,"w",encoding="utf-8"),indent=2)
print(json.dumps(result,indent=2))
