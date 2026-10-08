import csv, json, os
ROOT="data/fundamental"
MANIFEST=os.path.join(ROOT,"financial_source_manifest.csv")
COVERAGE=os.path.join(ROOT,"fundamental_coverage.csv")
OUT=os.path.join(ROOT,"fundamental_coverage_status.json")

manifest=list(csv.DictReader(open(MANIFEST,encoding="utf-8")))
coverage=list(csv.DictReader(open(COVERAGE,encoding="utf-8")))
result={
 "status":"PASS" if manifest else "REVIEW",
 "engine_changed":False,
 "manifest_rows":len(manifest),
 "coverage_rows":len(coverage),
 "tickers":sorted(set(r["ticker"] for r in manifest)),
 "source_hierarchy":"IDX statement/XBRL primary; IDX Financial Data and Ratio secondary cross-check; company IR verification; third-party discovery only",
 "notes":[
   "This is a data reliability/coverage layer, not an investment score.",
   "Future approved/licensed acquisition can populate the same manifest without changing downstream metrics."
 ]
}
json.dump(result,open(OUT,"w",encoding="utf-8"),indent=2)
print(json.dumps(result,indent=2))
