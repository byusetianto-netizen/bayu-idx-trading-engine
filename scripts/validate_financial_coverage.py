#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd

ALLOWED={"COVERED","INCOMPLETE","MISSING"}

def bool_series(s):
    return s.astype(str).str.strip().str.lower().isin(["true","1","yes","y"])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--universe",default="data/research_universe.csv")
    ap.add_argument("--source",default="data/fundamental/financial_statements_pit.csv")
    ap.add_argument("--assessment",default="data/fundamental/fundamental_coverage_assessment.csv")
    ap.add_argument("--queue",default="data/fundamental/acquisition_queue.csv")
    ap.add_argument("--analysis-date",required=True)
    ap.add_argument("--analysis-time",default="23:59:59")
    ap.add_argument("--output",default="data/fundamental/coverage_validation_status.json")
    a=ap.parse_args()

    u=pd.read_csv(a.universe)
    s=pd.read_csv(a.source)
    x=pd.read_csv(a.assessment)
    q=pd.read_csv(a.queue)

    diagnostics=[]
    req_u={"ticker","research_universe_eligible"}
    req_x={"ticker","analysis_timestamp","coverage_status","latest_period_end",
           "critical_metrics_present","critical_metrics_required",
           "missing_critical_metrics","acquisition_needed","reason"}
    if not req_u.issubset(u.columns):
        diagnostics.append(f"universe missing columns: {sorted(req_u-set(u.columns))}")
    if not req_x.issubset(x.columns):
        diagnostics.append(f"assessment missing columns: {sorted(req_x-set(x.columns))}")

    if diagnostics:
        payload={"status":"FAIL","diagnostics":diagnostics}
        Path(a.output).write_text(json.dumps(payload,indent=2),encoding="utf-8")
        print(json.dumps(payload,indent=2)); raise SystemExit(1)

    u["ticker"]=u["ticker"].astype(str).str.upper().str.strip()
    x["ticker"]=x["ticker"].astype(str).str.upper().str.strip()
    q["ticker"]=q["ticker"].astype(str).str.upper().str.strip()

    eligible=set(u.loc[bool_series(u["research_universe_eligible"]),"ticker"])
    noneligible=set(u["ticker"])-eligible
    assessed=set(x["ticker"])
    queued=set(q["ticker"])

    if x["ticker"].duplicated().any(): diagnostics.append("duplicate ticker(s) in coverage assessment")
    if q["ticker"].duplicated().any(): diagnostics.append("duplicate ticker(s) in acquisition queue")
    if assessed != eligible:
        miss=sorted(eligible-assessed); extra=sorted(assessed-eligible)
        if miss: diagnostics.append(f"eligible ticker(s) missing from assessment: {miss[:10]}")
        if extra: diagnostics.append(f"assessment ticker(s) outside eligible universe: {extra[:10]}")
    leaked=sorted(queued & noneligible)
    if leaked: diagnostics.append(f"non-eligible ticker(s) leaked into queue: {leaked[:10]}")
    outside=sorted(queued-eligible)
    if outside: diagnostics.append(f"queue ticker(s) outside eligible universe: {outside[:10]}")

    bad_status=sorted(set(x["coverage_status"].dropna().astype(str))-ALLOWED)
    if bad_status: diagnostics.append(f"invalid coverage_status value(s): {bad_status}")

    need=bool_series(x["acquisition_needed"])
    expected=set(x.loc[need,"ticker"])
    if queued != expected:
        missing=sorted(expected-queued); extra=sorted(queued-expected)
        if missing: diagnostics.append(f"acquisition-needed ticker(s) missing from queue: {missing[:10]}")
        if extra: diagnostics.append(f"queue contains ticker(s) not marked acquisition-needed: {extra[:10]}")

    covered=x["coverage_status"].eq("COVERED")
    if (covered & need).any(): diagnostics.append("COVERED row(s) marked acquisition_needed=True")
    if ((~covered) & (~need)).any(): diagnostics.append("non-COVERED row(s) marked acquisition_needed=False")

    analysis_ts=pd.Timestamp(f"{a.analysis_date} {a.analysis_time}")
    parsed_assessment=pd.to_datetime(x["analysis_timestamp"],errors="coerce")
    if parsed_assessment.isna().any() or (parsed_assessment != analysis_ts).any():
        diagnostics.append("assessment analysis_timestamp mismatch or missing")

    s["publication_timestamp"]=pd.to_datetime(s.get("publication_timestamp"),errors="coerce")
    future=int((s["publication_timestamp"]>analysis_ts).sum())

    payload={
        "status":"PASS" if not diagnostics else "FAIL",
        "analysis_timestamp":analysis_ts.isoformat(),
        "eligible_universe":int(len(eligible)),
        "assessment_rows":int(len(x)),
        "covered":int(covered.sum()),
        "queue_rows":int(len(q)),
        "future_source_rows_observed":future,
        "diagnostics":diagnostics
    }
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))
    raise SystemExit(0 if not diagnostics else 1)

if __name__=="__main__":
    main()
