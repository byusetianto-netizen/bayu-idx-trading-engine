#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd

CRITICAL = {
    "revenue","net_income","total_assets","total_liabilities",
    "total_equity","cash_from_operations"
}

def bool_series(s):
    return s.astype(str).str.strip().str.lower().isin(["true","1","yes","y"])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--universe",default="data/research_universe.csv")
    ap.add_argument("--source",default="data/fundamental/financial_statements_pit.csv")
    ap.add_argument("--analysis-date",required=True)
    ap.add_argument("--analysis-time",default="23:59:59")
    ap.add_argument("--assessment-output",default="data/fundamental/fundamental_coverage_assessment.csv")
    ap.add_argument("--queue-output",default="data/fundamental/acquisition_queue.csv")
    ap.add_argument("--status-output",default="data/fundamental/coverage_expansion_status.json")
    a=ap.parse_args()

    u=pd.read_csv(a.universe)
    s=pd.read_csv(a.source)
    req_u={"ticker","research_universe_eligible"}
    req_s={"ticker","metric","period_end","publication_timestamp","document_id","confidence"}
    if not req_u.issubset(u.columns):
        raise SystemExit(f"Universe missing columns: {sorted(req_u-set(u.columns))}")
    if not req_s.issubset(s.columns):
        raise SystemExit(f"Source missing columns: {sorted(req_s-set(s.columns))}")

    u["ticker"]=u["ticker"].astype(str).str.upper().str.strip()
    eligible=sorted(u.loc[bool_series(u["research_universe_eligible"]),"ticker"].dropna().unique())
    analysis_ts=pd.Timestamp(f"{a.analysis_date} {a.analysis_time}")

    s["ticker"]=s["ticker"].astype(str).str.upper().str.strip()
    s["metric"]=s["metric"].astype(str).str.strip()
    s["publication_timestamp"]=pd.to_datetime(s["publication_timestamp"],errors="coerce")
    s["period_end"]=pd.to_datetime(s["period_end"],errors="coerce")

    # Fail closed: missing publication_timestamp is never replaced by publication_date.
    pit=s[s["publication_timestamp"].notna() & (s["publication_timestamp"]<=analysis_ts)].copy()

    rows=[]
    for t in eligible:
        z=pit[pit["ticker"].eq(t)]
        if z.empty:
            rows.append(dict(
                ticker=t,analysis_timestamp=analysis_ts.isoformat(),
                coverage_status="MISSING",latest_period_end="",
                critical_metrics_present=0,critical_metrics_required=len(CRITICAL),
                missing_critical_metrics=";".join(sorted(CRITICAL)),
                acquisition_needed=True,reason="NO_PIT_SAFE_FINANCIAL_STATEMENT"))
            continue

        # Anchor the current coverage period on total_assets.
        ends=z.loc[z["metric"].eq("total_assets"),"period_end"].dropna()
        if ends.empty:
            rows.append(dict(
                ticker=t,analysis_timestamp=analysis_ts.isoformat(),
                coverage_status="INCOMPLETE",latest_period_end="",
                critical_metrics_present=0,critical_metrics_required=len(CRITICAL),
                missing_critical_metrics=";".join(sorted(CRITICAL)),
                acquisition_needed=True,reason="NO_ANCHOR_PERIOD"))
            continue

        e=ends.max()
        q=z[z["period_end"].eq(e)]
        present=set(q["metric"].dropna().astype(str))
        missing=sorted(CRITICAL-present)
        if missing:
            rows.append(dict(
                ticker=t,analysis_timestamp=analysis_ts.isoformat(),
                coverage_status="INCOMPLETE",latest_period_end=e.date().isoformat(),
                critical_metrics_present=len(CRITICAL)-len(missing),
                critical_metrics_required=len(CRITICAL),
                missing_critical_metrics=";".join(missing),
                acquisition_needed=True,reason="MISSING_CRITICAL_METRICS"))
        else:
            rows.append(dict(
                ticker=t,analysis_timestamp=analysis_ts.isoformat(),
                coverage_status="COVERED",latest_period_end=e.date().isoformat(),
                critical_metrics_present=len(CRITICAL),critical_metrics_required=len(CRITICAL),
                missing_critical_metrics="",acquisition_needed=False,reason=""))

    assessment=pd.DataFrame(rows)
    queue=assessment[bool_series(assessment["acquisition_needed"])].copy()
    for p in [a.assessment_output,a.queue_output,a.status_output]:
        Path(p).parent.mkdir(parents=True,exist_ok=True)
    assessment.to_csv(a.assessment_output,index=False)
    queue.to_csv(a.queue_output,index=False)

    counts=assessment["coverage_status"].value_counts().to_dict()
    payload={
        "status":"BUILT","analysis_timestamp":analysis_ts.isoformat(),
        "universe_rows":int(len(u)),"eligible_tickers":int(len(eligible)),
        "covered":int(counts.get("COVERED",0)),
        "incomplete":int(counts.get("INCOMPLETE",0)),
        "missing":int(counts.get("MISSING",0)),
        "acquisition_queue_rows":int(len(queue)),
        "noneligible_excluded":int(len(set(u["ticker"])-set(eligible))),
        "engine_changed":False,
        "notes":[
            "Coverage expansion only; no score, ranking, or trade decision.",
            "publication_timestamp is authoritative for PIT; publication_date never substitutes for a missing timestamp.",
            "Only research_universe_eligible=True tickers are acquisition targets.",
            "Coverage is assessed on the latest PIT-safe period anchored by total_assets."
        ]
    }
    Path(a.status_output).write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))

if __name__=="__main__":
    main()
