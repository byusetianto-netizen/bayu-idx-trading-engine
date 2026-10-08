import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path("."); V=ROOT/"data/valuation"; V.mkdir(exist_ok=True)
OUT=V/"valuation_snapshot.csv"
def main():
    if not OUT.exists():
        s={"status":"FAIL","checks":{"snapshot_present":False}}
    else:
        d=pd.read_csv(OUT)
        checks={
            "snapshot_present":True,
            "rows":len(d),
            "duplicate_ticker":int(d["ticker"].duplicated().sum()) if "ticker" in d else -1,
            "negative_eps_per_guard":int(((d["eps"]<=0)&d["per"].notna()).sum()) if "eps" in d and "per" in d else -1,
            "negative_bvps_pbv_guard":int(((d["bvps"]<=0)&d["pbv"].notna()).sum()) if "bvps" in d and "pbv" in d else -1,
            "classification_present":int(d["classification"].notna().sum()) if "classification" in d else 0
        }
        ok=checks["snapshot_present"] and checks["duplicate_ticker"]==0 and checks["negative_eps_per_guard"]==0 and checks["negative_bvps_pbv_guard"]==0
        s={"status":"PASS" if ok else "REVIEW","engine_changed":False,"checks":checks}
    (V/"valuation_validation_status.json").write_text(json.dumps(s,indent=2),encoding="utf-8")
    print(json.dumps(s,indent=2))
if __name__=="__main__": main()
