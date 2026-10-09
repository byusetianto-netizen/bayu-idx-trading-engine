from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FUND = ROOT / "data" / "fundamental"

RATIO = FUND / "idx_financial_ratio_snapshots.csv"
EVIDENCE = FUND / "publication_evidence.csv"
OUT = FUND / "idx_financial_ratio_snapshots_pit.csv"
STATUS = FUND / "ratio_pit_enrichment_status.json"

def read(path):
    return pd.read_csv(
        path,
        dtype=str,
        sep=None,
        engine="python",
        encoding="utf-8-sig"
    ).fillna("")

def norm_period(series):
    return (
        pd.to_datetime(series, errors="coerce", utc=True)
        .dt.strftime("%Y-%m-%d")
        .fillna("")
    )

if not RATIO.exists():
    raise FileNotFoundError(RATIO)

if not EVIDENCE.exists():
    raise FileNotFoundError(EVIDENCE)

ratio = read(RATIO)
evidence = read(EVIDENCE)

for name, df in [("ratio", ratio), ("publication_evidence", evidence)]:
    for col in ["ticker", "period_end"]:
        if col not in df.columns:
            raise ValueError(f"{name} missing {col}")

if "publication_timestamp" not in evidence.columns:
    raise ValueError(
        "publication_evidence missing publication_timestamp"
    )

ratio["_tk"] = ratio["ticker"].str.strip().str.upper()
ratio["_pk"] = norm_period(ratio["period_end"])

evidence["_tk"] = evidence["ticker"].str.strip().str.upper()
evidence["_pk"] = norm_period(evidence["period_end"])
evidence["_tsraw"] = evidence["publication_timestamp"].str.strip()
evidence["_ts"] = pd.to_datetime(
    evidence["_tsraw"],
    errors="coerce",
    utc=True
)
evidence["_valid"] = evidence["_ts"].notna()

# One authoritative evidence record per ticker + period.
# Prefer valid/latest timestamp when duplicate evidence exists.
evidence = (
    evidence
    .sort_values(
        ["_tk", "_pk", "_valid", "_ts"],
        na_position="first"
    )
    .drop_duplicates(["_tk", "_pk"], keep="last")
)

merged = ratio.merge(
    evidence[["_tk", "_pk", "_tsraw", "_ts"]],
    on=["_tk", "_pk"],
    how="left"
)

merged["publication_timestamp"] = merged["_tsraw"].fillna("")

period_ok = pd.to_datetime(
    merged["period_end"],
    errors="coerce",
    utc=True
).notna()

timestamp_ok = merged["_ts"].notna()

# Authoritative PIT readiness.
# Existing ratio pit_ready/publication_date are NOT evidence fallbacks.
merged["pit_ready"] = period_ok & timestamp_ok

merged = merged.drop(
    columns=["_tk", "_pk", "_tsraw", "_ts"]
)

cols = list(merged.columns)

if "publication_timestamp" in cols:
    cols.remove("publication_timestamp")

if "pit_ready" in cols:
    cols.remove("pit_ready")

insert_at = (
    cols.index("publication_date") + 1
    if "publication_date" in cols
    else cols.index("period_end") + 1
)

cols = (
    cols[:insert_at]
    + ["publication_timestamp", "pit_ready"]
    + cols[insert_at:]
)

merged = merged[cols]

OUT.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUT, index=False)

raw_ts = merged["publication_timestamp"].astype(str).str.strip()
parsed_ts = pd.to_datetime(
    raw_ts,
    errors="coerce",
    utc=True
)

missing = int(raw_ts.eq("").sum())
invalid = int((raw_ts.ne("") & parsed_ts.isna()).sum())
eligible = int(merged["pit_ready"].sum())

status = {
    "status": "PASS" if invalid == 0 else "REVIEW",
    "ratio_rows": int(len(merged)),
    "evidence_rows": int(len(evidence)),
    "pit_ready_rows": eligible,
    "blocked_rows": int(len(merged) - eligible),
    "publication_timestamp_missing_rows": missing,
    "publication_timestamp_invalid_rows": invalid,
    "join_key": "ticker + period_end",
    "ticker_only_fallback": False,
    "publication_date_fallback": False,
    "period_end_used_as_publication_date": False,
    "raw_pit_ready_trusted_as_timestamp_evidence": False,
    "notes": [
        "publication_timestamp from publication_evidence.csv is authoritative.",
        "Ratio publication_date is metadata only and is never converted into PIT timestamp.",
        "Rows without matching valid timestamp evidence remain pit_ready=false.",
        "Historical cutoff remains downstream: publication_timestamp <= analysis_timestamp."
    ]
}

STATUS.write_text(
    json.dumps(status, indent=2),
    encoding="utf-8"
)

print(json.dumps(status, indent=2))
