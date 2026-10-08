from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FS = ROOT / "data" / "fundamental" / "financial_statements.csv"
EVIDENCE = ROOT / "data" / "fundamental" / "publication_evidence.csv"
OUT = ROOT / "data" / "fundamental" / "financial_statements_pit.csv"
AUDIT = ROOT / "data" / "fundamental" / "pit_reconciliation_audit.csv"
STATUS = ROOT / "data" / "fundamental" / "pit_reconciliation_status.json"

def read_csv_flexible(path):
    return pd.read_csv(path, dtype=str, sep=None, engine="python", encoding="utf-8-sig").fillna("")

def norm_date(s):
    return pd.to_datetime(s, errors="coerce", utc=True).dt.strftime("%Y-%m-%d").fillna("")

if not FS.exists():
    raise FileNotFoundError(f"Missing financial statements: {FS}")
if not EVIDENCE.exists():
    raise FileNotFoundError(f"Missing publication evidence: {EVIDENCE}")

fs = read_csv_flexible(FS)
ev = read_csv_flexible(EVIDENCE)

for name, df in [("financial_statements", fs), ("publication_evidence", ev)]:
    if "ticker" not in df.columns:
        raise ValueError(f"{name} is missing required column: ticker")
    if "period_end" not in df.columns:
        raise ValueError(f"{name} is missing required column: period_end")

# Normalize join keys. Do NOT match ticker-only.
fs["_ticker_key"] = fs["ticker"].astype(str).str.strip().str.upper()
ev["_ticker_key"] = ev["ticker"].astype(str).str.strip().str.upper()
fs["_period_key"] = norm_date(fs["period_end"])
ev["_period_key"] = norm_date(ev["period_end"])

# Accept either publication_date or publication_timestamp.
pub_date_col = "publication_date" if "publication_date" in ev.columns else (
    "publication_timestamp" if "publication_timestamp" in ev.columns else ""
)
pub_time_col = "publication_time" if "publication_time" in ev.columns else ""

if not pub_date_col:
    raise ValueError("publication_evidence is missing publication_date/publication_timestamp")

ev["_pub_date"] = ev[pub_date_col].astype(str).str.strip()
ev["_pub_time"] = ev[pub_time_col].astype(str).str.strip() if pub_time_col else ""

# Build a single publication timestamp, preserving the supplied time when available.
def combine_pub(row):
    d = row["_pub_date"]
    t = row["_pub_time"]
    if not d:
        return ""
    if t:
        return f"{d} {t}"
    return d

ev["_publication_timestamp"] = ev.apply(combine_pub, axis=1)

# Deduplicate evidence deterministically by ticker + period.
ev = ev.sort_values(["_ticker_key", "_period_key", "_publication_timestamp"])
ev = ev.drop_duplicates(["_ticker_key", "_period_key"], keep="last")

# Keep all financial rows and left-join evidence.
merged = fs.merge(
    ev[["_ticker_key", "_period_key", "_publication_timestamp", "_pub_date", "_pub_time"]],
    on=["_ticker_key", "_period_key"],
    how="left",
    suffixes=("", "_evidence"),
)

merged["publication_timestamp"] = merged["_publication_timestamp"].astype(str).replace("nan", "")
merged["publication_date"] = merged["_pub_date"].astype(str).replace("nan", "")
merged["publication_time"] = merged["_pub_time"].astype(str).replace("nan", "")

# A publication date is availability evidence. Never infer it from period_end.
merged["pit_ready"] = merged["publication_date"].ne("") & merged["period_end"].ne("")

# Remove internal keys from published output.
drop_cols = ["_ticker_key", "_period_key", "_publication_timestamp", "_pub_date", "_pub_time"]
merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

# Reorder publication fields after period_end when possible.
cols = list(merged.columns)
for c in ["publication_date", "publication_time", "publication_timestamp", "pit_ready"]:
    if c in cols:
        cols.remove(c)
insert_at = cols.index("period_end") + 1 if "period_end" in cols else len(cols)
new_cols = cols[:insert_at] + [c for c in ["publication_date", "publication_time", "publication_timestamp", "pit_ready"] if c in merged.columns] + cols[insert_at:]
merged = merged[new_cols]

OUT.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUT, index=False, encoding="utf-8")

# Audit at ticker-period level.
audit = (
    merged.groupby(["ticker", "period_end"], dropna=False)
    .agg(
        financial_rows=("ticker", "size"),
        publication_date=("publication_date", "first"),
        publication_time=("publication_time", "first"),
        pit_ready=("pit_ready", "all"),
    )
    .reset_index()
)
audit["match_status"] = audit["publication_date"].apply(lambda x: "MATCHED" if str(x).strip() else "MISSING_EVIDENCE")
audit.to_csv(AUDIT, index=False, encoding="utf-8")

pit_ready_rows = int(merged["pit_ready"].sum())
pit_ready_tickers = int(merged.loc[merged["pit_ready"], "ticker"].nunique())
missing_pub = int(merged["publication_date"].eq("").sum())

status = {
    "status": "PASS" if missing_pub == 0 else "REVIEW",
    "engine_changed": False,
    "financial_rows": int(len(merged)),
    "tickers": int(merged["ticker"].nunique()),
    "evidence_rows": int(len(ev)),
    "pit_ready_rows": pit_ready_rows,
    "pit_ready_tickers": pit_ready_tickers,
    "publication_missing_rows": missing_pub,
    "join_key": "ticker + period_end",
    "ticker_only_fallback": False,
    "period_end_used_as_publication_date": False,
    "notes": [
        "Publication evidence is matched strictly by ticker + period_end.",
        "All financial statement rows are preserved by a left join.",
        "Publication date is never inferred from period_end.",
        "Publication time is optional."
    ],
}

STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
print(json.dumps(status, indent=2))
