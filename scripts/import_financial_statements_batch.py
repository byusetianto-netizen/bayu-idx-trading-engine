from pathlib import Path
import json, re, hashlib
from datetime import datetime
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INBOX = ROOT / "data/fundamental/inbox"
OUT = ROOT / "data/fundamental"
STATEMENTS = OUT / "financial_statements.csv"
AUDIT = OUT / "fundamental_import_audit.csv"
STATUS = OUT / "fundamental_import_status.json"
EVIDENCE = OUT / "publication_evidence.csv"

REQUIRED = [
    "ticker","metric","value","unit","currency",
    "period_start","period_end","period_type","is_cumulative",
    "publication_date","report_date","source","source_url",
    "document_id","version","confidence"
]

METRIC_PATTERNS = {
    "revenue": [r"revenue", r"sales", r"pendapatan", r"penjualan"],
    "gross_profit": [r"gross profit", r"laba bruto", r"laba kotor"],
    "profit_before_tax": [r"profit.*before tax", r"laba.*sebelum pajak"],
    "net_income": [r"net income", r"profit for the period", r"laba.*tahun berjalan", r"laba.*periode berjalan"],
    "net_income_parent": [r"attributable.*parent", r"yang dapat diatribusikan.*entitas induk"],
    "total_assets": [r"total assets", r"jumlah aset", r"total aset"],
    "total_liabilities": [r"total liabilities", r"jumlah liabilitas", r"total liabilitas"],
    "total_equity": [r"total equity", r"jumlah ekuitas", r"total ekuitas"],
    "cash_from_operations": [r"cash flows from operating", r"arus kas.*operasi"],
    "cash_from_investing": [r"cash flows from investing", r"arus kas.*investasi"],
    "cash_from_financing": [r"cash flows from financing", r"arus kas.*pendanaan"],
    "eps": [r"earnings per share", r"laba per saham", r"basic earnings per share"],
    "bvps": [r"book value per share", r"nilai buku per saham"],
}

def clean_text(x):
    return re.sub(r"\s+", " ", str(x or "")).strip()

def ticker_from_name(path):
    m = re.search(r"-([A-Z]{4})(?:\.[A-Z]+)?(?:\(|\.|$)", path.stem.upper())
    return m.group(1) if m else ""

def metric_key(label):
    s = clean_text(label).lower()
    for k, pats in METRIC_PATTERNS.items():
        if any(re.search(p, s) for p in pats):
            return k
    return None

def to_number(x):
    if pd.isna(x): return None
    if isinstance(x, (int,float)): return float(x)
    s = str(x).strip().replace(",", "")
    if s in ("", "-", "—", "nan", "None"): return None
    s = re.sub(r"[^\d\.\-\(\)]", "", s)
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try: return float(s)
    except: return None

def find_general_info(xls):
    info = {}
    for sh in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sh, header=None, nrows=120)
        except Exception:
            continue
        for i in range(len(df)):
            row = [clean_text(v) for v in df.iloc[i].tolist()]
            joined = " | ".join(row).lower()
            if "entity name" in joined or "nama entitas" in joined:
                for j,v in enumerate(row):
                    if "entity name" in v.lower() or "nama entitas" in v.lower():
                        if j+1 < len(row) and row[j+1]:
                            info["company_name"] = row[j+1]
            if "stock code" in joined or "kode saham" in joined:
                for j,v in enumerate(row):
                    if "stock code" in v.lower() or "kode saham" in v.lower():
                        if j+1 < len(row) and row[j+1]:
                            info["ticker"] = re.sub(r"[^A-Z]", "", row[j+1].upper())
            if "current period" in joined or "periode berjalan" in joined:
                # look for date-like values in row
                dates = pd.to_datetime(pd.Series(row), errors="coerce", dayfirst=False, format="mixed")
                vals = [d for d in dates if pd.notna(d)]
                if vals:
                    info["period_end"] = max(vals).date().isoformat()
        if info.get("ticker") or info.get("company_name"):
            break
    return info

def extract_file(path):
    xls = pd.ExcelFile(path)
    info = find_general_info(xls)
    ticker = info.get("ticker") or ticker_from_name(path)
    rows = []
    period_end = info.get("period_end")
    if not period_end:
        # fallback: find latest date in first few sheets
        for sh in xls.sheet_names[:10]:
            try:
                df = pd.read_excel(xls, sheet_name=sh, header=None, nrows=80)
            except: continue
            vals = pd.to_datetime(df.astype(str).stack(), errors="coerce", dayfirst=False, format="mixed").dropna()
            if len(vals):
                period_end = max(vals).date().isoformat()
                break

    for sh in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sh, header=None)
        except Exception:
            continue
        for i in range(len(df)):
            label = clean_text(df.iloc[i,0]) if len(df.columns) else ""
            mk = metric_key(label)
            if not mk:
                continue
            nums = []
            for v in df.iloc[i,1:].tolist():
                n = to_number(v)
                if n is not None:
                    nums.append(n)
            if not nums:
                continue
            value = nums[0]
            rows.append({
                "ticker": ticker,
                "metric": mk,
                "value": value,
                "unit": "reported",
                "currency": "IDR",
                "period_start": "",
                "period_end": period_end or "",
                "period_type": "UNKNOWN",
                "is_cumulative": "",
                "publication_date": "",
                "report_date": "",
                "source": "IDX FinancialStatement XLSX",
                "source_url": "",
                "document_id": hashlib.sha256(path.read_bytes()).hexdigest(),
                "version": path.name,
                "confidence": "MEDIUM",
            })
    return ticker, rows

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(INBOX.glob("*.xlsx")) + sorted(INBOX.glob("*.xls"))
    imported = []
    audit = []

    for path in files:
        try:
            ticker, rows = extract_file(path)
            imported.extend(rows)
            audit.append({
                "file": path.name,
                "ticker": ticker,
                "rows_extracted": len(rows),
                "status": "IMPORTED" if rows else "NO_METRICS_FOUND",
                "error": "",
            })
        except Exception as e:
            audit.append({
                "file": path.name,
                "ticker": ticker_from_name(path),
                "rows_extracted": 0,
                "status": "ERROR",
                "error": str(e)[:500],
            })

    # CLEAN REBUILD:
    # Reconstruct the financial statement table exclusively from the XLSX files
    # currently present in inbox. Do not merge the existing CSV, because an
    # earlier importer version could have left duplicate rows behind.
    combined = pd.DataFrame(imported, columns=REQUIRED)

    if not combined.empty:
        for c in REQUIRED:
            if c not in combined.columns:
                combined[c] = ""

        for c in ["ticker","metric","period_end","publication_date","document_id"]:
            combined[c] = combined[c].fillna("").astype(str).str.strip()

        combined["value"] = pd.to_numeric(combined["value"], errors="coerce")

        # Deduplicate by source document + financial identity. If the same
        # metric appears more than once inside an XLSX, retain one record.
        combined = combined[REQUIRED].drop_duplicates(
            subset=["ticker","metric","period_end","document_id"],
            keep="first"
        )

        combined.to_csv(STATEMENTS, index=False)
    else:
        # Explicitly rebuild an empty table if no source files are present.
        pd.DataFrame(columns=REQUIRED).to_csv(STATEMENTS, index=False)

    pd.DataFrame(audit).to_csv(AUDIT, index=False)

    status = {
        "status": "IMPORT_COMPLETE" if not any(a["status"]=="ERROR" for a in audit) else "IMPORT_PARTIAL",
        "engine_changed": False,
        "files_found": len(files),
        "files_imported": sum(a["status"]=="IMPORTED" for a in audit),
        "files_with_errors": sum(a["status"]=="ERROR" for a in audit),
        "rows_extracted": len(imported),
        "financial_statement_rows_total": int(len(combined)) if not combined.empty else 0,
        "tickers_with_financial_data": int(combined["ticker"].nunique()) if not combined.empty else 0,
        "notes": [
            "No financial values are fabricated.",
            "Publication date is not inferred from period_end.",
            "Rows without publication evidence are not PIT-ready.",
            "Importer is research-only and does not modify actual_engine.py."
        ]
    }
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
