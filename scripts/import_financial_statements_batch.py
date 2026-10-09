from pathlib import Path
import json
import re
import hashlib
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
    "ticker",
    "metric",
    "value",
    "unit",
    "currency",
    "period_start",
    "period_end",
    "period_type",
    "is_cumulative",
    "publication_date",
    "report_date",
    "source",
    "source_url",
    "document_id",
    "version",
    "confidence",
]


# ---------------------------------------------------------------------
# V2.1E deterministic financial-statement extraction contract
#
# Do NOT scan arbitrary disclosure sheets.
# Do NOT use broad keyword matching.
# Do NOT take the first numeric value found in a row.
#
# Each metric is allowed only inside its primary-statement family.
# Column B is accepted only when its context header explicitly identifies
# the current reporting period.
# ---------------------------------------------------------------------

STATEMENT_FAMILIES = {
    "balance_sheet": [
        "1210000",
        "1220000",
    ],
    "income_statement": [
        "1311000",
        "1312000",
        "1321000",
        "1322000",
    ],
    "cash_flow": [
        "1510000",
        "1520000",
    ],
}


CURRENT_CONTEXTS = {
    "balance_sheet": {
        "CurrentYearInstant",
    },
    "income_statement": {
        "CurrentYearDuration",
    },
    "cash_flow": {
        "CurrentYearDuration",
        "CurrentYearInstant",
    },
}


METRIC_SPECS = {
    "revenue": {
        "family": "income_statement",
        "labels": {
            "Sales and revenue",
        },
    },
    "gross_profit": {
        "family": "income_statement",
        "labels": {
            "Total gross profit",
        },
    },
    "profit_before_tax": {
        "family": "income_statement",
        "labels": {
            "Total profit (loss) before tax",
        },
    },
    "net_income": {
        "family": "income_statement",
        "labels": {
            "Total profit (loss)",
        },
    },
    "net_income_parent": {
        "family": "income_statement",
        "labels": {
            "Profit (loss) attributable to parent entity",
        },
    },
    "total_assets": {
        "family": "balance_sheet",
        "labels": {
            "Total assets",
        },
    },
    "total_liabilities": {
        "family": "balance_sheet",
        "labels": {
            "Total liabilities",
        },
    },
    "total_equity": {
        "family": "balance_sheet",
        "labels": {
            "Total equity",
        },
    },
    "cash_from_operations": {
        "family": "cash_flow",
        "labels": {
            "Total net cash flows received from (used in) operating activities",
        },
    },
    "cash_from_investing": {
        "family": "cash_flow",
        "labels": {
            "Total net cash flows received from (used in) investing activities",
        },
    },
    "cash_from_financing": {
        "family": "cash_flow",
        "labels": {
            "Total net cash flows received from (used in) financing activities",
        },
    },
}


def clean_text(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def ticker_from_name(path):
    m = re.search(
        r"-([A-Z]{4})(?:\.[A-Z]+)?(?:\(|\.|$)",
        path.stem.upper(),
    )
    return m.group(1) if m else ""


def to_number(x):
    if pd.isna(x):
        return None

    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).strip()

    if s in ("", "-", "—", "nan", "None"):
        return None

    negative = s.startswith("(") and s.endswith(")")

    if negative:
        s = s[1:-1]

    s = s.replace(",", "")
    s = re.sub(r"[^\d.\-]", "", s)

    if s in ("", "-", "."):
        return None

    try:
        value = float(s)
        return -value if negative else value
    except Exception:
        return None


def find_general_info(xls):
    """
    Read issuer/report metadata only.

    This function must NOT be used as publication evidence.
    period_end is a reporting-period field, not a publication timestamp.
    """

    info = {}

    for sh in xls.sheet_names:
        try:
            df = pd.read_excel(
                xls,
                sheet_name=sh,
                header=None,
                nrows=120,
            )
        except Exception:
            continue

        for i in range(len(df)):
            row = [clean_text(v) for v in df.iloc[i].tolist()]
            joined = " | ".join(row).lower()

            if "entity name" in joined or "nama entitas" in joined:
                for j, v in enumerate(row):
                    lv = v.lower()

                    if "entity name" in lv or "nama entitas" in lv:
                        if j + 1 < len(row) and row[j + 1]:
                            info["company_name"] = row[j + 1]

            if "stock code" in joined or "kode saham" in joined:
                for j, v in enumerate(row):
                    lv = v.lower()

                    if "stock code" in lv or "kode saham" in lv:
                        if j + 1 < len(row) and row[j + 1]:
                            info["ticker"] = re.sub(
                                r"[^A-Z]",
                                "",
                                row[j + 1].upper(),
                            )

            if "current period" in joined or "periode berjalan" in joined:
                dates = pd.to_datetime(
                    pd.Series(row),
                    errors="coerce",
                    dayfirst=False,
                    format="mixed",
                )

                vals = [d for d in dates if pd.notna(d)]

                if vals:
                    info["period_end"] = max(vals).date().isoformat()

        if info.get("ticker") or info.get("company_name"):
            break

    return info


def detect_period_end(xls, info):
    period_end = info.get("period_end")

    if period_end:
        return period_end

    # Legacy metadata fallback retained only for report-period detection.
    # It is NEVER publication evidence.
    for sh in xls.sheet_names[:10]:
        try:
            df = pd.read_excel(
                xls,
                sheet_name=sh,
                header=None,
                nrows=80,
            )
        except Exception:
            continue

        vals = pd.to_datetime(
            df.astype(str).stack(),
            errors="coerce",
            dayfirst=False,
            format="mixed",
        ).dropna()

        if len(vals):
            return max(vals).date().isoformat()

    return ""


def read_statement_sheet(xls, sheet_name):
    if sheet_name not in xls.sheet_names:
        return None

    try:
        return pd.read_excel(
            xls,
            sheet_name=sheet_name,
            header=None,
        )
    except Exception:
        return None


def current_context(df):
    """
    IDX primary statements observed in the audited workbooks use:
      column A = Indonesian label
      column B = current-period fact
      column C = comparative fact
      column D = English label

    Row 4 in Excel corresponds to dataframe index 3.
    """

    if df is None:
        return ""

    if len(df) < 4 or len(df.columns) < 2:
        return ""

    return clean_text(df.iloc[3, 1])


def extract_metric_candidates(xls, metric, spec):
    family = spec["family"]
    allowed_contexts = CURRENT_CONTEXTS[family]
    labels = spec["labels"]

    candidates = []

    for sheet_name in STATEMENT_FAMILIES[family]:
        df = read_statement_sheet(xls, sheet_name)

        if df is None:
            continue

        context = current_context(df)

        if context not in allowed_contexts:
            continue

        if len(df.columns) < 4:
            continue

        for i in range(len(df)):
            english_label = clean_text(df.iloc[i, 3])

            if english_label not in labels:
                continue

            value = to_number(df.iloc[i, 1])

            if value is None:
                continue

            candidates.append({
                "metric": metric,
                "sheet": sheet_name,
                "row": i + 1,
                "cell": f"B{i + 1}",
                "context": context,
                "label": english_label,
                "value": value,
            })

    return candidates


def validate_balance_sheet(metric_values):
    required = [
        "total_assets",
        "total_liabilities",
        "total_equity",
    ]

    if not all(k in metric_values for k in required):
        return {
            "status": "BLOCKED",
            "difference": None,
            "reason": "BALANCE_METRICS_MISSING",
        }

    assets = metric_values["total_assets"]
    liabilities = metric_values["total_liabilities"]
    equity = metric_values["total_equity"]

    difference = assets - (liabilities + equity)

    # Values are reported units and may occasionally contain very small
    # floating conversion noise. No financial value is adjusted.
    tolerance = max(
        0.01,
        abs(assets) * 1e-9,
    )

    if abs(difference) <= tolerance:
        return {
            "status": "PASS",
            "difference": difference,
            "reason": "",
        }

    return {
        "status": "FAIL",
        "difference": difference,
        "reason": "ACCOUNTING_IDENTITY_MISMATCH",
    }


def extract_file(path):
    xls = pd.ExcelFile(path)

    info = find_general_info(xls)

    ticker = info.get("ticker") or ticker_from_name(path)
    period_end = detect_period_end(xls, info)

    document_id = hashlib.sha256(path.read_bytes()).hexdigest()

    selected = {}
    diagnostics = []

    for metric, spec in METRIC_SPECS.items():
        candidates = extract_metric_candidates(
            xls,
            metric,
            spec,
        )

        diagnostics.append({
            "metric": metric,
            "candidate_count": len(candidates),
            "candidates": candidates,
        })

        if len(candidates) != 1:
            raise ValueError(
                f"{metric}: expected exactly 1 current-period "
                f"candidate, found {len(candidates)}; "
                f"candidates={candidates}"
            )

        selected[metric] = candidates[0]

    metric_values = {
        metric: item["value"]
        for metric, item in selected.items()
    }

    balance = validate_balance_sheet(metric_values)

    if balance["status"] != "PASS":
        raise ValueError(
            "Balance sheet validation failed: "
            f"status={balance['status']}, "
            f"difference={balance['difference']}, "
            f"reason={balance['reason']}"
        )

    rows = []

    for metric, item in selected.items():
        rows.append({
            "ticker": ticker,
            "metric": metric,
            "value": item["value"],
            "unit": "reported",
            "currency": "IDR",
            "period_start": "",
            "period_end": period_end,
            "period_type": "UNKNOWN",
            "is_cumulative": "",
            "publication_date": "",
            "report_date": "",
            "source": "IDX FinancialStatement XLSX",
            "source_url": "",
            "document_id": document_id,
            "version": path.name,
            "confidence": "HIGH",
        })

    extraction_meta = {
        "balance_check": balance["status"],
        "balance_difference": balance["difference"],
        "metric_sources": {
            metric: (
                item["sheet"]
                + ":"
                + item["cell"]
            )
            for metric, item in selected.items()
        },
    }

    return ticker, rows, extraction_meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    files = (
        sorted(INBOX.glob("*.xlsx"))
        + sorted(INBOX.glob("*.xls"))
    )

    imported = []
    audit = []

    for path in files:
        try:
            ticker, rows, meta = extract_file(path)

            imported.extend(rows)

            audit.append({
                "file": path.name,
                "ticker": ticker,
                "rows_extracted": len(rows),
                "status": "IMPORTED",
                "balance_check": meta["balance_check"],
                "balance_difference": meta["balance_difference"],
                "metric_sources": json.dumps(
                    meta["metric_sources"],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "error": "",
            })

        except Exception as e:
            audit.append({
                "file": path.name,
                "ticker": ticker_from_name(path),
                "rows_extracted": 0,
                "status": "BLOCKED",
                "balance_check": "BLOCKED",
                "balance_difference": "",
                "metric_sources": "",
                "error": str(e)[:1000],
            })

    # ---------------------------------------------------------------
    # FAIL CLOSED
    #
    # A partially successful batch must NOT replace the canonical
    # financial-statements table. This prevents a bad/unsupported
    # workbook from silently removing or contaminating canonical data.
    # ---------------------------------------------------------------

    blocked = [
        a for a in audit
        if a["status"] != "IMPORTED"
    ]

    if blocked:
        pd.DataFrame(audit).to_csv(
            AUDIT,
            index=False,
        )

        status = {
            "status": "IMPORT_BLOCKED",
            "engine_changed": False,
            "files_found": len(files),
            "files_imported": sum(
                a["status"] == "IMPORTED"
                for a in audit
            ),
            "files_blocked": len(blocked),
            "rows_extracted": len(imported),
            "financial_statement_rows_total": None,
            "tickers_with_financial_data": None,
            "canonical_written": False,
            "notes": [
                "Batch failed closed.",
                "financial_statements.csv was not replaced.",
                "No financial values are fabricated.",
                "Only exact primary-statement labels are accepted.",
                "Only explicit current-period contexts are accepted.",
                "Each required metric must have exactly one candidate.",
                "Balance sheet accounting identity must pass.",
                "Publication date is not inferred from period_end.",
                "Rows without publication evidence are not PIT-ready.",
                "Importer does not modify actual_engine.py.",
            ],
        }

        STATUS.write_text(
            json.dumps(status, indent=2),
            encoding="utf-8",
        )

        print(json.dumps(status, indent=2))

        raise SystemExit(1)

    # ---------------------------------------------------------------
    # CLEAN REBUILD
    # ---------------------------------------------------------------

    combined = pd.DataFrame(
        imported,
        columns=REQUIRED,
    )

    if not combined.empty:
        for c in REQUIRED:
            if c not in combined.columns:
                combined[c] = ""

        for c in [
            "ticker",
            "metric",
            "period_end",
            "publication_date",
            "document_id",
        ]:
            combined[c] = (
                combined[c]
                .fillna("")
                .astype(str)
                .str.strip()
            )

        combined["value"] = pd.to_numeric(
            combined["value"],
            errors="coerce",
        )

        # At this stage duplicates are an integrity failure.
        duplicate_mask = combined.duplicated(
            subset=[
                "ticker",
                "metric",
                "period_end",
                "document_id",
            ],
            keep=False,
        )

        if duplicate_mask.any():
            duplicate_rows = combined.loc[
                duplicate_mask,
                [
                    "ticker",
                    "metric",
                    "period_end",
                    "version",
                ],
            ].to_dict("records")

            pd.DataFrame(audit).to_csv(
                AUDIT,
                index=False,
            )

            status = {
                "status": "IMPORT_BLOCKED",
                "engine_changed": False,
                "files_found": len(files),
                "files_imported": len(audit),
                "files_blocked": 0,
                "rows_extracted": len(imported),
                "canonical_written": False,
                "error": (
                    "Duplicate financial identity detected: "
                    + str(duplicate_rows[:20])
                ),
            }

            STATUS.write_text(
                json.dumps(status, indent=2),
                encoding="utf-8",
            )

            print(json.dumps(status, indent=2))

            raise SystemExit(1)

        combined = combined[REQUIRED]

        combined.to_csv(
            STATEMENTS,
            index=False,
        )

    else:
        pd.DataFrame(
            columns=REQUIRED,
        ).to_csv(
            STATEMENTS,
            index=False,
        )

    pd.DataFrame(audit).to_csv(
        AUDIT,
        index=False,
    )

    status = {
        "status": "IMPORT_COMPLETE",
        "engine_changed": False,
        "files_found": len(files),
        "files_imported": len(audit),
        "files_blocked": 0,
        "rows_extracted": len(imported),
        "financial_statement_rows_total": int(len(combined)),
        "tickers_with_financial_data": (
            int(combined["ticker"].nunique())
            if not combined.empty
            else 0
        ),
        "canonical_written": True,
        "extraction_contract": (
            "PRIMARY_STATEMENT_EXACT_LABEL_CURRENT_CONTEXT"
        ),
        "notes": [
            "No financial values are fabricated.",
            "Only primary financial-statement families are scanned.",
            "Only exact English taxonomy labels are accepted.",
            "Only explicit current-period contexts are accepted.",
            "Each required metric must have exactly one candidate.",
            "Balance sheet accounting identity passed for every file.",
            "Publication date is not inferred from period_end.",
            "Rows without publication evidence are not PIT-ready.",
            "Importer does not modify actual_engine.py.",
        ],
    }

    STATUS.write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
