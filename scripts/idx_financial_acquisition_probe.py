from pathlib import Path
import hashlib, json, os, re, time
from urllib.parse import urlencode, urljoin

import pandas as pd

try:
    from curl_cffi import requests
except Exception as exc:
    raise SystemExit(f"curl_cffi is required: {exc}")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "fundamental"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "acquisition_probe.csv"
OUT_STATUS = OUT_DIR / "acquisition_probe_status.json"

BASE = "https://www.idx.co.id/primary/ListedCompany/GetFinancialReport"
YEAR = int(os.getenv("PROBE_YEAR", "2025"))
PERIOD = os.getenv("PROBE_PERIOD", "audit")
REPORT_TYPE = "rdf"
MAX_TICKERS = int(os.getenv("MAX_PROBE_TICKERS", "3"))
DOWNLOAD_SAMPLE = os.getenv("DOWNLOAD_SAMPLE_ATTACHMENT", "1") == "1"
TIMEOUT = 30

DEFAULT_TICKERS = ["BBCA", "TLKM", "ASII"]


def choose_tickers():
    raw = os.getenv("PROBE_TICKERS", "").strip()
    if raw:
        return [x.strip().upper().replace(".JK", "") for x in raw.split(",") if x.strip()][:MAX_TICKERS]
    p = ROOT / "data" / "research_universe.csv"
    if p.exists():
        try:
            df = pd.read_csv(p)
            for col in ["ticker", "Ticker", "code", "Code"]:
                if col in df.columns:
                    vals = df[col].dropna().astype(str).str.upper().str.replace(".JK", "", regex=False).tolist()
                    vals = [v for v in vals if re.fullmatch(r"[A-Z0-9]{2,6}", v)]
                    if vals:
                        return vals[:MAX_TICKERS]
        except Exception:
            pass
    return DEFAULT_TICKERS[:MAX_TICKERS]


def get_json(session, ticker):
    params = {
        "year": YEAR,
        "reportType": REPORT_TYPE,
        "periode": PERIOD,
        "kodeEmiten": ticker,
    }
    url = BASE + "?" + urlencode(params)
    r = session.get(url, timeout=TIMEOUT, headers={"Accept": "application/json,text/plain,*/*", "Referer": "https://www.idx.co.id/"})
    return r, url


def attachment_url(path):
    if not path:
        return ""
    if str(path).startswith("http"):
        return str(path)
    return urljoin("https://www.idx.co.id", str(path))


def main():
    tickers = choose_tickers()
    session = requests.Session(impersonate="chrome", timeout=TIMEOUT)
    rows = []
    status = {
        "status": "PROBE_STARTED",
        "engine_changed": False,
        "year": YEAR,
        "period": PERIOD,
        "tickers": tickers,
        "metadata_success": 0,
        "metadata_failed": 0,
        "attachment_success": 0,
        "attachment_failed": 0,
        "notes": [],
    }

    for ticker in tickers:
        try:
            resp, url = get_json(session, ticker)
            content_type = resp.headers.get("content-type", "")
            row = {
                "ticker": ticker,
                "year": YEAR,
                "period": PERIOD,
                "http_status": resp.status_code,
                "content_type": content_type,
                "metadata_url": url,
                "metadata_ok": False,
                "attachment_count": 0,
                "attachment_names": "",
                "sample_attachment_url": "",
                "sample_attachment_type": "",
                "sample_attachment_http_status": "",
                "sample_attachment_sha256": "",
                "error": "",
            }
            if resp.status_code != 200:
                row["error"] = f"HTTP {resp.status_code}"
                status["metadata_failed"] += 1
                rows.append(row)
                continue

            try:
                payload = resp.json()
            except Exception as exc:
                row["error"] = f"JSON parse failed: {exc}"
                status["metadata_failed"] += 1
                rows.append(row)
                continue

            results = payload.get("Results") or payload.get("results") or []
            if isinstance(results, dict):
                results = [results]
            attachments = []
            for result in results:
                if isinstance(result, dict):
                    attachments.extend(result.get("Attachments") or result.get("attachments") or [])

            row["metadata_ok"] = True
            row["attachment_count"] = len(attachments)
            names = []
            sample = None
            for a in attachments:
                if not isinstance(a, dict):
                    continue
                name = a.get("File_Name") or a.get("file_name") or ""
                ftype = a.get("File_Type") or a.get("file_type") or ""
                path = a.get("File_Path") or a.get("file_path") or ""
                names.append(name)
                low = (name + " " + ftype).lower()
                if sample is None and ("instance.zip" in low or "inline" in low or ftype.lower() == ".xlsx" or name.lower().endswith(".xlsx")):
                    sample = (name, ftype, path)

            row["attachment_names"] = " | ".join(names[:20])
            status["metadata_success"] += 1

            if sample:
                name, ftype, path = sample
                u = attachment_url(path)
                row["sample_attachment_url"] = u
                row["sample_attachment_type"] = ftype or Path(name).suffix
                if DOWNLOAD_SAMPLE and u:
                    try:
                        ar = session.get(u, timeout=TIMEOUT, headers={"Referer": "https://www.idx.co.id/"})
                        row["sample_attachment_http_status"] = ar.status_code
                        if ar.status_code == 200 and ar.content:
                            row["sample_attachment_sha256"] = hashlib.sha256(ar.content).hexdigest()
                            status["attachment_success"] += 1
                        else:
                            status["attachment_failed"] += 1
                    except Exception as exc:
                        row["sample_attachment_http_status"] = "ERROR"
                        row["error"] = f"Attachment fetch failed: {exc}"
                        status["attachment_failed"] += 1
            rows.append(row)
            time.sleep(1.0)
        except Exception as exc:
            rows.append({
                "ticker": ticker, "year": YEAR, "period": PERIOD,
                "http_status": "ERROR", "content_type": "",
                "metadata_url": "", "metadata_ok": False,
                "attachment_count": 0, "attachment_names": "",
                "sample_attachment_url": "", "sample_attachment_type": "",
                "sample_attachment_http_status": "", "sample_attachment_sha256": "",
                "error": repr(exc),
            })
            status["metadata_failed"] += 1

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    if status["metadata_success"] == len(tickers) and status["attachment_success"] > 0:
        status["status"] = "PROBE_SUCCESS_METADATA_AND_ATTACHMENT"
    elif status["metadata_success"] > 0:
        status["status"] = "PROBE_PARTIAL_METADATA_ONLY"
        status["notes"].append("IDX metadata accessible but no sample attachment was successfully downloaded.")
    else:
        status["status"] = "PROBE_BLOCKED_OR_UNAVAILABLE"
        status["notes"].append("Do not invent or substitute financial values. Use an approved IDX export/data-service route if access remains blocked.")

    OUT_STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
