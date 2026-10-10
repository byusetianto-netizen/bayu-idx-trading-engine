from pathlib import Path
import csv
import json
import os
import re
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fundamental"

TICKER_FILE = DATA / "research_tickers.txt"

AUTO_CANDIDATES = DATA / "publication_auto_candidates.csv"
STATUS = DATA / "publication_collection_status.json"
RAW = DATA / "publication_collection_raw.json"

DEFAULT_TICKERS = ["AADI", "AALI", "ABBA"]

IDX_DISCLOSURE = (
    "https://www.idx.co.id/id/perusahaan-tercatat/"
    "keterbukaan-informasi/"
)

ANNOUNCE_ENDPOINT = (
    "https://www.idx.co.id/primary/NewsAnnouncement/"
    "GetAnnouncement?kodeEmiten={ticker}&lang=id"
)

FIN_REPORT_ENDPOINT = (
    "https://www.idx.co.id/primary/ListedCompany/"
    "GetFinancialReport?"
    "periode={period}&year={year}&indexFrom=0&pageSize=1000"
    "&reportType=rdf&kodeEmiten={ticker}"
)

PERIOD_END_MAP = {
    "TW1": "03-31",
    "TW2": "06-30",
    "TW3": "09-30",
    "AUDIT": "12-31",
}

PERIOD_TERMS = {
    "TW1": [
        "tw1",
        "tw 1",
        "tw i",
        "triwulan 1",
        "triwulan i",
        "triwulan pertama",
        "quarter 1",
        "quarter i",
        "first quarter",
        "q1",
        "31 maret",
        "31 march",
    ],
    "TW2": [
        "tw2",
        "tw 2",
        "tw ii",
        "triwulan 2",
        "triwulan ii",
        "triwulan kedua",
        "quarter 2",
        "quarter ii",
        "second quarter",
        "q2",
        "30 juni",
        "30 june",
        "semester 1",
        "semester i",
    ],
    "TW3": [
        "tw3",
        "tw 3",
        "tw iii",
        "triwulan 3",
        "triwulan iii",
        "triwulan ketiga",
        "quarter 3",
        "quarter iii",
        "third quarter",
        "q3",
        "30 september",
    ],
    "AUDIT": [
        "audit",
        "audited",
        "tahunan",
        "annual",
        "31 desember",
        "31 december",
    ],
}

FINANCIAL_TERMS = [
    "laporan keuangan",
    "financial report",
    "financial statement",
    "penyampaian laporan keuangan",
    "laporan keuangan interim",
    "laporan keuangan tahunan",
]

MONTHS = {
    "jan": 1,
    "januari": 1,
    "january": 1,
    "feb": 2,
    "februari": 2,
    "february": 2,
    "mar": 3,
    "maret": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "mei": 5,
    "may": 5,
    "jun": 6,
    "juni": 6,
    "june": 6,
    "jul": 7,
    "juli": 7,
    "july": 7,
    "agu": 8,
    "agustus": 8,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "september": 9,
    "okt": 10,
    "oktober": 10,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "des": 12,
    "desember": 12,
    "dec": 12,
    "december": 12,
}


def clean_text(value):
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def load_tickers():
    if TICKER_FILE.exists():
        vals = [
            x.strip().upper()
            for x in TICKER_FILE.read_text(
                encoding="utf-8"
            ).splitlines()
            if x.strip()
        ]

        if vals:
            return sorted(set(vals))

    return DEFAULT_TICKERS


def normalize_period(value):
    p = clean_text(value).upper()

    aliases = {
        "TW1": "TW1",
        "Q1": "TW1",
        "TW2": "TW2",
        "Q2": "TW2",
        "TW3": "TW3",
        "Q3": "TW3",
        "AUDIT": "AUDIT",
        "TAHUNAN": "AUDIT",
        "ANNUAL": "AUDIT",
        "FY": "AUDIT",
    }

    return aliases.get(p)


def period_end(year, period):
    suffix = PERIOD_END_MAP.get(period)

    if not suffix:
        return ""

    return f"{int(year):04d}-{suffix}"


def record_text(obj):
    parts = []

    def walk(value):
        if isinstance(value, dict):
            for v in value.values():
                walk(v)

        elif isinstance(value, list):
            for v in value:
                walk(v)

        elif isinstance(value, (str, int, float)):
            parts.append(str(value))

    walk(obj)

    return clean_text(" ".join(parts))


def parse_timestamp(value):
    """
    Return timezone-aware WIB timestamp text.

    No date-only fallback is permitted here.
    A real clock time must be present.
    """

    if not isinstance(value, str):
        return None

    s = clean_text(value)

    # yyyy-mm-dd HH:MM[:SS]
    m = re.search(
        r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})"
        r"[ T](\d{1,2}):(\d{2})(?::(\d{2}))?",
        s,
        re.I,
    )

    if m:
        y, mo, d, hh, mi, ss = m.groups()

        return (
            f"{int(y):04d}-{int(mo):02d}-{int(d):02d} "
            f"{int(hh):02d}:{int(mi):02d}:"
            f"{int(ss or 0):02d}+07:00"
        )

    # dd Month yyyy HH:MM[:SS]
    m = re.search(
        r"(\d{1,2})[\s/-]+"
        r"([A-Za-z]+)[\s/-]+"
        r"(\d{4})"
        r"(?:\s+|,\s*)"
        r"(\d{1,2}):(\d{2})(?::(\d{2}))?",
        s,
        re.I,
    )

    if m:
        d, mon, y, hh, mi, ss = m.groups()

        mo = MONTHS.get(mon.lower())

        if not mo:
            return None

        return (
            f"{int(y):04d}-{mo:02d}-{int(d):02d} "
            f"{int(hh):02d}:{int(mi):02d}:"
            f"{int(ss or 0):02d}+07:00"
        )

    return None


def timestamps_in_record(record):
    found = []

    def walk(value, path=""):
        if isinstance(value, dict):
            for k, v in value.items():
                child = f"{path}.{k}" if path else str(k)
                walk(v, child)

        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, f"{path}[{i}]")

        elif isinstance(value, str):
            ts = parse_timestamp(value)

            if ts:
                found.append(
                    {
                        "path": path,
                        "raw": value,
                        "timestamp": ts,
                    }
                )

    walk(record)

    # Deterministic de-duplication.
    unique = {}

    for item in found:
        key = (
            item["path"],
            item["timestamp"],
        )
        unique[key] = item

    return list(unique.values())


def has_financial_context(text):
    lower = text.lower()

    return any(
        term in lower
        for term in FINANCIAL_TERMS
    )


def has_exact_period_context(text, period, year):
    """
    Period reconciliation must be explicit.

    Year alone is NOT sufficient.
    """

    lower = text.lower()

    if str(year) not in lower:
        return False

    terms = PERIOD_TERMS.get(period, [])

    return any(
        term in lower
        for term in terms
    )


def ticker_matches(text, ticker):
    return ticker.lower() in text.lower()


def iter_dict_records(obj):
    """
    Yield every dict in the payload.

    Candidate acceptance remains fail-closed later.
    """

    if isinstance(obj, dict):
        yield obj

        for value in obj.values():
            yield from iter_dict_records(value)

    elif isinstance(obj, list):
        for value in obj:
            yield from iter_dict_records(value)


def find_exact_candidates(
    obj,
    ticker,
    year,
    period,
    source_kind,
):
    candidates = []

    pe = period_end(year, period)

    for record in iter_dict_records(obj):
        text = record_text(record)

        if not text:
            continue

        if not ticker_matches(text, ticker):
            continue

        if not has_financial_context(text):
            continue

        if not has_exact_period_context(
            text,
            period,
            year,
        ):
            continue

        timestamps = timestamps_in_record(record)

        # Fail closed:
        # exactly one timestamp must be attributable
        # to this candidate record.
        if len(timestamps) != 1:
            continue

        ts = timestamps[0]["timestamp"]

        candidates.append(
            {
                "ticker": ticker,
                "period_end": pe,
                "period_code": period,
                "year": str(year),
                "publication_timestamp": ts,
                "publication_timezone": "Asia/Jakarta",
                "source": "IDX",
                "source_kind": source_kind,
                "source_url": IDX_DISCLOSURE,
                "evidence_type": (
                    "IDX automatic exact-period candidate"
                ),
                "confidence": "REVIEW",
                "verification_status": "CANDIDATE",
                "notes": (
                    "Automatically discovered from IDX. "
                    "Ticker, reporting year and reporting "
                    "period matched explicitly. Candidate "
                    "must pass automatic reconciliation "
                    "before promotion to canonical evidence."
                ),
            }
        )

    # Remove identical duplicates.
    unique = {}

    for row in candidates:
        key = (
            row["ticker"],
            row["period_end"],
            row["publication_timestamp"],
        )
        unique[key] = row

    return list(unique.values())


def browser_fetch(urls):
    """
    Open the real IDX origin first, then fetch endpoints
    from the browser session.

    No timestamp is generated from request time.
    """

    from playwright.sync_api import sync_playwright

    result = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features="
                "AutomationControlled"
            ],
        )

        context = browser.new_context(
            locale="id-ID",
            timezone_id="Asia/Jakarta",
        )

        page = context.new_page()

        page.goto(
            IDX_DISCLOSURE,
            wait_until="domcontentloaded",
            timeout=60000,
        )

        page.wait_for_timeout(4000)

        for key, url in urls.items():
            try:
                data = page.evaluate(
                    """
                    async (url) => {
                        const r = await fetch(
                            url,
                            {credentials: 'include'}
                        );
                        return {
                            status: r.status,
                            text: await r.text()
                        };
                    }
                    """,
                    url,
                )

                result[key] = data

            except Exception as exc:
                result[key] = {
                    "status": 0,
                    "text": "",
                    "error": str(exc),
                }

        browser.close()

    return result


def main():
    DATA.mkdir(
        parents=True,
        exist_ok=True,
    )

    tickers = load_tickers()

    year = int(
        os.getenv(
            "IDX_PUBLICATION_YEAR",
            "2026",
        )
    )

    period = normalize_period(
        os.getenv(
            "IDX_PUBLICATION_PERIOD",
            "TW1",
        )
    )

    if not period:
        status = {
            "status": "BLOCKED",
            "engine_changed": False,
            "reason": "INVALID_PERIOD",
            "allowed_periods": [
                "TW1",
                "TW2",
                "TW3",
                "AUDIT",
            ],
        }

        STATUS.write_text(
            json.dumps(
                status,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(
            json.dumps(
                status,
                indent=2,
                ensure_ascii=False,
            )
        )

        raise SystemExit(1)

    urls = {}

    for ticker in tickers:
        urls[f"ann_{ticker}"] = (
            ANNOUNCE_ENDPOINT.format(
                ticker=quote(ticker)
            )
        )

        urls[f"fin_{ticker}"] = (
            FIN_REPORT_ENDPOINT.format(
                ticker=quote(ticker),
                period=period,
                year=year,
            )
        )

    try:
        fetched = browser_fetch(urls)

    except Exception as exc:
        status = {
            "status": "BROWSER_COLLECTOR_UNAVAILABLE",
            "engine_changed": False,
            "tickers": tickers,
            "error": str(exc),
        }

        STATUS.write_text(
            json.dumps(
                status,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(
            json.dumps(
                status,
                indent=2,
                ensure_ascii=False,
            )
        )

        raise SystemExit(1)

    raw_payload = {}
    errors = []
    discovered = []

    for key, response in fetched.items():
        text = response.get("text", "")

        raw_payload[key] = {
            "status": response.get("status"),
            "bytes": len(text),
            "error": response.get("error"),
            "body": text,
        }

        if response.get("status") != 200:
            errors.append(
                {
                    "request": key,
                    "status": response.get(
                        "status"
                    ),
                    "error": response.get(
                        "error"
                    ),
                }
            )
            continue

        try:
            obj = json.loads(text)

        except Exception as exc:
            errors.append(
                {
                    "request": key,
                    "status": response.get(
                        "status"
                    ),
                    "error": (
                        "NON_JSON_RESPONSE: "
                        + str(exc)
                    ),
                }
            )
            continue

        if key.startswith("ann_"):
            ticker = key[4:]

            discovered.extend(
                find_exact_candidates(
                    obj=obj,
                    ticker=ticker,
                    year=year,
                    period=period,
                    source_kind=(
                        "IDX_ANNOUNCEMENT_ENDPOINT"
                    ),
                )
            )

        elif key.startswith("fin_"):
            ticker = key[4:]

            discovered.extend(
                find_exact_candidates(
                    obj=obj,
                    ticker=ticker,
                    year=year,
                    period=period,
                    source_kind=(
                        "IDX_FINANCIAL_REPORT_ENDPOINT"
                    ),
                )
            )

    # ---------------------------------------------------------
    # Cross-endpoint reconciliation
    #
    # Automatic evidence is promotable only when the same:
    # ticker + period_end + timestamp
    # is independently present in >= 2 source kinds.
    #
    # Otherwise it remains REVIEW and MUST NOT become canonical.
    # ---------------------------------------------------------

    grouped = {}

    for row in discovered:
        key = (
            row["ticker"],
            row["period_end"],
            row["publication_timestamp"],
        )

        grouped.setdefault(
            key,
            [],
        ).append(row)

    final_candidates = []

    for key, group in grouped.items():
        source_kinds = sorted(
            set(
                row["source_kind"]
                for row in group
            )
        )

        base = dict(group[0])

        base["source_kind"] = "|".join(
            source_kinds
        )

        base["independent_source_count"] = len(
            source_kinds
        )

        if len(source_kinds) >= 2:
            base["confidence"] = "HIGH"
            base["verification_status"] = (
                "AUTO_VERIFIED"
            )
            base["notes"] = (
                "Automatically verified by exact "
                "ticker + period_end + publication "
                "timestamp agreement across multiple "
                "IDX endpoint families."
            )
        else:
            base["confidence"] = "REVIEW"
            base["verification_status"] = (
                "REVIEW"
            )
            base["notes"] = (
                "Exact ticker and reporting period "
                "were detected, but independent IDX "
                "endpoint confirmation is insufficient. "
                "Not canonical PIT evidence."
            )

        final_candidates.append(base)

    fields = [
        "ticker",
        "period_end",
        "period_code",
        "year",
        "publication_timestamp",
        "publication_timezone",
        "source",
        "source_kind",
        "source_url",
        "evidence_type",
        "confidence",
        "verification_status",
        "independent_source_count",
        "notes",
    ]

    with AUTO_CANDIDATES.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            sorted(
                final_candidates,
                key=lambda x: (
                    x.get("ticker", ""),
                    x.get("period_end", ""),
                    x.get(
                        "publication_timestamp",
                        "",
                    ),
                ),
            )
        )

    RAW.write_text(
        json.dumps(
            raw_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    successful_requests = sum(
        1
        for response in fetched.values()
        if response.get("status") == 200
    )

    if successful_requests == 0:
        status = {
            "status": "COLLECTION_BLOCKED",
            "engine_changed": False,
            "canonical_evidence_modified": False,
            "year": year,
            "period": period,
            "period_end": period_end(
                year,
                period,
            ),
            "tickers_requested": len(tickers),
            "tickers": tickers,
            "successful_requests": 0,
            "http_errors": len(errors),
            "errors": errors[:20],
            "reason": (
                "No IDX request returned HTTP 200. "
                "Automatic publication evidence "
                "collection is unavailable."
            ),
            "notes": [
                (
                    "Automatic evidence was not "
                    "promoted to canonical evidence."
                ),
                (
                    "Canonical publication_evidence.csv "
                    "remains unchanged."
                ),
                (
                    "Fail-closed: collection failure "
                    "must not be reported as PIT PASS."
                ),
            ],
        }

        STATUS.write_text(
            json.dumps(
                status,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(
            json.dumps(
                status,
                indent=2,
                ensure_ascii=False,
            )
        )

        raise SystemExit(1)
    auto_verified = sum(
        row.get("verification_status")
        == "AUTO_VERIFIED"
        for row in final_candidates
    )

    review = sum(
        row.get("verification_status")
        == "REVIEW"
        for row in final_candidates
    )

    status = {
        "status": (
            "COLLECTION_COMPLETE"
            if not errors
            else "COLLECTION_PARTIAL"
        ),
        "engine_changed": False,
        "canonical_evidence_modified": False,
        "year": year,
        "period": period,
        "period_end": period_end(
            year,
            period,
        ),
        "tickers_requested": len(tickers),
        "tickers": tickers,
        "raw_candidates": len(discovered),
        "reconciled_candidates": len(
            final_candidates
        ),
        "auto_verified_candidates": (
            auto_verified
        ),
        "review_candidates": review,
        "http_errors": len(errors),
        "errors": errors[:20],
        "notes": [
            (
                "Automatic collector does not modify "
                "publication_evidence.csv."
            ),
            (
                "No publication timestamp is inferred "
                "from period_end."
            ),
            (
                "Date-only evidence is not accepted by "
                "this automatic collector."
            ),
            (
                "Ticker, reporting year and reporting "
                "period must be explicit."
            ),
            (
                "Automatic evidence remains separate "
                "until reconciliation validates it."
            ),
            (
                "Manual HIGH evidence remains canonical "
                "during migration."
            ),
        ],
    }

    STATUS.write_text(
        json.dumps(
            status,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            status,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()