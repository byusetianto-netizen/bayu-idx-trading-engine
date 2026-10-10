import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fundamental"

EVIDENCE = DATA / "publication_evidence.csv"
AUTO_CANDIDATES = DATA / "publication_auto_candidates.csv"
FIN = DATA / "financial_statements.csv"
OUT = DATA / "pit_publication_validation.json"

WIB_OFFSET = "+07:00"


def read_rows(path):
    if not path.exists():
        return []

    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def clean(value):
    return (value or "").strip()


def normalize_date(value):
    value = clean(value)

    if not value:
        return ""

    return value[:10]


def parse_timestamp(value):
    """
    Strict PIT timestamp parser.

    Requirements:
    - exact date + clock time
    - explicit timezone offset
    - no publication_date fallback
    - no period_end inference
    """

    value = clean(value)

    if not value:
        return None

    candidate = value.replace(" ", "T", 1)

    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        return None

    if dt.tzinfo is None:
        return None

    return dt


def timestamp_text(dt):
    if dt is None:
        return ""

    return dt.isoformat(
        sep=" ",
        timespec="seconds",
    )


def canonical_timestamp(row):
    """
    Canonical evidence must expose publication_timestamp.

    publication_date is informational only and MUST NOT
    be used as PIT fallback.
    """

    return parse_timestamp(
        row.get("publication_timestamp")
    )


def candidate_timestamp(row):
    return parse_timestamp(
        row.get("publication_timestamp")
    )


def canonical_status(row):
    """
    Support the canonical V2.1G schema.

    A row is PIT-ready only when:
    - timestamp is valid and timezone-aware
    - source status is VERIFIED
    - pit_rule_ready is true
    """

    source_status = clean(
        row.get("source_status")
    ).upper()

    pit_ready = clean(
        row.get("pit_rule_ready")
    ).lower()

    confidence = clean(
        row.get("publication_confidence")
    ).upper()

    return {
        "source_status": source_status,
        "pit_rule_ready": pit_ready
        in ("true", "1", "yes"),
        "confidence": confidence,
    }


def candidate_status(row):
    return {
        "verification_status": clean(
            row.get("verification_status")
        ).upper(),
        "confidence": clean(
            row.get("confidence")
        ).upper(),
    }


def validate_canonical(rows):
    valid = []
    invalid = []

    seen = {}

    for index, row in enumerate(
        rows,
        start=2,
    ):
        ticker = clean(
            row.get("ticker")
        ).upper()

        period_end = normalize_date(
            row.get("period_end")
        )

        ts = canonical_timestamp(row)
        state = canonical_status(row)

        reasons = []

        if not ticker:
            reasons.append("MISSING_TICKER")

        if not period_end:
            reasons.append(
                "MISSING_PERIOD_END"
            )

        if ts is None:
            reasons.append(
                "MISSING_OR_INVALID_PUBLICATION_TIMESTAMP"
            )

        if state["source_status"] != "VERIFIED":
            reasons.append(
                "SOURCE_NOT_VERIFIED"
            )

        if not state["pit_rule_ready"]:
            reasons.append(
                "PIT_RULE_NOT_READY"
            )

        if (
            ts is not None
            and period_end
        ):
            try:
                pe = datetime.fromisoformat(
                    period_end
                ).replace(
                    tzinfo=ts.tzinfo
                )

                if ts <= pe:
                    reasons.append(
                        "PUBLICATION_NOT_AFTER_PERIOD_END"
                    )

            except ValueError:
                reasons.append(
                    "INVALID_PERIOD_END"
                )

        key = (
            ticker,
            period_end,
        )

        if (
            ticker
            and period_end
        ):
            if key in seen:
                reasons.append(
                    "DUPLICATE_TICKER_PERIOD"
                )
            else:
                seen[key] = index

        item = {
            "row": index,
            "ticker": ticker,
            "period_end": period_end,
            "publication_timestamp": (
                timestamp_text(ts)
            ),
            "source_status": (
                state["source_status"]
            ),
            "pit_rule_ready": (
                state["pit_rule_ready"]
            ),
            "publication_confidence": (
                state["confidence"]
            ),
            "reasons": reasons,
        }

        if reasons:
            invalid.append(item)
        else:
            valid.append(item)

    return valid, invalid


def validate_candidates(rows):
    auto_verified = []
    review = []
    invalid = []

    seen = set()

    for index, row in enumerate(
        rows,
        start=2,
    ):
        ticker = clean(
            row.get("ticker")
        ).upper()

        period_end = normalize_date(
            row.get("period_end")
        )

        ts = candidate_timestamp(row)
        state = candidate_status(row)

        try:
            source_count = int(
                clean(
                    row.get(
                        "independent_source_count"
                    )
                )
                or "0"
            )
        except ValueError:
            source_count = 0

        reasons = []

        if not ticker:
            reasons.append(
                "MISSING_TICKER"
            )

        if not period_end:
            reasons.append(
                "MISSING_PERIOD_END"
            )

        if ts is None:
            reasons.append(
                "MISSING_OR_INVALID_PUBLICATION_TIMESTAMP"
            )

        if (
            state["verification_status"]
            == "AUTO_VERIFIED"
            and state["confidence"]
            != "HIGH"
        ):
            reasons.append(
                "AUTO_VERIFIED_WITHOUT_HIGH_CONFIDENCE"
            )

        if (
            state["verification_status"]
            == "AUTO_VERIFIED"
            and source_count < 2
        ):
            reasons.append(
                "AUTO_VERIFIED_WITH_INSUFFICIENT_SOURCE_COUNT"
            )

        if (
            state["verification_status"]
            not in (
                "AUTO_VERIFIED",
                "REVIEW",
                "CANDIDATE",
            )
        ):
            reasons.append(
                "INVALID_VERIFICATION_STATUS"
            )

        if (
            ts is not None
            and period_end
        ):
            try:
                pe = datetime.fromisoformat(
                    period_end
                ).replace(
                    tzinfo=ts.tzinfo
                )

                if ts <= pe:
                    reasons.append(
                        "PUBLICATION_NOT_AFTER_PERIOD_END"
                    )

            except ValueError:
                reasons.append(
                    "INVALID_PERIOD_END"
                )

        key = (
            ticker,
            period_end,
            timestamp_text(ts),
        )

        if key in seen:
            reasons.append(
                "DUPLICATE_CANDIDATE"
            )
        else:
            seen.add(key)

        item = {
            "row": index,
            "ticker": ticker,
            "period_end": period_end,
            "publication_timestamp": (
                timestamp_text(ts)
            ),
            "verification_status": (
                state["verification_status"]
            ),
            "confidence": (
                state["confidence"]
            ),
            "independent_source_count": (
                source_count
            ),
            "reasons": reasons,
        }

        if reasons:
            invalid.append(item)

        elif (
            state["verification_status"]
            == "AUTO_VERIFIED"
        ):
            auto_verified.append(item)

        else:
            review.append(item)

    return (
        auto_verified,
        review,
        invalid,
    )


def financial_keys(rows):
    keys = set()

    for row in rows:
        ticker = clean(
            row.get("ticker")
            or row.get("KodeEmiten")
        ).upper()

        period_end = normalize_date(
            row.get("period_end")
            or row.get("period_end_date")
        )

        if ticker and period_end:
            keys.add(
                (
                    ticker,
                    period_end,
                )
            )

    return keys


def main():
    canonical_rows = read_rows(
        EVIDENCE
    )

    candidate_rows = read_rows(
        AUTO_CANDIDATES
    )

    financial_rows = read_rows(
        FIN
    )

    canonical_valid, canonical_invalid = (
        validate_canonical(
            canonical_rows
        )
    )

    (
        auto_verified,
        candidate_review,
        candidate_invalid,
    ) = validate_candidates(
        candidate_rows
    )

    fin_keys = financial_keys(
        financial_rows
    )

    canonical_keys = {
        (
            row["ticker"],
            row["period_end"],
        )
        for row in canonical_valid
    }

    auto_keys = {
        (
            row["ticker"],
            row["period_end"],
        )
        for row in auto_verified
    }

    canonical_matched = sorted(
        canonical_keys & fin_keys
    )

    auto_matched = sorted(
        auto_keys & fin_keys
    )

    # -----------------------------------------------------
    # Regression comparison
    #
    # Automatic evidence does NOT replace canonical
    # evidence here.
    #
    # We only compare timestamp equality for overlapping
    # ticker + period_end keys.
    # -----------------------------------------------------

    canonical_map = {
        (
            row["ticker"],
            row["period_end"],
        ): row["publication_timestamp"]
        for row in canonical_valid
    }

    auto_map = {
        (
            row["ticker"],
            row["period_end"],
        ): row["publication_timestamp"]
        for row in auto_verified
    }

    regression_matches = []
    regression_mismatches = []

    for key in sorted(
        set(canonical_map)
        & set(auto_map)
    ):
        item = {
            "ticker": key[0],
            "period_end": key[1],
            "canonical_timestamp": (
                canonical_map[key]
            ),
            "automatic_timestamp": (
                auto_map[key]
            ),
        }

        if (
            canonical_map[key]
            == auto_map[key]
        ):
            regression_matches.append(
                item
            )
        else:
            regression_mismatches.append(
                item
            )

    # Automatic collector remains migration-stage.
    # A mismatch is a hard block.
    # REVIEW candidates are allowed but never PIT-ready.

    hard_fail = bool(
        canonical_invalid
        or candidate_invalid
        or regression_mismatches
    )

    if hard_fail:
        status = "BLOCKED"
    elif canonical_valid:
        status = "PASS"
    else:
        status = "REVIEW"

    result = {
        "status": status,
        "engine_changed": False,
        "pit_contract": (
            "publication_timestamp "
            "<= analysis_timestamp"
        ),
        "publication_date_fallback": False,
        "period_end_used_as_publication_evidence": False,
        "automatic_candidate_promoted_to_canonical": False,
        "canonical_evidence": {
            "rows_loaded": len(
                canonical_rows
            ),
            "valid_rows": len(
                canonical_valid
            ),
            "invalid_rows": len(
                canonical_invalid
            ),
            "matched_to_financial_statements": len(
                canonical_matched
            ),
        },
        "automatic_candidates": {
            "rows_loaded": len(
                candidate_rows
            ),
            "auto_verified_rows": len(
                auto_verified
            ),
            "review_rows": len(
                candidate_review
            ),
            "invalid_rows": len(
                candidate_invalid
            ),
            "auto_verified_matched_to_financial_statements": len(
                auto_matched
            ),
        },
        "regression_against_canonical": {
            "overlapping_rows": (
                len(regression_matches)
                + len(regression_mismatches)
            ),
            "exact_timestamp_matches": len(
                regression_matches
            ),
            "timestamp_mismatches": len(
                regression_mismatches
            ),
            "matches": regression_matches,
            "mismatches": regression_mismatches,
        },
        "invalid_canonical_rows": (
            canonical_invalid
        ),
        "invalid_candidate_rows": (
            candidate_invalid
        ),
        "review_candidates": (
            candidate_review
        ),
        "auto_verified_candidates": (
            auto_verified
        ),
        "notes": [
            (
                "Canonical PIT evidence requires an "
                "explicit timezone-aware "
                "publication_timestamp."
            ),
            (
                "publication_date is never used as "
                "fallback publication evidence."
            ),
            (
                "period_end identifies the accounting "
                "period only; it is never publication "
                "evidence."
            ),
            (
                "Automatic candidates remain separate "
                "from canonical publication evidence."
            ),
            (
                "Automatic evidence must reproduce "
                "canonical timestamps exactly before "
                "migration can be trusted."
            ),
            (
                "REVIEW candidates are never treated "
                "as PIT-ready."
            ),
        ],
    }

    OUT.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    if status == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()