"""
V1.3F-5 Research Session Eligibility Layer

Purpose
-------
Build a row-level research-session eligibility layer from the existing
price datasets and V1.3F-4 placeholder audit.

Important
---------
- Does NOT modify raw price datasets.
- Does NOT modify actual_engine.py.
- PLACEHOLDER_CANDIDATE is a data-quality classification only.
- It is NOT proof that a date is an official IDX non-trading session.
- Eligibility is intended for research involving ordered trading-session
  calculations such as returns, rolling windows, and forward labels.
"""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

BASE_PRICE = DATA / "idx_stock_prices.csv"
EXPANSION_PRICE = DATA / "idx_stock_prices_expansion.csv"

PROBE_DIR = DATA / "execution_probe"
PLACEHOLDER_AUDIT = PROBE_DIR / "expansion_calendar_placeholder_audit.csv"

OUTPUT_DETAIL = PROBE_DIR / "research_session_eligibility.csv"
OUTPUT_SUMMARY = PROBE_DIR / "research_session_eligibility_summary.json"

KEY = ["date", "ticker"]

REQUIRED_PRICE_COLUMNS = {
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
}

REQUIRED_AUDIT_COLUMNS = {
    "date",
    "ticker",
    "placeholder_candidate",
    "classification",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def normalize_key(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype("string").str.strip().str.upper()

    if df["date"].isna().any():
        fail(f"{source_name}: invalid/missing date detected.")

    if df["ticker"].isna().any() or (df["ticker"] == "").any():
        fail(f"{source_name}: invalid/missing ticker detected.")

    return df


def parse_bool(series: pd.Series, source_name: str) -> pd.Series:
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    normalized = series.astype("string").str.strip().str.lower()
    parsed = normalized.map(mapping)

    if parsed.isna().any():
        bad_values = sorted(
            normalized.loc[parsed.isna()].dropna().unique().tolist()
        )
        fail(
            f"{source_name}: invalid placeholder_candidate values: "
            f"{bad_values[:10]}"
        )

    return parsed.astype(bool)


def assert_unique_key(df: pd.DataFrame, source_name: str) -> None:
    duplicated = df.duplicated(KEY, keep=False)

    if duplicated.any():
        examples = (
            df.loc[duplicated, KEY]
            .head(10)
            .astype(str)
            .to_dict("records")
        )
        fail(
            f"{source_name}: duplicate date+ticker keys detected. "
            f"Examples: {examples}"
        )


def load_price(path: Path, source_name: str) -> pd.DataFrame:
    if not path.exists():
        fail(f"{source_name}: file not found: {path}")

    df = pd.read_csv(path)

    missing = REQUIRED_PRICE_COLUMNS - set(df.columns)
    if missing:
        fail(
            f"{source_name}: missing required columns: "
            f"{sorted(missing)}"
        )

    df = normalize_key(df, source_name)
    assert_unique_key(df, source_name)

    return df


def load_placeholder_audit() -> pd.DataFrame:
    if not PLACEHOLDER_AUDIT.exists():
        fail(
            "V1.3F-4 placeholder audit is missing. "
            f"Expected: {PLACEHOLDER_AUDIT}"
        )

    audit = pd.read_csv(PLACEHOLDER_AUDIT)

    missing = REQUIRED_AUDIT_COLUMNS - set(audit.columns)
    if missing:
        fail(
            "V1.3F-4 placeholder audit missing required columns: "
            f"{sorted(missing)}"
        )

    audit = normalize_key(audit, "placeholder audit")
    assert_unique_key(audit, "placeholder audit")

    audit["placeholder_candidate"] = parse_bool(
        audit["placeholder_candidate"],
        "placeholder audit",
    )

    # V1.3F-4 output should contain only stock-only observations.
    # Keep only the fields needed by this eligibility layer.
    return audit[
        KEY + ["placeholder_candidate", "classification"]
    ].copy()


def build_layer(
    price: pd.DataFrame,
    audit: pd.DataFrame,
    universe_name: str,
) -> pd.DataFrame:

    result = price[KEY].copy()

    result = result.merge(
        audit,
        on=KEY,
        how="left",
        validate="one_to_one",
    )

    result["in_v13f4_audit"] = result["classification"].notna()

    # Rows absent from V1.3F-4 are NOT automatically proven valid sessions.
    # They simply have no placeholder evidence from that diagnostic.
    result["placeholder_candidate"] = (
        result["placeholder_candidate"]
        .fillna(False)
        .astype(bool)
    )

    result["research_session_eligible"] = (
        ~result["placeholder_candidate"]
    )

    result["eligibility_reason"] = "NO_V13F4_PLACEHOLDER_EVIDENCE"

    result.loc[
        result["placeholder_candidate"],
        "eligibility_reason",
    ] = "EXCLUDED_PLACEHOLDER_CANDIDATE"

    result["source_universe"] = universe_name

    return result[
        [
            "date",
            "ticker",
            "source_universe",
            "in_v13f4_audit",
            "placeholder_candidate",
            "research_session_eligible",
            "eligibility_reason",
            "classification",
        ]
    ]


def summarize_layer(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "unique_tickers": int(df["ticker"].nunique()),
        "unique_dates": int(df["date"].nunique()),
        "v13f4_audit_overlap_rows": int(
            df["in_v13f4_audit"].sum()
        ),
        "placeholder_candidate_rows": int(
            df["placeholder_candidate"].sum()
        ),
        "research_session_eligible_rows": int(
            df["research_session_eligible"].sum()
        ),
        "research_session_ineligible_rows": int(
            (~df["research_session_eligible"]).sum()
        ),
    }


def main() -> None:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)

    audit = load_placeholder_audit()

    base = load_price(
        BASE_PRICE,
        "base price dataset",
    )

    expansion = load_price(
        EXPANSION_PRICE,
        "expansion price dataset",
    )

    base_layer = build_layer(
        base,
        audit,
        "BASE_95",
    )

    expansion_layer = build_layer(
        expansion,
        audit,
        "EXPANSION_903",
    )

    detail = pd.concat(
        [base_layer, expansion_layer],
        ignore_index=True,
    )

    summary = {
        "version": "V1.3F-5",
        "status": (
            "RESEARCH_SESSION_ELIGIBILITY_COMPLETE"
            "__NOT_ENABLED_IN_ENGINE"
        ),
        "policy": {
            "raw_price_datasets_modified": False,
            "actual_engine_modified": False,
            "placeholder_is_proven_non_trading_session": False,
            "eligibility_scope": (
                "Research calculations based on ordered "
                "price observations."
            ),
            "exclusion_rule": (
                "Exclude rows explicitly classified by V1.3F-4 "
                "as placeholder_candidate=True."
            ),
            "absence_from_v13f4_means_proven_valid_session": False,
        },
        "v13f4": {
            "audit_rows": int(len(audit)),
            "placeholder_candidate_rows": int(
                audit["placeholder_candidate"].sum()
            ),
            "unique_dates": int(audit["date"].nunique()),
            "unique_tickers": int(audit["ticker"].nunique()),
        },
        "base": summarize_layer(base_layer),
        "expansion": summarize_layer(expansion_layer),
        "notes": [
            (
                "This layer does not establish an official historical "
                "IDX trading calendar."
            ),
            (
                "PLACEHOLDER_CANDIDATE remains a diagnostic "
                "data-quality classification."
            ),
            (
                "No production engine consumes this eligibility "
                "layer in V1.3F-5."
            ),
        ],
    }

    detail.to_csv(
        OUTPUT_DETAIL,
        index=False,
        date_format="%Y-%m-%d",
    )

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("V1.3F-5 Research Session Eligibility Layer")
    print("------------------------------------------")
    print(f"V1.3F-4 audit rows: {len(audit):,}")
    print(
        "V1.3F-4 placeholder candidates: "
        f"{audit['placeholder_candidate'].sum():,}"
    )
    print()
    print("BASE:")
    for key, value in summary["base"].items():
        print(f"  {key}: {value:,}")
    print()
    print("EXPANSION:")
    for key, value in summary["expansion"].items():
        print(f"  {key}: {value:,}")
    print()
    print(
        "STATUS: "
        "RESEARCH_SESSION_ELIGIBILITY_COMPLETE"
        "__NOT_ENABLED_IN_ENGINE"
    )


if __name__ == "__main__":
    main()