import os
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.indexalpha.id"
API_KEY = os.getenv("INDEXALPHA_API_KEY")
MAX_TICKERS_PER_DAY = int(os.getenv("MAX_TICKERS_PER_DAY", "5"))
BATCH_SIZE = min(int(os.getenv("BATCH_SIZE", "50")), 50)
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "1"))
INVESTOR = os.getenv("BROKER_INVESTOR", "all")
MARKET = os.getenv("BROKER_MARKET", "RG")

UNIVERSE_FILE = Path("data/research_universe.csv")
OUT_DIR = Path("data/broker")
OUT_FILE = OUT_DIR / "broker_summary_daily.csv"
STATUS_FILE = OUT_DIR / "v20_collection_status.json"

def fail(msg):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps({
        "status": "FAILED",
        "message": msg,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    raise RuntimeError(msg)

def load_tickers():
    if not UNIVERSE_FILE.exists():
        fail(f"Missing {UNIVERSE_FILE}")
    df = pd.read_csv(UNIVERSE_FILE)
    col = next((c for c in ["ticker", "code", "Ticker", "Code"] if c in df.columns), None)
    if not col:
        fail("research_universe.csv must contain ticker or code")
    tickers = (
        df[col].astype(str).str.upper().str.replace(".JK", "", regex=False)
        .str.strip().dropna().drop_duplicates().tolist()
    )
    return [x for x in tickers if x and x != "NAN"]

def trading_dates(df):
    if df.empty:
        return []
    if "date" not in df.columns:
        return []
    d = pd.to_datetime(df["date"], errors="coerce").dropna().dt.date
    return sorted(set(d))

def choose_tickers(tickers):
    # Deterministic rotating queue so free mode gradually covers the universe.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state_path = OUT_DIR / "ticker_queue_state.json"
    state = {"offset": 0}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except Exception:
            pass
    offset = int(state.get("offset", 0)) % max(len(tickers), 1)
    ordered = tickers[offset:] + tickers[:offset]
    chosen = ordered[:MAX_TICKERS_PER_DAY]
    state_path.write_text(json.dumps({
        "offset": (offset + len(chosen)) % max(len(tickers), 1),
        "last_selected": chosen,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    return chosen

def request_batch(tickers, date_str):
    if not API_KEY:
        fail("INDEXALPHA_API_KEY is not set")
    url = f"{BASE_URL}/stocks/broker-summary/batch"
    payload = {
        "tickers": tickers,
        "from": date_str,
        "to": date_str,
        "investor": INVESTOR,
        "market": MARKET,
    }
    r = requests.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"},
        timeout=45,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"Index Alpha HTTP {r.status_code}: {r.text[:500]}")
    body = r.json()
    if not body.get("success", False):
        raise RuntimeError(f"Index Alpha error: {body.get('error')}")
    return body.get("data", {}), r.headers

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = load_tickers()
    if not tickers:
        fail("No tickers in research universe")

    today = datetime.now(timezone.utc).astimezone().date()
    # The provider is updated around 19:00 WIB. For scheduled evening runs,
    # today is normally appropriate. If data is unavailable, the run fails safely.
    dates = [today - timedelta(days=i) for i in range(LOOKBACK_DAYS)]
    selected = choose_tickers(tickers)

    rows = []
    errors = []
    quota_remaining = None

    for d in dates:
        date_str = d.isoformat()
        for start in range(0, len(selected), BATCH_SIZE):
            batch = selected[start:start+BATCH_SIZE]
            try:
                data, headers = request_batch(batch, date_str)
                quota_remaining = headers.get("X-Monthly-Remaining", quota_remaining)
                for ticker, broker_rows in data.items():
                    for x in broker_rows or []:
                        rows.append({
                            "date": date_str,
                            "ticker": str(ticker).upper(),
                            "broker_code": x.get("code"),
                            "buy_freq": x.get("buy_freq", 0),
                            "buy_volume": x.get("buy_volume", 0),
                            "buy_value": x.get("buy_value", 0),
                            "sell_freq": x.get("sell_freq", 0),
                            "sell_volume": x.get("sell_volume", 0),
                            "sell_value": x.get("sell_value", 0),
                            "buy_avg": x.get("buy_avg"),
                            "sell_avg": x.get("sell_avg"),
                            "investor": INVESTOR,
                            "market": MARKET,
                            "source": "indexalpha",
                        })
            except Exception as e:
                errors.append({"date": date_str, "batch": batch, "error": str(e)})

    new_df = pd.DataFrame(rows)
    if OUT_FILE.exists():
        old = pd.read_csv(OUT_FILE)
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df

    if not combined.empty:
        combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")
        combined = combined.drop_duplicates(
            subset=["date", "ticker", "broker_code", "investor", "market"],
            keep="last"
        ).sort_values(["date", "ticker", "broker_code"])
        combined.to_csv(OUT_FILE, index=False)

    status = {
        "status": "SUCCESS" if len(errors) == 0 else "PARTIAL",
        "selected_tickers": selected,
        "selected_count": len(selected),
        "rows_received": len(new_df),
        "rows_total": len(combined),
        "errors": errors,
        "quota_remaining": quota_remaining,
        "provider": "indexalpha",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    STATUS_FILE.write_text(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
