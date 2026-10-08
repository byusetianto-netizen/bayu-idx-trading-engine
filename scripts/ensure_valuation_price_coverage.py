#!/usr/bin/env python3
"""Ensure V2.2 has point-in-time prices for PIT-eligible financial tickers.

This does NOT rebuild the main IDX price dataset. It only creates a small
valuation supplement for PIT tickers that are missing from the main dataset
and from the optional V1.3B expansion dataset.
"""
from pathlib import Path
import json
from datetime import timedelta
import pandas as pd
import numpy as np

ROOT = Path('.')
PRICE = ROOT / 'data' / 'idx_stock_prices.csv'
EXPANSION = ROOT / 'data' / 'idx_stock_prices_expansion.csv'
FS_PIT = ROOT / 'data' / 'fundamental' / 'financial_statements_pit.csv'
OUTDIR = ROOT / 'data' / 'valuation'
SUPPLEMENT = OUTDIR / 'valuation_price_supplement.csv'
STATUS = OUTDIR / 'valuation_price_coverage_status.json'
LOOKBACK_DAYS = 3650


def read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def norm(df):
    if df.empty:
        return df
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if 'ticker' in df.columns:
        df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None)
    return df


def available_tickers(*dfs):
    out = set()
    for df in dfs:
        if not df.empty and 'ticker' in df.columns:
            out |= set(df['ticker'].dropna().astype(str).str.upper().str.strip())
    return out


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    prices = norm(read_csv(PRICE))
    expansion = norm(read_csv(EXPANSION))
    fs = norm(read_csv(FS_PIT))

    if prices.empty or not {'ticker', 'date', 'close'}.issubset(prices.columns):
        status = {'status': 'BLOCKED', 'reason': 'Main price dataset unavailable'}
        STATUS.write_text(json.dumps(status, indent=2), encoding='utf-8')
        print(json.dumps(status, indent=2))
        return

    prices = prices.dropna(subset=['ticker', 'date', 'close'])
    analysis_date = prices['date'].max()

    if fs.empty or not {'ticker', 'publication_date'}.issubset(fs.columns):
        status = {'status': 'NO_PIT_SOURCE', 'analysis_date': analysis_date.date().isoformat(),
                  'pit_tickers_required': [], 'supplement_rows': 0}
        STATUS.write_text(json.dumps(status, indent=2), encoding='utf-8')
        print(json.dumps(status, indent=2))
        return

    fs['publication_date'] = pd.to_datetime(fs['publication_date'], errors='coerce').dt.tz_localize(None)
    if 'pit_ready' in fs.columns:
        ready = fs['pit_ready'].astype(str).str.lower().isin({'true', '1', 'yes', 'y'})
    else:
        ready = pd.Series(True, index=fs.index)
    eligible = fs[ready & fs['publication_date'].notna() & (fs['publication_date'] <= analysis_date)].copy()
    pit_tickers = sorted(set(eligible['ticker'].astype(str).str.upper().str.strip()))

    main_tickers = available_tickers(prices)
    expansion_tickers = available_tickers(expansion)
    existing = main_tickers | expansion_tickers
    missing = [t for t in pit_tickers if t not in existing]

    existing_supp = norm(read_csv(SUPPLEMENT))
    existing_supp = existing_supp[existing_supp['ticker'].isin(pit_tickers)] if not existing_supp.empty else existing_supp
    supp_tickers = available_tickers(existing_supp)
    missing_after_supp = [t for t in missing if t not in supp_tickers]

    rows = []
    errors = []
    fetched = []

    if missing_after_supp:
        try:
            import yfinance as yf
        except Exception as exc:
            errors.append({'error': 'yfinance_unavailable', 'detail': str(exc)})
            missing_after_supp = []
        else:
            start = (analysis_date - pd.Timedelta(days=LOOKBACK_DAYS)).date().isoformat()
            end = (analysis_date + pd.Timedelta(days=1)).date().isoformat()
            for ticker in missing_after_supp:
                symbol = ticker if ticker.endswith('.JK') else f'{ticker}.JK'
                try:
                    h = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=False, actions=False)
                    if h is None or h.empty:
                        errors.append({'ticker': ticker, 'error': 'no_history_returned'})
                        continue
                    h = h.reset_index()
                    date_col = 'Date' if 'Date' in h.columns else 'date'
                    close_col = 'Close' if 'Close' in h.columns else 'close'
                    if date_col not in h.columns or close_col not in h.columns:
                        errors.append({'ticker': ticker, 'error': 'missing_date_or_close'})
                        continue
                    h['date'] = pd.to_datetime(h[date_col], errors='coerce').dt.tz_localize(None)
                    h['close'] = pd.to_numeric(h[close_col], errors='coerce')
                    h = h[(h['date'].notna()) & (h['close'].notna()) & (h['date'] <= analysis_date)]
                    if h.empty:
                        errors.append({'ticker': ticker, 'error': 'no_valid_rows_on_or_before_analysis_date'})
                        continue
                    h['ticker'] = ticker
                    cols = ['ticker', 'date', 'close']
                    for src, dst in [('Open','open'), ('High','high'), ('Low','low'), ('Volume','volume'), ('Adj Close','adj_close')]:
                        if src in h.columns:
                            h[dst] = pd.to_numeric(h[src], errors='coerce')
                            cols.append(dst)
                    h = h[cols].drop_duplicates(['ticker', 'date'])
                    rows.append(h)
                    fetched.append({'ticker': ticker, 'rows': int(len(h)), 'last_date': h['date'].max().date().isoformat()})
                except Exception as exc:
                    errors.append({'ticker': ticker, 'error': type(exc).__name__, 'detail': str(exc)})

    new_supp = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not existing_supp.empty and not new_supp.empty:
        supplement = pd.concat([existing_supp, new_supp], ignore_index=True)
    elif not existing_supp.empty:
        supplement = existing_supp.copy()
    else:
        supplement = new_supp.copy()

    if not supplement.empty:
        supplement = norm(supplement)
        supplement = supplement.drop_duplicates(['ticker', 'date'], keep='last').sort_values(['ticker', 'date'])
        supplement.to_csv(SUPPLEMENT, index=False)
    elif SUPPLEMENT.exists():
        # Keep an existing valid supplement untouched.
        supplement = norm(read_csv(SUPPLEMENT))

    covered = main_tickers | expansion_tickers | available_tickers(supplement)
    still_missing = [t for t in pit_tickers if t not in covered]
    status = {
        'status': 'COVERAGE_COMPLETE' if not still_missing else 'COVERAGE_PARTIAL',
        'analysis_date': analysis_date.date().isoformat(),
        'pit_tickers_required': pit_tickers,
        'main_price_tickers': len(main_tickers),
        'expansion_price_tickers': len(expansion_tickers),
        'supplement_price_tickers': len(available_tickers(supplement)),
        'missing_before_supplement': missing,
        'fetched': fetched,
        'still_missing': still_missing,
        'errors': errors,
        'supplement_rows': int(len(supplement)),
        'notes': [
            'Main IDX price dataset is never overwritten by this script.',
            'Only PIT-required tickers missing from the main/expansion price universe are supplemented.',
            'All fetched prices are restricted to dates on/before the analysis date.',
            'If Yahoo Finance is unavailable, the engine reports the missing coverage instead of inventing prices.'
        ]
    }
    STATUS.write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
