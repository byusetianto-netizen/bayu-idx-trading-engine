import argparse, json, math
from pathlib import Path
import pandas as pd
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data')
    p.add_argument('--min-rows-a', type=int, default=500)
    p.add_argument('--min-rows-b', type=int, default=200)
    p.add_argument('--max-lag-days-a', type=int, default=10)
    p.add_argument('--max-gap-days-a', type=int, default=45)
    return p.parse_args()


def norm_cols(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {
        'date':'date','price':'date','datetime':'date',
        'ticker':'ticker','symbol':'ticker','code':'ticker',
        'open':'open','high':'high','low':'low','close':'close','adj close':'adj_close',
        'volume':'volume','vol':'volume'
    }
    rename = {}
    for c in df.columns:
        if c in aliases and aliases[c] != c:
            rename[c] = aliases[c]
    return df.rename(columns=rename)


def load_price(path):
    df = norm_cols(pd.read_csv(path, low_memory=False))
    required = {'date','ticker','open','high','low','close','volume'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns in {path}: {sorted(missing)}')
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None)
    df['ticker'] = df['ticker'].astype(str).str.upper().str.replace(r'\\.JK$', '', regex=True).str.strip()
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def load_ihsg(path):
    df = norm_cols(pd.read_csv(path, low_memory=False))
    date_col = 'date' if 'date' in df.columns else None
    if date_col is None and 'price' in df.columns:
        date_col = 'price'
    if date_col is None:
        raise ValueError('IHSG date column not found')
    df['date'] = pd.to_datetime(df[date_col], errors='coerce').dt.tz_localize(None)
    return df[['date']].dropna().drop_duplicates().sort_values('date')


def max_calendar_gap(dates):
    if len(dates) < 2:
        return 0
    vals = pd.Series(pd.to_datetime(sorted(pd.unique(dates))))
    return int(vals.diff().dt.days.max() or 0)


def max_session_gap(dates, calendar):
    dates = pd.to_datetime(pd.Series(dates)).dropna().drop_duplicates().sort_values()
    if len(dates) < 2 or len(calendar) < 2:
        return 0
    cal = pd.DatetimeIndex(calendar)
    pos = cal.get_indexer(dates)
    pos = pos[pos >= 0]
    if len(pos) < 2:
        return 0
    return int(np.diff(pos).max() - 1)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    expansion = data_dir / 'idx_stock_prices_expansion.csv'
    ihsg = data_dir / 'idx_ihsg_index.csv'
    if not expansion.exists():
        raise FileNotFoundError(f'Not found: {expansion}')
    if not ihsg.exists():
        raise FileNotFoundError(f'Not found: {ihsg}')

    df = load_price(expansion)
    cal_df = load_ihsg(ihsg)
    calendar = pd.DatetimeIndex(cal_df['date'].drop_duplicates().sort_values())
    global_latest = calendar.max()

    raw_rows = len(df)
    exact_dupes = int(df.duplicated(['ticker','date'], keep=False).sum())
    dedup_rows = int(df.duplicated(['ticker','date'], keep='first').sum())

    # Row-level checks
    invalid_date = int(df['date'].isna().sum())
    invalid_ticker = int(df['ticker'].isna().sum() + (df['ticker'].str.len() == 0).sum())
    missing_ohlcv = int(df[['open','high','low','close','volume']].isna().sum().sum())
    nonpositive_price = int(((df[['open','high','low','close']] <= 0).any(axis=1)).sum())
    invalid_ohlc = int(((df['high'] < df['low']) | (df['open'] > df['high']) | (df['open'] < df['low']) | (df['close'] > df['high']) | (df['close'] < df['low'])).fillna(False).sum())
    nonpositive_volume = int((df['volume'].fillna(0) < 0).sum())

    records = []
    for ticker, g in df.groupby('ticker', sort=True):
        g = g.sort_values('date')
        dates = g['date'].dropna().drop_duplicates()
        first_date = dates.min() if len(dates) else pd.NaT
        last_date = dates.max() if len(dates) else pd.NaT
        rows = len(g)
        dup_rows = int(g.duplicated(['ticker','date'], keep=False).sum())
        duplicate_keys = int(g.duplicated(['ticker','date'], keep='first').sum())
        miss = int(g[['open','high','low','close','volume']].isna().sum().sum())
        bad_price = int(((g[['open','high','low','close']] <= 0).any(axis=1)).sum())
        bad_ohlc = int(((g['high'] < g['low']) | (g['open'] > g['high']) | (g['open'] < g['low']) | (g['close'] > g['high']) | (g['close'] < g['low'])).fillna(False).sum())
        bad_vol = int((g['volume'].fillna(0) < 0).sum())
        lag_days = int((global_latest - last_date).days) if pd.notna(last_date) else 99999
        if pd.notna(first_date) and pd.notna(last_date):
            expected = int(((calendar >= first_date) & (calendar <= last_date)).sum())
        else:
            expected = 0
        coverage = float(rows / expected) if expected else 0.0
        session_gap = max_session_gap(dates, calendar)
        cal_gap = max_calendar_gap(dates)

        critical = (rows < args.min_rows_b) or (lag_days > 30) or (bad_price > 0) or (bad_ohlc > 0) or (miss > 0) or (duplicate_keys > 0)
        a_ok = (rows >= args.min_rows_a and lag_days <= args.max_lag_days_a and miss == 0 and bad_price == 0 and bad_ohlc == 0 and duplicate_keys == 0 and session_gap <= args.max_gap_days_a)
        b_ok = (rows >= args.min_rows_b and lag_days <= 30 and miss == 0 and bad_price == 0 and bad_ohlc == 0 and duplicate_keys == 0)
        if a_ok:
            grade = 'A'
            verdict = 'RESEARCH_READY'
        elif b_ok:
            grade = 'B'
            verdict = 'RESEARCH_WITH_REVIEW'
        else:
            grade = 'C'
            verdict = 'DO_NOT_USE_YET'

        records.append({
            'ticker': ticker, 'rows': rows, 'first_date': first_date.date().isoformat() if pd.notna(first_date) else '',
            'last_date': last_date.date().isoformat() if pd.notna(last_date) else '', 'lag_days': lag_days,
            'expected_sessions_between_dates': expected, 'session_coverage': round(coverage, 6),
            'duplicate_rows': dup_rows, 'duplicate_keys': duplicate_keys, 'missing_ohlcv_cells': miss,
            'nonpositive_price_rows': bad_price, 'invalid_ohlc_rows': bad_ohlc, 'negative_volume_rows': bad_vol,
            'max_calendar_gap_days': cal_gap, 'max_missing_trading_sessions': session_gap,
            'quality_grade': grade, 'verdict': verdict
        })

    audit = pd.DataFrame(records).sort_values(['quality_grade','ticker'])
    out_audit = data_dir / 'price_data_quality_audit.csv'
    audit.to_csv(out_audit, index=False)

    grade_counts = audit['quality_grade'].value_counts().to_dict()
    verdict_counts = audit['verdict'].value_counts().to_dict()
    summary = {
        'version': 'V1.3B-2',
        'status': 'AUDIT_ONLY__MAIN_DATASET_UNCHANGED',
        'input_dataset': str(expansion),
        'ihsg_calendar': str(ihsg),
        'raw_rows': raw_rows,
        'unique_tickers': int(audit['ticker'].nunique()),
        'global_latest_ihsg_date': global_latest.date().isoformat(),
        'duplicate_rows_flagged': exact_dupes,
        'duplicate_rows_to_remove_if_deduplicating': dedup_rows,
        'row_level_invalid_date': invalid_date,
        'row_level_invalid_ticker': invalid_ticker,
        'row_level_missing_ohlcv_cells': missing_ohlcv,
        'row_level_nonpositive_price_rows': nonpositive_price,
        'row_level_invalid_ohlc_rows': invalid_ohlc,
        'row_level_negative_volume_rows': nonpositive_volume,
        'quality_grade_counts': {k:int(v) for k,v in grade_counts.items()},
        'verdict_counts': {k:int(v) for k,v in verdict_counts.items()},
        'research_ready_tickers': sorted(audit.loc[audit.quality_grade=='A','ticker'].tolist()),
        'research_review_tickers': sorted(audit.loc[audit.quality_grade=='B','ticker'].tolist()),
        'do_not_use_yet_tickers': sorted(audit.loc[audit.quality_grade=='C','ticker'].tolist()),
        'important': [
            'This audit does not change idx_stock_prices.csv.',
            'This audit does not solve point-in-time membership, delisting, suspension history, or survivorship bias.',
            'Grade A means price-data quality is suitable for research review, not that the ticker is historically tradable for the whole sample.',
            'Current-universe membership must not be treated as historical-universe membership.'
        ]
    }
    (data_dir / 'price_data_quality_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    status = {
        'version': 'V1.3B-2', 'status': summary['status'], 'tickers': int(audit['ticker'].nunique()),
        'grade_A': int(grade_counts.get('A',0)), 'grade_B': int(grade_counts.get('B',0)), 'grade_C': int(grade_counts.get('C',0)),
        'latest_date': global_latest.date().isoformat(), 'output_audit': str(out_audit),
        'output_summary': str(data_dir / 'price_data_quality_summary.json')
    }
    (data_dir / 'price_data_quality_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')

    print(f'Expansion rows: {raw_rows}')
    print(f'Unique tickers: {audit["ticker"].nunique()}')
    print(f'Grade A: {grade_counts.get("A",0)}')
    print(f'Grade B: {grade_counts.get("B",0)}')
    print(f'Grade C: {grade_counts.get("C",0)}')
    print(f'Latest IHSG date: {global_latest.date().isoformat()}')
    print('STATUS: AUDIT_ONLY__MAIN_DATASET_UNCHANGED')

if __name__ == '__main__':
    main()
