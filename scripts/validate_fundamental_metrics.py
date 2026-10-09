#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import pandas as pd

EXPECTED = {
    'revenue_growth','gross_margin','net_margin','roa_period','roe_period',
    'debt_to_equity','current_ratio','cfo','free_cash_flow','cfo_to_net_income'
}

def read_csv_safe(path, required_columns):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=required_columns), f'missing file: {p}'
    if p.stat().st_size == 0:
        return pd.DataFrame(columns=required_columns), f'empty file: {p}'
    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=required_columns), f'empty CSV: {p}'
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        return df, f'missing columns in {p}: {missing}'
    return df, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics', default='data/fundamental/fundamental_metrics.csv')
    ap.add_argument('--assessment', default='data/fundamental/fundamental_assessment.csv')
    ap.add_argument('--analysis-date', default='2026-09-01')
    ap.add_argument('--output', default='data/fundamental/fundamental_metrics_validation.json')
    args = ap.parse_args()

    metric_cols = ['ticker','metric','period_end','source_document_id']
    assessment_cols = ['ticker','publication_date']
    m, metric_error = read_csv_safe(args.metrics, metric_cols)
    a, assessment_error = read_csv_safe(args.assessment, assessment_cols)

    try:
        ad = pd.Timestamp(args.analysis_date)
    except Exception as exc:
        raise SystemExit(f'Invalid --analysis-date {args.analysis_date!r}: {exc}')

    diagnostics = [x for x in (metric_error, assessment_error) if x]
    bad = 0
    tickers = []
    present = set()

    if not metric_error:
        m['period_end'] = pd.to_datetime(m['period_end'], errors='coerce')
        m['source_document_id'] = m['source_document_id'].astype(str)
        present = set(m['metric'].dropna().astype(str))

    if not assessment_error:
        a['publication_date'] = pd.to_datetime(a['publication_date'], errors='coerce')
        bad = int((a['publication_date'] > ad).sum())
        tickers = sorted(a['ticker'].dropna().astype(str).unique().tolist())

    missing = sorted(EXPECTED - present)
    if len(m) == 0:
        diagnostics.append('fundamental_metrics.csv contains zero metric rows')
    if len(a) == 0:
        diagnostics.append('fundamental_assessment.csv contains zero assessment rows')
    if bad:
        diagnostics.append(f'{bad} assessment row(s) have publication_date after analysis_date')
    if missing:
        diagnostics.append(f'missing required metrics: {missing}')

    passed = not diagnostics and len(a) > 0 and len(m) > 0
    status = {
        'status': 'PASS' if passed else 'FAIL',
        'analysis_date': args.analysis_date,
        'assessment_rows': int(len(a)),
        'metric_rows': int(len(m)),
        'publication_dates_after_analysis': bad,
        'missing_required_metrics': missing,
        'tickers': tickers,
        'diagnostics': diagnostics,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(json.dumps(status, indent=2))
    raise SystemExit(0 if passed else 1)

if __name__ == '__main__':
    main()
