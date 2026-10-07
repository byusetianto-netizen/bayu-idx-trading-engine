"""Import a user-provided authoritative CSV/XLSX into the normalized PIT schema.

This utility intentionally requires an explicit input file. It never scrapes undocumented IDX endpoints.
"""
from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/fundamental/financial_statements.csv'
REQ = ['ticker','statement','metric','value','period_start','period_end','period_type','is_cumulative','publication_date','report_date','source','source_url','document_id','version','confidence']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input_file')
    args = ap.parse_args()
    p = Path(args.input_file)
    df = pd.read_excel(p) if p.suffix.lower() in {'.xlsx','.xls'} else pd.read_csv(p)
    missing = [c for c in REQ if c not in df.columns]
    if missing:
        raise ValueError(f'Missing PIT schema columns: {missing}')
    df.to_csv(OUT, index=False)
    print(f'Imported {len(df)} rows to {OUT}')

if __name__ == '__main__':
    main()
