import argparse, hashlib, json, re, zipfile
from pathlib import Path
from datetime import datetime
import openpyxl
import pandas as pd

STATEMENTS = {
    "1210000": {"statement":"statement_of_financial_position", "period_type":"instant", "cumulative":False},
    "1321000": {"statement":"profit_and_loss", "period_type":"duration", "cumulative":True},
    "1410000": {"statement":"statement_of_changes_in_equity", "period_type":"duration", "cumulative":True},
    "1510000": {"statement":"cash_flow", "period_type":"duration", "cumulative":True},
}

CANONICAL = {
    "cash_and_cash_equivalents":"cash",
    "total_current_assets":"total_current_assets",
    "total_assets":"total_assets",
    "total_current_liabilities":"total_current_liabilities",
    "total_liabilities":"total_liabilities",
    "total_equity":"total_equity",
    "sales_and_revenue":"revenue",
    "revenue":"revenue",
    "total_gross_profit":"gross_profit",
    "operating_profit_loss":"operating_profit",
    "total_profit_loss_before_tax":"profit_before_tax",
    "profit_loss_for_the_period":"net_income",
    "profit_for_the_period":"net_income",
    "total_profit_loss_from_continuing_operations":"net_income_continuing",
    "total_profit_loss":"net_income",
    "profit_loss_attributable_to_parent_entity":"net_income_attributable_parent",
    "profit_loss_attributable_to_non_controlling_interests":"net_income_attributable_nci",
    "total_net_cash_flows_received_from_used_in_operating_activities":"cfo",
    "total_net_cash_flows_received_from_used_in_investing_activities":"cfi",
    "total_net_cash_flows_received_from_used_in_financing_activities":"cff",
    "total_net_increase_decrease_in_cash_and_cash_equivalents":"net_change_cash",
}

def snake(s):
    s = str(s or "").strip().lower()
    s = re.sub(r"[’'()\[\],.:;/\\-]+", " ", s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def parse_date(v):
    if v is None or v=="": return None
    if hasattr(v,'date'): return v.date().isoformat()
    s=str(v).strip()
    for fmt in ("%Y-%m-%d","%d/%m/%Y","%m/%d/%Y"):
        try: return datetime.strptime(s,fmt).date().isoformat()
        except ValueError: pass
    return s

def meta_from_general(ws):
    m={}
    for r in ws.iter_rows(values_only=True):
        if len(r)>=2 and r[0] not in (None,""):
            m[str(r[0]).strip()] = r[1]
    return m

def build_rows(xlsx, publication_date, source_url):
    wb=openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    general=meta_from_general(wb["1000000"])
    ticker=str(general.get("Kode entitas") or "").strip()
    company=str(general.get("Nama entitas") or "").strip()
    currency=str(general.get("Mata uang pelaporan") or "").strip()
    rounding=str(general.get("Pembulatan yang digunakan dalam penyajian jumlah dalam laporan keuangan") or "").strip()
    board_date=parse_date(general.get("Tanggal Surat Pernyataan Direksi"))
    report_date=parse_date(general.get("Tanggal laporan audit atau hasil laporan review"))
    period_start=parse_date(general.get("Tanggal awal periode berjalan"))
    period_end=parse_date(general.get("Tanggal akhir periode berjalan"))
    prior_start=parse_date(general.get("Tanggal awal periode sebelumnya"))
    prior_end=parse_date(general.get("Tanggal akhir periode sebelumnya"))
    prior_year_end=parse_date(general.get("Tanggal akhir tahun sebelumnya"))
    report_type=str(general.get("Jenis laporan atas laporan keuangan") or "").strip()
    audit_opinion=str(general.get("Jenis opini auditor") or "").strip()
    document_id=sha256(xlsx)
    rows=[]
    for code, cfg in STATEMENTS.items():
        if code not in wb.sheetnames: continue
        ws=wb[code]
        # Standard IDX financial statement layout: row 4 contains period tokens, rows start at 5.
        for i,row in enumerate(ws.iter_rows(min_row=5, values_only=True), start=5):
            if len(row)<4: continue
            label_id,label_en=row[0],row[3]
            if label_id is None and label_en is None: continue
            label_id=str(label_id or "").strip(); label_en=str(label_en or label_id).strip()
            key=snake(label_en)
            if not key or key in {"assets","current_assets","non_current_assets","liabilities","equity","cash_flows_from_operating_activities","cash_flows_from_investing_activities","cash_flows_from_financing_activities"}: continue
            vals=[row[1] if len(row)>1 else None, row[2] if len(row)>2 else None]
            periods=[(vals[0],period_start,period_end,True),(vals[1],prior_start,prior_end,True)]
            if cfg["period_type"]=="instant":
                periods=[(vals[0],period_end,period_end,False),(vals[1],prior_year_end,prior_year_end,False)]
            for value,pstart,pend,is_current in periods:
                if value is None or not isinstance(value,(int,float)) or pd.isna(value): continue
                metric_key=CANONICAL.get(key,key)
                unit = "USD_thousand"
                if "earnings_per_share" in key or "per_share" in key:
                    unit = "USD_per_share"
                elif "percentage" in key or key.endswith("_percent"):
                    unit = "percent"
                rows.append({
                    "ticker":ticker,"company_name":company,"statement":cfg["statement"],
                    "metric_key":metric_key,"metric_label":label_en,"metric_label_id":label_id,
                    "value":float(value),"unit":unit,"currency":"USD",
                    "period_start":pstart,"period_end":pend,"period_type":cfg["period_type"],
                    "is_cumulative":cfg["cumulative"],"publication_date":publication_date,
                    "report_date":report_date,"source":"IDX_XLSX","source_url":source_url,
                    "document_id":document_id,"version":"1","confidence":"HIGH",
                    "board_statement_date":board_date,"report_type":report_type,"audit_opinion":audit_opinion,
                    "source_rounding":rounding,"sheet_code":code,"source_row":i,"is_current_period":is_current
                })
    meta={"ticker":ticker,"company_name":company,"currency":currency,"rounding":rounding,
          "period_start":period_start,"period_end":period_end,"prior_start":prior_start,"prior_end":prior_end,
          "prior_year_end":prior_year_end,"board_statement_date":board_date,"report_date":report_date,
          "publication_date":publication_date,"report_type":report_type,"audit_opinion":audit_opinion,
          "document_id":document_id,"source_url":source_url,"engine_changed":False}
    return pd.DataFrame(rows),meta

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--xlsx',required=True); ap.add_argument('--publication-date',required=True)
    ap.add_argument('--source-url',default='https://www.idx.co.id/id/perusahaan-tercatat/keterbukaan-informasi/')
    ap.add_argument('--output-dir',default='data/fundamental')
    args=ap.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    df,meta=build_rows(args.xlsx,args.publication_date,args.source_url)
    df.to_csv(out/'financial_statements.csv',index=False)
    (out/'financial_import_metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    status={"status":"IMPORT_OK" if len(df)>0 else "IMPORT_EMPTY","engine_changed":False,
            "ticker":meta["ticker"],"rows":len(df),"document_id":meta["document_id"],
            "publication_date":args.publication_date,"period_end":meta["period_end"],
            "statements":sorted(df.statement.unique().tolist()) if len(df) else []}
    (out/'acquisition_status.json').write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(status,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
