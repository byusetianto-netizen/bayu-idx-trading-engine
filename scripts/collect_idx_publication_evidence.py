import csv, json, os, re, time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fundamental"
EVIDENCE = DATA / "publication_evidence.csv"
STATUS = DATA / "publication_collection_status.json"
RAW = DATA / "publication_collection_raw.json"

TICKER_FILE = DATA / "research_tickers.txt"
DEFAULT_TICKERS = ["AADI", "AALI", "ABBA"]

IDX_DISCLOSURE = "https://www.idx.co.id/id/perusahaan-tercatat/keterbukaan-informasi/"
IDX_PROFILE = "https://www.idx.co.id/id/perusahaan-tercatat/profil-perusahaan-tercatat/{ticker}"
ANNOUNCE_ENDPOINT = "https://www.idx.co.id/primary/NewsAnnouncement/GetAnnouncement?kodeEmiten={ticker}&lang=id"
FIN_REPORT_ENDPOINT = "https://www.idx.co.id/primary/ListedCompany/GetFinancialReport?periode={period}&year={year}&indexFrom=0&pageSize=1000&reportType=rdf&kodeEmiten={ticker}"

FIN_KW = [
    "laporan keuangan", "financial report", "financial statement",
    "laporan keuangan interim", "laporan keuangan tahunan",
    "penyampaian laporan keuangan"
]
PERIOD_MAP = {"TW1":"TW1", "TW2":"TW2", "TW3":"TW3", "audit":"audit", "TAHUNAN":"audit"}


def load_tickers():
    if TICKER_FILE.exists():
        vals = [x.strip().upper() for x in TICKER_FILE.read_text().splitlines() if x.strip()]
        if vals:
            return sorted(set(vals))
    return DEFAULT_TICKERS


def load_existing():
    rows=[]
    if EVIDENCE.exists():
        with EVIDENCE.open(newline='', encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
    return rows


def flatten_strings(obj, path=""):
    out=[]
    if isinstance(obj, dict):
        for k,v in obj.items():
            p=f"{path}.{k}" if path else k
            out.extend(flatten_strings(v,p))
    elif isinstance(obj, list):
        for i,v in enumerate(obj):
            out.extend(flatten_strings(v,f"{path}[{i}]"))
    elif isinstance(obj, str):
        out.append((path,obj))
    return out


def parse_timestamp(s):
    if not isinstance(s,str): return None
    s=s.strip()
    patterns=[
        r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})[ T](\d{1,2}):(\d{2})(?::(\d{2}))?",
        r"(\d{1,2})[ -](\w+)[ -](\d{4})[ ,]+(\d{1,2}):(\d{2})",
    ]
    months={"jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,"mei":5,"may":5,"jun":6,"june":6,"jul":7,"july":7,"agu":8,"aug":8,"august":8,"sep":9,"september":9,"okt":10,"oct":10,"nov":11,"des":12,"dec":12}
    m=re.search(patterns[0],s,re.I)
    if m:
        y,mo,d,hh,mi,ss=m.groups(); return f"{int(y):04d}-{int(mo):02d}-{int(d):02d} {int(hh):02d}:{int(mi):02d}:{int(ss or 0):02d}"
    m=re.search(patterns[1],s,re.I)
    if m:
        d,mon,y,hh,mi=m.groups(); mon=months.get(mon.lower());
        if mon: return f"{int(y):04d}-{mon:02d}-{int(d):02d} {int(hh):02d}:{int(mi):02d}:00"
    return None


def find_candidate_timestamps(obj, ticker, year, period):
    # Conservative: inspect each dict/list record and only accept records whose
    # textual content looks like a financial-report disclosure for this issuer/year.
    candidates=[]
    def walk(x):
        if isinstance(x, dict):
            text=" ".join(str(v) for v in x.values() if isinstance(v,(str,int,float))).lower()
            is_fin=any(k.lower() in text for k in FIN_KW)
            has_ticker=ticker.lower() in text
            has_year=str(year) in text
            if is_fin and (has_ticker or not ticker) and has_year:
                ts=None; title=""; code=""
                for k,v in x.items():
                    kl=str(k).lower()
                    if isinstance(v,str):
                        p=parse_timestamp(v)
                        if p and not ts: ts=p
                        if any(z in kl for z in ["title","judul","description","subject"]): title=v
                        if any(z in kl for z in ["type","code","template"]): code=v
                if ts:
                    candidates.append({"timestamp":ts,"title":title,"code":code,"record":x})
            for v in x.values(): walk(v)
        elif isinstance(x,list):
            for v in x: walk(v)
    walk(obj)
    return candidates


def browser_fetch(urls):
    # Executed inside Playwright in the GitHub Action. The page is opened on the
    # real IDX origin first, then fetch() runs in the browser context so the
    # anti-bot/session layer has a chance to establish.
    from playwright.sync_api import sync_playwright
    result={}
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
        context=browser.new_context(locale="id-ID", timezone_id="Asia/Jakarta")
        page=context.new_page()
        page.goto(IDX_DISCLOSURE, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(4000)
        for key,url in urls.items():
            try:
                data=page.evaluate("""async (url) => {
                    const r = await fetch(url, {credentials:'include'});
                    return {status:r.status, text:await r.text()};
                }""", url)
                result[key]=data
            except Exception as e:
                result[key]={"status":0,"text":"","error":str(e)}
        browser.close()
    return result


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    tickers=load_tickers()
    year=int(os.getenv("IDX_PUBLICATION_YEAR", "2026"))
    period=os.getenv("IDX_PUBLICATION_PERIOD", "TW1")
    urls={}
    for t in tickers:
        urls[f"ann_{t}"]=ANNOUNCE_ENDPOINT.format(ticker=quote(t))
        urls[f"fin_{t}"]=FIN_REPORT_ENDPOINT.format(ticker=quote(t),period=period,year=year)
    fetched={}
    try:
        fetched=browser_fetch(urls)
    except Exception as e:
        status={"status":"BROWSER_COLLECTOR_UNAVAILABLE","engine_changed":False,"tickers":tickers,"error":str(e)}
        STATUS.write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps(status,indent=2,ensure_ascii=False)); return

    raw_meta={}; discovered=[]; errors=[]
    for key,res in fetched.items():
        text=res.get("text","")
        raw_meta[key]={"status":res.get("status"),"bytes":len(text),"error":res.get("error")}
        if res.get("status")!=200:
            errors.append({"request":key,"status":res.get("status"),"error":res.get("error")})
            continue
        try:
            obj=json.loads(text)
        except Exception:
            continue
        if key.startswith("ann_"):
            t=key[4:]
            cs=find_candidate_timestamps(obj,t,year,period)
            for c in cs:
                discovered.append({"ticker":t,"period_end":"","period_code":period,"year":year,
                    "publication_date":c["timestamp"],"publication_timezone":"Asia/Jakarta","source":"IDX",
                    "source_url":IDX_DISCLOSURE,"evidence_type":"IDX disclosure index","confidence":"MEDIUM",
                    "notes":"Automatically discovered; requires exact filing-period reconciliation before PIT-safe use."})

    existing=load_existing()
    # Merge by ticker + period_code + year + publication_date, preserving high-confidence manual evidence.
    merged={}
    for r in existing+discovered:
        key=(r.get("ticker",""),r.get("period_code",""),r.get("year",""),r.get("publication_date",""))
        if key not in merged or merged[key].get("confidence")!="HIGH": merged[key]=r
    fields=["ticker","period_end","period_code","year","publication_date","publication_timezone","source","source_url","evidence_type","confidence","notes"]
    with EVIDENCE.open("w",newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(sorted(merged.values(),key=lambda x:(x.get('ticker',''),x.get('publication_date',''))))

    raw=DATA/"publication_collection_raw.json"
    raw.write_text(json.dumps(raw_meta,indent=2,ensure_ascii=False),encoding='utf-8')
    status={
        "status":"COLLECTION_COMPLETE" if not errors else "COLLECTION_PARTIAL",
        "engine_changed":False,
        "tickers_requested":len(tickers),"tickers":tickers,
        "discovered_candidates":len(discovered),"stored_evidence_rows":len(merged),
        "http_errors":len(errors),"errors":errors[:20],
        "notes":["Automatic collector never invents publication timestamps.","High-confidence manual IDX evidence is preserved until automatic evidence can replace it with equivalent evidence."]
    }
    STATUS.write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(status,indent=2,ensure_ascii=False))

if __name__=="__main__": main()
