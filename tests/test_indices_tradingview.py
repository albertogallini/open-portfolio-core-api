"""
Test script to check which indices from get_indices have working
TradingView /components/ pages. 

Run with:
    cd /Users/albertogallini/pythonprj/oport
    python -m pytest tests/test_indices_tradingview.py -s -v
or standalone:
    python tests/test_indices_tradingview.py
"""

import requests
import time
from bs4 import BeautifulSoup

# ─── Same cookies/headers as oport_api.py ───────────────────────────────────
HARVESTED_COOKIES = {
    'sessionid': 'REMOVED_SECRET',
    'sessionid_sign': 'REMOVED_SECRET',
    '_sp_id.cf1a': 'REMOVED_SECRET',
    '_sp_ses.cf1a': '*',
    'device_t': 'REMOVED_SECRET',
    'etg': '31fe40b3-4542-4e0a-958a-3ea3949bb1fe',
    'png': '31fe40b3-4542-4e0a-958a-3ea3949bb1fe',
    'cachec': '31fe40b3-4542-4e0a-958a-3ea3949bb1fe',
    'tv_ecuid': '31fe40b3-4542-4e0a-958a-3ea3949bb1fe',
    'cookiePrivacyPreferenceBannerProduction': 'ignored',
    'g_state': '{"i_l":0,"i_ll":1776616386431,"i_e":{"enable_itp_optimization":1},"i_et":1776616386367}',
    'sp': '50b337f0-6623-46fe-989a-2906b4122858',
}

REQUEST_HEADERS = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-encoding': 'gzip, deflate, br, zstd',
    'accept-language': 'en-US,en;q=0.9,it;q=0.8',
    'cache-control': 'max-age=0',
    'priority': 'u=0, i',
    'sec-ch-ua': '"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Linux"',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
}

# ─── All indices from get_indices ─────────────────────────────────────────────
ALL_INDICES = [
    # US Indices
    {"label": "S&P 500 Index",                          "value": "SPX"},
    {"label": "S&P 400",                                "value": "MID"},
    {"label": "S&P 100 Index",                          "value": "OEX"},
    {"label": "Dow Jones Industrial Average Index",     "value": "DJCFD-DJI"},
    {"label": "NASDAQ Composite Index",                 "value": "IXIC"},
    {"label": "NASDAQ 100 Index",                       "value": "NDX"},
    {"label": "US Small Cap 3000 Index",                "value": "RUA"},
    {"label": "US Small Cap 2000 Index",                "value": "RUT"},
    {"label": "US Small Cap 1000 Index",                "value": "RUI"},
    {"label": "NYSE COMPOSITE INDEX",                   "value": "NYA"},
    {"label": "NYSE AMERICAN COMPOSITE INDEX",          "value": "XAX"},
    {"label": "PHLX Oil Service Sector",                "value": "OSX"},
    {"label": "PHLX Gold/Silver Sector",                "value": "XAU"},
    {"label": "PHLX Housing Sector",                    "value": "HGX"},
    {"label": "PHLX Utility Sector",                    "value": "UTY"},
    {"label": "PHLX Semiconductor",                     "value": "SOX"},
    # Europe Indices
    {"label": "DAX Index",                              "value": "DAX"},
    {"label": "CAC 40 Index",                           "value": "PX1"},
    {"label": "FTSE 100 Index",                         "value": "UKX"},
    {"label": "OMX Stockholm 30 Index",                 "value": "OMXS30"},
    {"label": "IBEX 35 Index",                          "value": "IBC"},
    {"label": "AEX-Index",                              "value": "AEX"},
    {"label": "STOXX 50",                               "value": "SX5E"},
    {"label": "STOXX 600",                              "value": "SXXP"},
    {"label": "Swiss Market Index",                     "value": "SMI"},
    {"label": "FTSE MIB Index",                         "value": "FTSEMIB"},
    {"label": "WIG20 Index",                            "value": "WIG20"},
    {"label": "BIST 100 Index",                         "value": "XU100"},
    {"label": "Russia Index",                           "value": "IRUS"},
    {"label": "MDAX Index",                             "value": "MDAX"},
    {"label": "Oslo Børs Benchmark GI Index",           "value": "OSEBX"},
    {"label": "OMX Copenhagen 25 Index",                "value": "OMXC25"},
    {"label": "BEL 20 Index",                           "value": "BEL20"},
    {"label": "Budapest Stock Exchange Index",          "value": "BUX"},
    {"label": "Austrian Traded Index in EUR",            "value": "ATX"},
    {"label": "PX Index",                               "value": "PX"},
    {"label": "PSI Index",                              "value": "PSI20"},
    {"label": "DAXK Kursindex",                         "value": "DAXK"},
    {"label": "TecDAX Index",                           "value": "TDXP"},
    {"label": "SDAX Index",                             "value": "SSDXP"},
    # Asia Indices
    {"label": "Japan 225 Index (Nikkei)",               "value": "NI225"},
    {"label": "Hang Seng Index",                        "value": "HSI"},
    {"label": "SSE Composite Index",                    "value": "000001"},
    {"label": "Nifty 50 Index",                         "value": "NIFTY"},
    {"label": "BSE Sensex Index",                       "value": "SENSEX"},
    {"label": "KOSPI Composite Index",                  "value": "KOSPI"},
    {"label": "TSEC Capitalization Weighted Stock Index","value": "IX0001"},
    {"label": "IDX Composite Index",                    "value": "COMPOSITE"},
    {"label": "FBM KLCI",                               "value": "FBMKLCI"},
    {"label": "SET Index",                              "value": "SET"},
    {"label": "VNINDEX",                                "value": "VNINDEX"},
    {"label": "TOPIX Index",                            "value": "TOPIX"},
    {"label": "CSI 300 Index",                          "value": "000300"},
    {"label": "Shenzhen Component Index",               "value": "399001"},
    {"label": "Hang Seng TECH Index",                   "value": "HHSTECH"},
    {"label": "Nifty Bank Index",                       "value": "BANKNIFTY"},
    {"label": "Nifty Financial Services Index",         "value": "CNXFINANCE"},
    {"label": "Nifty MidCap Select Index",              "value": "NIFTY_MID_SELECT"},
    {"label": "Nifty Next 50 Index",                    "value": "NIFTYJR"},
    {"label": "Nifty SmallCap 100 Index",               "value": "CNXSMALLCAP"},
]


def check_index(session: requests.Session, index: dict) -> dict:
    """Check if an index has a working /components/ page with extractable rows."""
    ticker = index["value"]
    label = index["label"]
    url = f"https://www.tradingview.com/symbols/{ticker}/components/"

    session.headers.update({'referer': url})
    try:
        resp = session.get(url, timeout=20)
        status = resp.status_code
        rows = []
        suggestion = ""

        if status == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            rows = soup.find_all("tr", {"class": "row-HX5UXsDj listRow"})

            if not rows:
                # Try to detect redirect / canonical URL
                # Look for canonical link tag
                canonical = soup.find("link", rel="canonical")
                canon_href = canonical["href"] if canonical else ""
                # Check if the page title looks like an index page or a redirect
                title_tag = soup.find("title")
                page_title = title_tag.get_text() if title_tag else ""
                suggestion = f"0 rows found. Title='{page_title[:80]}'. Canonical={canon_href}"
        else:
            suggestion = f"HTTP {status}"

        return {
            "ticker": ticker,
            "label": label,
            "url": url,
            "status": status,
            "rows_found": len(rows),
            "working": status == 200 and len(rows) > 0,
            "suggestion": suggestion,
        }

    except Exception as e:
        return {
            "ticker": ticker,
            "label": label,
            "url": url,
            "status": -1,
            "rows_found": 0,
            "working": False,
            "suggestion": str(e),
        }


def run_all_tests():
    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)
    for name, value in HARVESTED_COOKIES.items():
        session.cookies.set(name, value)

    results = []
    print(f"\n{'='*80}")
    print(f"Testing {len(ALL_INDICES)} indices against TradingView /components/")
    print(f"{'='*80}\n")

    for idx in ALL_INDICES:
        result = check_index(session, idx)
        results.append(result)

        status_icon = "✅" if result["working"] else "❌"
        print(f"{status_icon} [{result['ticker']:20s}] HTTP {result['status']:3d}  rows={result['rows_found']:4d}  {result['label']}")
        if not result["working"] and result["suggestion"]:
            print(f"   ↳ {result['suggestion'][:120]}")

        time.sleep(0.3)  # be polite to TradingView

    # Summary
    working = [r for r in results if r["working"]]
    not_working = [r for r in results if not r["working"]]

    print(f"\n{'='*80}")
    print(f"SUMMARY: {len(working)}/{len(results)} indices working")
    print(f"{'='*80}")

    print(f"\n✅ WORKING ({len(working)}):")
    for r in working:
        print(f"   {r['ticker']:20s} → {r['rows_found']} constituents  [{r['label']}]")

    print(f"\n❌ NOT WORKING ({len(not_working)}):")
    for r in not_working:
        print(f"   {r['ticker']:20s} HTTP {r['status']:3d}  [{r['label']}]")
        if r["suggestion"]:
            print(f"      Hint: {r['suggestion'][:120]}")

    return results


if __name__ == "__main__":
    run_all_tests()
