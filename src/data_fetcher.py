"""
ND Factor Demo — Data Fetcher
Public API for fetching post data from East Money Guba.
"""
import requests, re, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://guba.eastmoney.com/",
}
CACHE_DIR = Path(__file__).parent.parent / "data"
CACHE_DIR.mkdir(exist_ok=True)


def fetch_stock_posts(stock_code, page=1):
    """
    Fetch post titles from East Money Guba for a single stock.

    Parameters
    ----------
    stock_code : str
        6-digit Chinese stock code (e.g., "600519")
    page : int
        Page number (default: 1)

    Returns
    -------
    list of dict
        [{'title': str, 'timestamp': str}, ...]
    """
    url = f"https://guba.eastmoney.com/list,{stock_code}_{page}.html"
    resp = requests.get(url, headers=HEADERS, timeout=15)

    match = re.search(r'var\s+article_list\s*=\s*(\{.*?\});', resp.text, re.DOTALL)
    if not match:
        return []

    data = json.loads(match.group(1))
    posts = []
    for p in data.get('re', []):
        title = p.get('post_title', '').strip()
        ts = p.get('post_last_time', '')
        if title and len(title) > 5:
            posts.append({'title': title, 'timestamp': ts[:10] if ts else ''})
    return posts


def fetch_universe_posts(codes, max_pages=1, max_workers=8, use_cache=True):
    """
    Fetch posts for a list of stocks (parallel).

    Parameters
    ----------
    codes : list of str
        Stock codes
    max_pages : int
        Pages per stock
    max_workers : int
        Parallel workers
    use_cache : bool
        Use cached results if available

    Returns
    -------
    dict : stock_code -> {'titles': [...], 'timestamps': [...]}
    """
    cache_file = CACHE_DIR / "posts_cache.json"

    if use_cache and cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    results = {}

    def fetch_one(code):
        posts = fetch_stock_posts(code, page=1)
        return code, [p['title'] for p in posts], [p['timestamp'] for p in posts]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, c): c for c in codes}
        for future in as_completed(futures):
            code, titles, timestamps = future.result()
            results[code] = {'titles': titles, 'timestamps': timestamps}
            time.sleep(0.05)

    if use_cache:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False)

    return results
