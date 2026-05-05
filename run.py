#!/usr/bin/env python3
"""
ND (Narrative Dispersion) Factor Demo — Main Entry Point
《叙事离散度》因子 — 市场结构风险指标 — 学术验证与复现

Usage:
    python run.py                  # Standard demo (50 stocks)
    python run.py --full           # Full market analysis (CSI 300, ~3 min)
    python run.py --demo           # Quick demo with pre-computed results
    python run.py --charts-only    # Regenerate charts from cached results
"""
import sys, os, json, argparse, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Add project root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

CACHE_DIR = ROOT / "data"
CACHE_DIR.mkdir(exist_ok=True)
CHART_DIR = ROOT / "results" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)


def print_header():
    print("=" * 68)
    print("  ND (Narrative Dispersion) Factor — Academic Verification")
    print("  《叙事离散度》因子 — 市场结构风险指标 — 学术验证与复现")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 68)


def step1_data(stock_codes):
    """Fetch post data."""
    from src.data_fetcher import fetch_universe_posts
    print(f"\n  [Step 1] Fetching data for {len(stock_codes)} stocks...")
    t0 = time.time()
    posts = fetch_universe_posts(stock_codes, max_pages=1, use_cache=True)
    total = sum(len(d['titles']) for d in posts.values())
    print(f"    Done: {len(posts)} stocks, {total:,} total posts ({time.time()-t0:.0f}s)")
    return posts


def step2_nd(posts):
    """Compute ND factor."""
    from src.factor_calculator import compute_nd_factor
    print(f"\n  [Step 2] Computing Narrative Dispersion...")
    t0 = time.time()
    results = compute_nd_factor(posts, use_cache=True)
    nd_vals = [r['nd'] for r in results]
    print(f"    ND computed for {len(results)} stocks ({time.time()-t0:.0f}s)")
    print(f"    ND range: [{min(nd_vals):.4f}, {max(nd_vals):.4f}], mean={np.mean(nd_vals):.4f}")
    return results


def step3_prices(stock_codes):
    """Fetch price data."""
    import akshare as ak
    from datetime import timedelta
    print(f"\n  [Step 3] Fetching price data...")
    t0 = time.time()
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=45)).strftime('%Y%m%d')
    prices = {}
    for i, code in enumerate(stock_codes):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
            if not df.empty:
                prices[code] = df
        except:
            pass
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(stock_codes)} ({time.time()-t0:.0f}s)")
    print(f"    Done: {len(prices)} stocks ({time.time()-t0:.0f}s)")
    return prices


def step4_ic(nd_results, prices):
    """Run IC test."""
    from src.ic_test import compute_ic
    print(f"\n  [Step 4] Cross-Sectional IC Test...")
    t0 = time.time()
    ic_results = compute_ic(nd_results, prices)
    for h, r in ic_results.items():
        print(f"    {h:>4s}: IC={r['ic_rank']:+.4f}, ICIR={r['icir']:+.3f}, p={r['bs_pval']:.4f} ({r['verdict']})")
    print(f"    ({time.time()-t0:.0f}s)")
    return ic_results


def step5_quintile(nd_results, prices):
    """Quintile analysis."""
    from src.backtest import quintile_analysis
    print(f"\n  [Step 5] Quintile Portfolio Analysis...")
    t0 = time.time()
    for h in [5, 10]:
        panel, stats = quintile_analysis(nd_results, prices, horizon=h)
        if 'group' in panel.columns:
            print(f"    {h}-Day Quintile Returns:")
            for g in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                if g in stats.index:
                    ret = stats.loc[g, 'mean_return_pct']
                    print(f"      {g}: {ret:+.2f}%  ({int(stats.loc[g, 'count'])} stocks)")
    print(f"    ({time.time()-t0:.0f}s)")


def step6_charts(nd_results, ic_results, price_data):
    """Generate charts."""
    print(f"\n  [Step 6] Generating publication-quality charts...")
    t0 = time.time()
    from src.visualization import plot_nd_distribution, plot_ic_horizons, plot_quintile_returns, plot_nd_vs_returns

    nd_vals = [r['nd'] for r in nd_results]
    plot_nd_distribution(nd_vals)
    plot_ic_horizons(ic_results)

    # Quintile returns
    from src.backtest import quintile_analysis
    _, stats_5d = quintile_analysis(nd_results, price_data, horizon=5)
    _, stats_10d = quintile_analysis(nd_results, price_data, horizon=10)
    if 'mean_return' in stats_5d.columns:
        plot_quintile_returns(stats_5d['mean_return'].tolist(), stats_10d['mean_return'].tolist())

    # Scatter
    nd_map = {r['code']: r['nd'] for r in nd_results}
    nd_arr, ret_arr = [], []
    for code, df in price_data.items():
        if code in nd_map and len(df) >= 11:
            closes = df['收盘'].values
            nd_arr.append(nd_map[code])
            ret_arr.append(closes[-1] / closes[-11] - 1)
    if nd_arr:
        plot_nd_vs_returns(nd_arr, ret_arr)

    chart_files = list(CHART_DIR.glob("*.png"))
    print(f"    Generated {len(chart_files)} charts ({time.time()-t0:.0f}s)")
    for f in chart_files:
        print(f"      {f.name}")


def main():
    parser = argparse.ArgumentParser(description='ND Factor Academic Demo')
    parser.add_argument('--full', action='store_true', help='Full CSI 300 analysis')
    parser.add_argument('--demo', action='store_true', help='Quick demo with cached results')
    parser.add_argument('--charts-only', action='store_true', help='Regenerate charts from cache')
    parser.add_argument('--stocks', type=int, default=50, help='Number of stocks (default: 50)')
    args = parser.parse_args()

    print_header()

    if args.demo:
        print("\n  [Demo Mode] Loading pre-computed results...")
        cache_file = CACHE_DIR / "nd_results.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                nd_data = json.load(f)
            nd_results = nd_data['results']
            nd_vals = [r['nd'] for r in nd_results]
            print(f"    Loaded {len(nd_results)} ND results")
            print(f"    ND range: [{min(nd_vals):.4f}, {max(nd_vals):.4f}]")
            print(f"    ND mean:  {np.mean(nd_vals):.4f}")
            print(f"\n  [Done] Use 'python run.py --charts-only' to regenerate charts")
        else:
            print("    Error: No cached results found.")
        return

    if args.charts_only:"}],[{
        print("\n  [Charts Only] Loading cached results...")
        cache_file = CACHE_DIR / "nd_results.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                nd_data = json.load(f)
            nd_results = nd_data['results']
            nd_vals = [r['nd'] for r in nd_results]
            print(f"    Loaded {len(nd_results)} ND results")
            plot_nd_distribution(nd_vals)
            print(f"    Charts regenerated in {CHART_DIR}")
        else:
            print("    Error: No cached results found. Run 'python run.py --demo' first.")
        return

    # Stock universe
    import akshare as ak
    print("\n  [Universe] Getting CSI 300 constituents...")
    try:
        df = ak.index_stock_cons(symbol="000300")
        codes = df['品种代码'].astype(str).str.zfill(6).tolist()
    except:
        codes = [str(i).zfill(6) for i in range(1, 1000)]
    
    if not args.full:
        codes = codes[:args.stocks]

    posts = step1_data(codes)
    nd_results = step2_nd(posts)
    stock_codes = [r['code'] for r in nd_results]
    prices = step3_prices(stock_codes)
    ic_results = step4_ic(nd_results, prices)
    step5_quintile(nd_results, prices)
    step6_charts(nd_results, ic_results, prices)

    print(f"\n{'=' * 68}")
    print("  Analysis Complete!")
    print(f"  Results: {CHART_DIR.parent}")
    print(f"{'=' * 68}")


if __name__ == '__main__':
    main()
