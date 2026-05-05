"""
ND Factor Demo — Factor Calculator
Computes Narrative Dispersion from post titles.
Public API: compute_nd_factor()
"""
import sys, os, json
import numpy as np
from pathlib import Path

# Black-box core
sys.path.insert(0, str(Path(__file__).parent.parent))
from core import compute_narrative_divergence

CACHE_DIR = Path(__file__).parent.parent / "data"
CACHE_DIR.mkdir(exist_ok=True)


def compute_nd_factor(posts_data, use_cache=True):
    """
    Compute ND (Narrative Dispersion) for each stock.

    Parameters
    ----------
    posts_data : dict
        stock_code -> {'titles': [...], 'timestamps': [...]}
    use_cache : bool
        Use cached ND results if available

    Returns
    -------
    list of dict
        [{'code': str, 'n_posts': int, 'nd': float, 'nd_std': float, ...}, ...]
    """
    cache_file = CACHE_DIR / "nd_results.json"

    if use_cache and cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)['results']

    results = []
    for code, data in posts_data.items():
        titles = data['titles']
        n = len(titles)
        if n < 5:
            continue

        nd, nd_std, sims = compute_narrative_divergence(titles)

        results.append({
            'code': code,
            'n_posts': n,
            'nd': nd,
            'nd_std': nd_std
        })

    # Cache
    if use_cache:
        nd_vals = [r['nd'] for r in results]
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': 'cached',
                'n_stocks': len(results),
                'total_posts': sum(r['n_posts'] for r in results),
                'nd_mean': float(np.mean(nd_vals)),
                'nd_std': float(np.std(nd_vals, ddof=1)),
                'results': results
            }, f, ensure_ascii=False, indent=2)

    return results
