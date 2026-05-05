"""
ND Factor Demo — Quintile Analysis
Tests the inverted-U relationship between ND and forward returns.
"""
import numpy as np
import pandas as pd


def quintile_analysis(nd_results, price_data, horizon=5):
    """
    Sort stocks into 5 quintiles by ND and compute mean forward return.

    Parameters
    ----------
    nd_results : list of dict
    price_data : dict
    horizon : int (trading days)

    Returns
    -------
    pd.DataFrame with quintile returns, pd.Series with grouping
    """
    nd_map = {r['code']: r['nd'] for r in nd_results}

    rows = []
    for code, df in price_data.items():
        if code not in nd_map:
            continue
        closes = df['收盘'].values
        if len(closes) < horizon + 1:
            continue
        ret = closes[-1] / closes[-(horizon + 1)] - 1
        rows.append({'code': code, 'nd': nd_map[code], f'ret_{horizon}d': ret})

    panel = pd.DataFrame(rows)
    if len(panel) < 25:
        return panel

    panel['group'] = pd.qcut(panel['nd'].rank(method='first'), 5,
                             labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                             duplicates='drop')

    group_stats = panel.groupby('group')[f'ret_{horizon}d'].agg(['mean', 'std', 'count'])
    group_stats.columns = ['mean_return', 'std_return', 'count']
    group_stats['mean_return_pct'] = group_stats['mean_return'] * 100

    return panel, group_stats


def compute_nonmonotonicity(returns):
    """
    Compute how inverted-U-like the return pattern is.
    Score: 0 = monotonic, 1 = perfect inverted-U
    """
    if len(returns) < 5:
        return 0.0
    # Check if Q1 and Q5 are lower than Q2-Q4
    r = np.array(returns)
    q1, q2, q3, q4, q5 = r
    mid_mean = np.mean([q2, q3, q4])
    # Inverted-U if both ends below middle
    score = (mid_mean - q1 + mid_mean - q5) / (2 * abs(mid_mean) + 1e-8)
    return max(0, min(1, score))
