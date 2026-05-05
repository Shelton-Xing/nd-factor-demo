"""
ND Factor Demo — IC Test Suite
Cross-sectional Rank IC with bootstrap inference.
"""
import numpy as np
from scipy import stats


def compute_ic(nd_results, price_data):
    """
    Compute cross-sectional Rank IC for multiple horizons.

    Parameters
    ----------
    nd_results : list of dict
        [{'code': ..., 'nd': ...}, ...]
    price_data : dict
        stock_code -> akshare DataFrame with '收盘'

    Returns
    -------
    dict : horizon -> {'ic_rank': ..., 'p_val': ..., 'icir': ..., 'verdict': ...}
    """
    nd_map = {r['code']: r['nd'] for r in nd_results}

    ic_results = {}
    for horizon, col_suffix in [('1d', 1), ('5d', 5), ('10d', 10), ('20d', 20)]:

        stocks = []
        nd_vals = []
        ret_vals = []

        for code, df in price_data.items():
            if code not in nd_map:
                continue
            closes = df['收盘'].values
            if len(closes) < col_suffix + 1:
                continue
            ret = closes[-1] / closes[-(col_suffix + 1)] - 1
            stocks.append(code)
            nd_vals.append(nd_map[code])
            ret_vals.append(ret)

        if len(stocks) < 10:
            continue

        nd_arr = np.array(nd_vals)
        ret_arr = np.array(ret_vals)

        # Spearman rank IC
        ic_rank, p_val = stats.spearmanr(nd_arr, ret_arr)

        # Bootstrap CI
        n_bootstrap = 10000
        bs_ics = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(stocks), len(stocks), replace=True)
            bs_ic, _ = stats.spearmanr(nd_arr[idx], ret_arr[idx])
            bs_ics.append(bs_ic)

        bs_ics = np.array(bs_ics)
        ic_mean = float(np.mean(bs_ics))
        ic_std = float(np.std(bs_ics, ddof=1))
        icir = ic_mean / ic_std if ic_std > 0 else 0

        # Bootstrap p-value
        bs_p = float(2 * min(np.mean(bs_ics >= 0), np.mean(bs_ics <= 0)))

        if bs_p < 0.01:
            verdict = 'SIGNIFICANT (p<0.01)'
        elif bs_p < 0.05:
            verdict = 'SIGNIFICANT (p<0.05)'
        elif bs_p < 0.10:
            verdict = 'MARGINAL (p<0.10)'
        else:
            verdict = 'INSIGNIFICANT'

        ic_results[horizon] = {
            'ic_rank': ic_rank,
            'p_val': p_val,
            'ic_mean_bs': ic_mean,
            'icir': icir,
            'bs_pval': bs_p,
            'verdict': verdict,
            'n_stocks': len(stocks)
        }

    return ic_results
