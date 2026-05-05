"""ND Factor Demo — Visualization module."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

CHART_DIR = Path(__file__).parent.parent / "results" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

_BG = '#FAFAFA'


def plot_nd_distribution(nd_vals, save=True):
    """Histogram of ND values across stocks."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    counts, bins, _ = ax.hist(nd_vals, bins=12, edgecolor='white', linewidth=1.2, color='#4393C3')
    
    ax.axvline(np.mean(nd_vals), color='#D6604D', linestyle='--', linewidth=1.5, label=f'Mean = {np.mean(nd_vals):.3f}')
    ax.legend(fontsize=10)
    ax.set_xlabel('Narrative Dispersion (ND)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Stocks', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Sectional Distribution of ND', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save:
        fig.savefig(CHART_DIR / 'nd_distribution.png', dpi=180, bbox_inches='tight')
    plt.close()
    return fig


def plot_ic_horizons(ic_results, save=True):
    """IC bar chart by forecast horizon."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    horizons = list(ic_results.keys())
    ics = [ic_results[h]['ic_rank'] for h in horizons]
    pvals = [ic_results[h]['bs_pval'] for h in horizons]

    bars = ax.bar(horizons, ics, width=0.5, edgecolor='white', linewidth=1.2)
    for bar, ic, p in zip(bars, ics, pvals):
        bar.set_facecolor('#2166AC' if ic > 0 and p < 0.05 else '#4393C3' if ic > 0 else '#D6604D')
        label = '**' if p < 0.01 else '*' if p < 0.05 else f'p={p:.3f}'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{label}\nIC={ic:.3f}', ha='center', va='bottom', fontsize=8,
                color='#2166AC' if ic > 0 else '#D6604D', fontweight='bold')

    ax.axhline(y=0, color='#333', linewidth=0.8)
    ax.set_ylabel('Rank IC', fontsize=12, fontweight='bold')
    ax.set_xlabel('Forecast Horizon', fontsize=12, fontweight='bold')
    ax.set_title('ND Cross-Sectional IC by Horizon', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save:
        fig.savefig(CHART_DIR / 'ic_by_horizon.png', dpi=180, bbox_inches='tight')
    plt.close()
    return fig


def plot_quintile_returns(returns_5d, returns_10d, save=True):
    """Inverted-U quintile return chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(_BG)

    for ax, rets, title in [(ax1, returns_5d, '5-Day'), (ax2, returns_10d, '10-Day')]:
        ax.set_facecolor(_BG)
        labels = ['Q1\n(Low)', 'Q2', 'Q3', 'Q4', 'Q5\n(High)']
        x = np.arange(5)
        colors = ['#D73027' if r < 0 else '#4393C3' for r in rets]
        bars = ax.bar(x, [r * 100 for r in rets], width=0.55, edgecolor='white', linewidth=1.2, color=colors)
        
        for i, (bar, r) in enumerate(zip(bars, rets)):
            va = 'bottom' if r >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.1 if r >= 0 else -0.1),
                    f'{r*100:.2f}%', ha='center', va=va, fontsize=9, fontweight='bold',
                    color='#D73027' if r < 0 else '#2166AC')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('Mean Return (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{title} Quintile Portfolio Returns', fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axhline(y=0, color='#333', linewidth=0.5)

    plt.suptitle('ND Risk Groups: The Goldilocks Effect', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig(CHART_DIR / 'quintile_returns.png', dpi=180, bbox_inches='tight')
    plt.close()
    return fig


def plot_nd_vs_returns(nd_vals, ret_vals, save=True):
    """Scatter plot of ND vs forward returns with smoothed trend."""
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    ax.scatter(nd_vals, [r * 100 for r in ret_vals], alpha=0.4, s=12, c='#4393C3', edgecolors='white', linewidth=0.3)
    
    # Smoothed trend
    sort_idx = np.argsort(nd_vals)
    nd_s, ret_s = np.array(nd_vals)[sort_idx], np.array(ret_vals)[sort_idx]
    window = max(20, len(nd_s) // 10)
    rolling_mean = np.convolve(ret_s * 100, np.ones(window)/window, mode='valid')
    rolling_nd = np.convolve(nd_s, np.ones(window)/window, mode='valid')
    ax.plot(rolling_nd, rolling_mean, color='#D6604D', linewidth=2.5, label='Smoothed Trend')
    
    ax.axhline(y=0, color='#333', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Narrative Dispersion (ND)', fontsize=12, fontweight='bold')
    ax.set_ylabel('10-Day Forward Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('ND vs. 10-Day Forward Returns', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save:
        fig.savefig(CHART_DIR / 'nd_vs_returns_scatter.png', dpi=180, bbox_inches='tight')
    plt.close()
    return fig
