import argparse
import os

import pandas as pd


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _plot_metric(df: pd.DataFrame, metric: str, out_dir: str):
    import matplotlib.pyplot as plt

    d = df[~df['is_fixed']].copy()
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for i, grp in enumerate(['prc', 'obs']):
        g = d[d['group'] == grp].sort_values(metric, ascending=True)
        color = '#2F855A' if grp == 'prc' else '#C05621'
        ax[i].bar(g['parameter'], g[metric], color=color, alpha=0.85)
        ax[i].set_title(f'{grp.upper()} {metric}')
        ax[i].tick_params(axis='x', rotation=65)
        ax[i].grid(axis='y', alpha=0.2)
        if metric in {'pearson_r_t1_t2', 'icc3_1_t1_t2'}:
            ax[i].axhline(0.4, color='k', ls='--', lw=1)
    plt.tight_layout()
    out = os.path.join(out_dir, f'figure_test_retest_{metric}.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out


def _plot_scatter(df: pd.DataFrame, out_dir: str):
    import matplotlib.pyplot as plt

    d = df[~df['is_fixed']].copy()
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for i, grp in enumerate(['prc', 'obs']):
        g = d[d['group'] == grp]
        ax[i].scatter(g['mean_t1'], g['mean_t2'], s=45, alpha=0.8, color='#2B6CB0')
        for _, row in g.iterrows():
            ax[i].annotate(row['parameter'], (row['mean_t1'], row['mean_t2']), fontsize=7, alpha=0.8)
        lo = min(g['mean_t1'].min(), g['mean_t2'].min())
        hi = max(g['mean_t1'].max(), g['mean_t2'].max())
        ax[i].plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax[i].set_title(f'{grp.upper()} mean(T1) vs mean(T2)')
        ax[i].set_xlabel('mean_t1')
        ax[i].set_ylabel('mean_t2')
        ax[i].grid(alpha=0.2)
    plt.tight_layout()
    out = os.path.join(out_dir, 'figure_test_retest_mean_t1_t2.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out


def _plot_parameter_scatter(wide_df: pd.DataFrame, rel_df: pd.DataFrame, group: str, out_dir: str):
    import matplotlib.pyplot as plt
    import numpy as np

    d = wide_df[(~wide_df['is_fixed']) & (wide_df['group'] == group)].copy()
    d = d.dropna(subset=['t1', 't2'])
    if d.empty:
        return None

    params = sorted(d['parameter'].unique().tolist())
    ncols = 4
    nrows = int(np.ceil(len(params) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax in axes.ravel():
        ax.set_visible(False)

    for i, param in enumerate(params):
        ax = axes.ravel()[i]
        ax.set_visible(True)
        dd = d[d['parameter'] == param]
        x = dd['t1'].to_numpy(dtype=float)
        y = dd['t2'].to_numpy(dtype=float)
        ax.scatter(x, y, s=38, alpha=0.85, color='#2B6CB0')
        lo = min(np.min(x), np.min(y))
        hi = max(np.max(x), np.max(y))
        if lo == hi:
            pad = 0.1 if lo == 0 else abs(lo) * 0.05
            lo -= pad
            hi += pad
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
        rr = rel_df[(rel_df['group'] == group) & (rel_df['parameter'] == param)]
        r = rr['pearson_r_t1_t2'].iloc[0] if not rr.empty else np.nan
        icc = rr['icc3_1_t1_t2'].iloc[0] if not rr.empty else np.nan
        ax.set_title(f'{param}\nr={r:.2f}, ICC={icc:.2f}')
        ax.set_xlabel('t1')
        ax.set_ylabel('t2')
        ax.grid(alpha=0.2)

    plt.tight_layout()
    out = os.path.join(out_dir, f'figure_test_retest_scatter_{group}.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out


def _plot_log_zeta_scatter(wide_df: pd.DataFrame, rel_df: pd.DataFrame, out_dir: str):
    import matplotlib.pyplot as plt
    import numpy as np

    d = wide_df[
        (~wide_df['is_fixed'])
        & (wide_df['group'] == 'obs')
        & (wide_df['parameter'] == 'ze')
    ].copy()
    d = d.dropna(subset=['t1', 't2'])
    if d.empty:
        return None

    x = np.log(d['t1'].to_numpy(dtype=float))
    y = np.log(d['t2'].to_numpy(dtype=float))
    rr = rel_df[(rel_df['group'] == 'obs') & (rel_df['parameter'] == 'ze')]
    r = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan
    icc = np.nan
    if not rr.empty:
        # Show the empirical log-scale Pearson; ICC is not stored on log scale in the table.
        pass

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))
    ax.scatter(x, y, s=28, alpha=0.75, color='#C05621')
    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax.set_title(f'log(zeta) Test-Retest\nr={r:.2f}')
    ax.set_xlabel('log zeta t1')
    ax.set_ylabel('log zeta t2')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    out = os.path.join(out_dir, 'figure_test_retest_scatter_log_ze.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description='Plot test-retest reliability outputs.')
    parser.add_argument('--reliability-csv', required=True)
    parser.add_argument('--wide-csv', required=True)
    parser.add_argument('--fig-out-dir', required=True)
    args = parser.parse_args()

    _ensure_dir(args.fig_out_dir)
    df = pd.read_csv(args.reliability_csv)
    wide_df = pd.read_csv(args.wide_csv)
    if 'is_fixed' not in df.columns:
        df['is_fixed'] = False
    if 'is_fixed' not in wide_df.columns:
        wide_df['is_fixed'] = False

    files = [
        _plot_metric(df, 'pearson_r_t1_t2', args.fig_out_dir),
        _plot_metric(df, 'icc3_1_t1_t2', args.fig_out_dir),
        _plot_metric(df, 'mean_abs_delta', args.fig_out_dir),
        _plot_scatter(df, args.fig_out_dir),
        _plot_parameter_scatter(wide_df, df, 'prc', args.fig_out_dir),
        _plot_parameter_scatter(wide_df, df, 'obs', args.fig_out_dir),
        _plot_log_zeta_scatter(wide_df, df, args.fig_out_dir),
    ]
    print('Generated figures:')
    for f in files:
        print(f)


if __name__ == '__main__':
    main()
