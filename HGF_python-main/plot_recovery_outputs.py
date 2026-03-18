import argparse
import os

import numpy as np
import pandas as pd


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _load_tables(recovery_out_dir: str):
    summary = pd.read_csv(os.path.join(recovery_out_dir, 'fit_recovery_summary.csv'))
    corr = pd.read_csv(os.path.join(recovery_out_dir, 'parameter_correspondence.csv'))
    report = pd.read_csv(os.path.join(recovery_out_dir, 'parameter_recovery_report.csv'))
    if 'is_fixed' not in corr.columns:
        corr['is_fixed'] = False
    if 'is_fixed' not in report.columns:
        report['is_fixed'] = False
    return summary, corr, report


def _plot_overall_correspondence(summary_df: pd.DataFrame, out_dir: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))

    ax[0].scatter(summary_df['LME_obs'], summary_df['LME_sim'], alpha=0.8, color='#2C7FB8')
    lo = min(summary_df['LME_obs'].min(), summary_df['LME_sim'].min())
    hi = max(summary_df['LME_obs'].max(), summary_df['LME_sim'].max())
    ax[0].plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax[0].set_title('Observed vs Recovered LME')
    ax[0].set_xlabel('LME_obs')
    ax[0].set_ylabel('LME_sim')

    ax[1].hist(summary_df['all_corr'], bins=8, color='#41AE76', alpha=0.85)
    ax[1].set_title('All-Parameter Correlation')
    ax[1].set_xlabel('all_corr (free params only)')
    ax[1].set_ylabel('count')

    ax[2].hist(summary_df['all_rmse'], bins=8, color='#DD6B20', alpha=0.85)
    ax[2].set_title('All-Parameter RMSE')
    ax[2].set_xlabel('all_rmse (free params only)')
    ax[2].set_ylabel('count')

    plt.tight_layout()
    out = os.path.join(out_dir, 'figure_recovery_overall.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out


def _plot_group_scatter(corr_df: pd.DataFrame, group: str, out_dir: str):
    import matplotlib.pyplot as plt

    d = corr_df[(corr_df['group'] == group) & (~corr_df['is_fixed'])].copy()
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
        x = dd['fitted_on_raw'].to_numpy(dtype=float)
        y = dd['recovered_from_sim'].to_numpy(dtype=float)
        ax.scatter(x, y, s=26, alpha=0.8, color='#2B6CB0')
        lo = min(np.min(x), np.min(y))
        hi = max(np.max(x), np.max(y))
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
        ax.set_title(param)
        ax.set_xlabel('Generating')
        ax.set_ylabel('Recovered')
        ax.grid(alpha=0.2)

    plt.tight_layout()
    out = os.path.join(out_dir, f'figure_recovery_scatter_{group}.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out


def _plot_log_zeta_recovery(corr_df: pd.DataFrame, out_dir: str):
    import matplotlib.pyplot as plt

    d = corr_df[
        (corr_df['group'] == 'obs')
        & (~corr_df['is_fixed'])
        & (corr_df['parameter'] == 'ze')
    ].copy()
    if d.empty:
        return None

    x = np.log(d['fitted_on_raw'].to_numpy(dtype=float))
    y = np.log(d['recovered_from_sim'].to_numpy(dtype=float))
    r = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))
    ax.scatter(x, y, s=28, alpha=0.75, color='#C05621')
    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax.set_title(f'log(zeta) Recovery\nr={r:.2f}')
    ax.set_xlabel('log generating zeta')
    ax.set_ylabel('log recovered zeta')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    out = os.path.join(out_dir, 'figure_recovery_scatter_log_ze.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out


def _plot_parameter_bars(report_df: pd.DataFrame, metric: str, out_dir: str):
    import matplotlib.pyplot as plt

    d = report_df[~report_df['is_fixed']].copy()
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for i, grp in enumerate(['prc', 'obs']):
        g = d[d['group'] == grp].sort_values(metric, ascending=(metric == 'pearson_r'))
        colors = '#2F855A' if grp == 'prc' else '#C05621'
        ax[i].bar(g['parameter'], g[metric], color=colors, alpha=0.85)
        ax[i].set_title(f'{grp.upper()} {metric}')
        ax[i].tick_params(axis='x', rotation=65)
        ax[i].grid(axis='y', alpha=0.2)
        if metric == 'pearson_r':
            ax[i].axhline(0.4, color='k', ls='--', lw=1)
    plt.tight_layout()
    out = os.path.join(out_dir, f'figure_parameter_{metric}.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description='Plot corrected recovery outputs.')
    parser.add_argument('--recovery-out-dir', required=True)
    parser.add_argument('--fig-out-dir', default='')
    args = parser.parse_args()

    fig_out_dir = args.fig_out_dir or os.path.join(args.recovery_out_dir, 'figures')
    _ensure_dir(fig_out_dir)

    summary_df, corr_df, report_df = _load_tables(args.recovery_out_dir)

    files = []
    files.append(_plot_overall_correspondence(summary_df, fig_out_dir))
    files.append(_plot_group_scatter(corr_df, 'prc', fig_out_dir))
    files.append(_plot_group_scatter(corr_df, 'obs', fig_out_dir))
    files.append(_plot_log_zeta_recovery(corr_df, fig_out_dir))
    files.append(_plot_parameter_bars(report_df, 'pearson_r', fig_out_dir))
    files.append(_plot_parameter_bars(report_df, 'mean_abs_error', fig_out_dir))
    files = [f for f in files if f is not None]

    print('Generated figures:')
    for f in files:
        print(f)


if __name__ == '__main__':
    main()
