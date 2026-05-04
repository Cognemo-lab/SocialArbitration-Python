import argparse
import os

import numpy as np
import pandas as pd


OBS_LABELS = {
    'be0': 'Intercept',
    'be1': 'Surprise',
    'be2': 'Arbitration',
    'be3': 'Informational uncertainty - advice',
    'be4': 'Informational uncertainty - reward location',
    'be5': 'Volatility advice',
    'be6': 'Volatility reward location',
    'be_ch': 'Choice noise',
    'be_wager': 'Wager noise',
    'ze': 'Social bias',
}

PRC_PLOT_ORDER = ['ka_a', 'ka_r', 'm_a', 'om_a', 'sa3a_0', 'sa3r_0', 'th_a', 'th_r']

PRC_LABELS = {
    'ka_a': 'Kappa-Advice',
    'ka_r': 'Kappa-Reward',
    'm_a': 'Equilibrium-Advice',
    'om_a': 'Omega-Advice',
    'sa3a_0': 'Prior Uncertainty-Advice',
    'sa3r_0': 'Prior Uncertainty-Reward',
    'th_a': 'Theta-Advice',
    'th_r': 'Theta-Reward',
}


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


def _metric_scale(parameter: str, x: np.ndarray, y: np.ndarray):
    if parameter == 'ze':
        mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        x = np.log(x[mask])
        y = np.log(y[mask])
        return x, y, 'log'
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask], 'native'


def _display_label(parameter: str, metric_scale: str) -> str:
    if parameter == 'ze' and metric_scale == 'log':
        return 'Social bias'
    if parameter in PRC_LABELS:
        return PRC_LABELS[parameter]
    return OBS_LABELS.get(parameter, parameter)


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


def _plot_group_scatter(corr_df: pd.DataFrame, report_df: pd.DataFrame, group: str, out_dir: str):
    import matplotlib.pyplot as plt

    report_group = report_df[(report_df['group'] == group) & (~report_df['is_fixed'])].copy()
    if report_group.empty:
        return None

    if group == 'prc':
        params = [p for p in PRC_PLOT_ORDER if p in report_group['parameter'].tolist()]
    else:
        params = sorted(report_group['parameter'].unique().tolist())
    d = corr_df[
        (corr_df['group'] == group)
        & (~corr_df['is_fixed'])
        & (corr_df['parameter'].isin(params))
    ].copy()
    ncols = 4
    nrows = int(np.ceil(len(params) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax in axes.ravel():
        ax.set_visible(False)

    for i, param in enumerate(params):
        ax = axes.ravel()[i]
        ax.set_visible(True)
        dd = d[d['parameter'] == param]
        x_raw = dd['fitted_on_raw'].to_numpy(dtype=float)
        y_raw = dd['recovered_from_sim'].to_numpy(dtype=float)
        x, y, metric_scale = _metric_scale(param, x_raw, y_raw)
        ax.scatter(x, y, s=26, alpha=0.8, color='#2B6CB0')
        lo = min(np.min(x), np.min(y))
        hi = max(np.max(x), np.max(y))
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
        report_row = _get_report_row(param, group, report_df)
        title = _display_label(param, metric_scale)
        if report_row is not None:
            r = report_row.get('pearson_r', np.nan)
            icc = report_row.get('icc3_1', np.nan)
            stats_line = f'r={r:.2f}, ICC={icc:.2f}'
        else:
            stats_line = ''
        ax.set_title(title, fontsize=10.5, pad=26, wrap=True)
        if stats_line:
            ax.text(
                0.5,
                1.18,
                stats_line,
                transform=ax.transAxes,
                ha='center',
                va='bottom',
                fontsize=8.8,
                color='#4A5568',
                clip_on=False,
            )
        xlabel = 'Generating' if metric_scale == 'native' else 'Generating log(ze)'
        ylabel = 'Recovered' if metric_scale == 'native' else 'Recovered log(ze)'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)

    fig.tight_layout(pad=1.4, h_pad=3.2, w_pad=1.2)
    out = os.path.join(out_dir, f'figure_recovery_scatter_{group}.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out


def _plot_log_zeta_recovery(corr_df: pd.DataFrame, report_df: pd.DataFrame, out_dir: str):
    import matplotlib.pyplot as plt

    report_row = _get_report_row('ze', 'obs', report_df)
    if report_row is not None and bool(report_row.get('is_fixed', False)):
        return None

    d = corr_df[
        (corr_df['group'] == 'obs')
        & (~corr_df['is_fixed'])
        & (corr_df['parameter'] == 'ze')
    ].copy()
    if d.empty:
        return None

    x, y, _ = _metric_scale(
        'ze',
        d['fitted_on_raw'].to_numpy(dtype=float),
        d['recovered_from_sim'].to_numpy(dtype=float),
    )
    r = report_row.get('pearson_r', np.nan) if report_row is not None else np.nan
    icc = report_row.get('icc3_1', np.nan) if report_row is not None else np.nan

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))
    ax.scatter(x, y, s=28, alpha=0.75, color='#C05621')
    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax.set_title('Social bias', fontsize=12, pad=24)
    ax.text(
        0.5,
        1.14,
        f'r={r:.2f}, ICC={icc:.2f}',
        transform=ax.transAxes,
        ha='center',
        va='bottom',
        fontsize=10,
        color='#4A5568',
        clip_on=False,
    )
    ax.set_xlabel('log generating zeta')
    ax.set_ylabel('log recovered zeta')
    ax.grid(alpha=0.2)
    fig.tight_layout(pad=1.2)
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


def _get_report_row(parameter: str, group: str, report_df: pd.DataFrame):
    d = report_df[(report_df['group'] == group) & (report_df['parameter'] == parameter)]
    if d.empty:
        return None
    return d.iloc[0]


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
    files.append(_plot_group_scatter(corr_df, report_df, 'prc', fig_out_dir))
    files.append(_plot_group_scatter(corr_df, report_df, 'obs', fig_out_dir))
    files.append(_plot_log_zeta_recovery(corr_df, report_df, fig_out_dir))
    files.append(_plot_parameter_bars(report_df, 'pearson_r', fig_out_dir))
    if 'icc3_1' in report_df.columns:
        files.append(_plot_parameter_bars(report_df, 'icc3_1', fig_out_dir))
    files.append(_plot_parameter_bars(report_df, 'mean_abs_error', fig_out_dir))
    files = [f for f in files if f is not None]

    print('Generated figures:')
    for f in files:
        print(f)


if __name__ == '__main__':
    main()
