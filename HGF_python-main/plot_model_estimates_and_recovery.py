import argparse
import os

import numpy as np
import pandas as pd


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _plot_fit_quality(summary_df, out_dir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

    ax[0].hist(summary_df['LME_obs'], bins=20, alpha=0.8, color='tab:blue')
    ax[0].set_title('LME (Observed Fits)')
    ax[0].set_xlabel('LME_obs')
    ax[0].set_ylabel('count')

    ax[1].scatter(summary_df['LME_obs'], summary_df['LME_sim'], alpha=0.75, color='tab:green')
    lo = min(summary_df['LME_obs'].min(), summary_df['LME_sim'].min())
    hi = max(summary_df['LME_obs'].max(), summary_df['LME_sim'].max())
    ax[1].plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax[1].set_title('LME: Observed vs Sim-Recovered')
    ax[1].set_xlabel('LME_obs')
    ax[1].set_ylabel('LME_sim')

    ax[2].scatter(summary_df['obs_advice_taken_mean'], summary_df['sim_advice_taken_mean'], alpha=0.75, label='Advice taking', color='tab:orange')
    ax[2].scatter(summary_df['obs_wager_mean'], summary_df['sim_wager_mean'], alpha=0.75, label='Wager mean', color='tab:red')
    ax[2].plot([0, 1], [0, 1], 'k--', lw=1)
    ax[2].set_title('Behavioral Correspondence')
    ax[2].set_xlabel('Observed')
    ax[2].set_ylabel('Simulated')
    ax[2].legend(loc='best')

    plt.tight_layout()
    out = os.path.join(out_dir, 'figure_fit_quality_and_behavior.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out


def _boxplot_params(param_df, group_name, out_dir):
    import matplotlib.pyplot as plt

    d = param_df[param_df['group'] == group_name].copy()
    if d.empty:
        return None

    order = sorted(d['parameter'].unique().tolist())
    data = [d.loc[d['parameter'] == p, 'fitted_on_raw'].to_numpy(dtype=float) for p in order]

    fig, ax = plt.subplots(1, 1, figsize=(max(10, 0.65 * len(order)), 5))
    ax.boxplot(data, labels=order, showfliers=False)
    ax.set_title(f'Fitted Parameter Estimates ({group_name.upper()})')
    ax.set_ylabel('fitted_on_raw')
    ax.tick_params(axis='x', rotation=60)
    ax.grid(axis='y', alpha=0.2)
    plt.tight_layout()

    out = os.path.join(out_dir, f'figure_parameter_estimates_{group_name}.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out


def _recovery_scatter(param_df, group_name, out_dir):
    import matplotlib.pyplot as plt

    d = param_df[param_df['group'] == group_name].copy()
    if d.empty:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    params = sorted(d['parameter'].unique().tolist())
    cmap = plt.get_cmap('tab20')

    for i, p in enumerate(params):
        dd = d[d['parameter'] == p]
        ax.scatter(dd['sim_generating'], dd['recovered_from_sim'], s=35, alpha=0.75, label=p, color=cmap(i % 20))

    lo = min(d['sim_generating'].min(), d['recovered_from_sim'].min())
    hi = max(d['sim_generating'].max(), d['recovered_from_sim'].max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax.set_title(f'Parameter Recovery Scatter ({group_name.upper()})')
    ax.set_xlabel('Sim-generating (fitted_on_raw)')
    ax.set_ylabel('Recovered_from_sim')
    ax.grid(alpha=0.2)

    if len(params) <= 12:
        ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    out = os.path.join(out_dir, f'figure_parameter_recovery_scatter_{group_name}.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out


def _recovery_error_bars(param_df, out_dir):
    import matplotlib.pyplot as plt

    d = param_df.copy()
    d['abs_error'] = np.abs(d['recovery_error'])
    agg = d.groupby(['group', 'parameter'])['abs_error'].mean().reset_index()
    agg = agg.rename(columns={'abs_error': 'mean_abs_error'})

    fig, ax = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
    outs = []
    for i, grp in enumerate(['prc', 'obs']):
        g = agg[agg['group'] == grp].sort_values('mean_abs_error', ascending=False)
        ax[i].bar(g['parameter'], g['mean_abs_error'], color='tab:purple' if grp == 'prc' else 'tab:brown')
        ax[i].set_title(f'Mean Absolute Recovery Error ({grp.upper()})')
        ax[i].set_ylabel('mean |recovery_error|')
        ax[i].tick_params(axis='x', rotation=65)
        ax[i].grid(axis='y', alpha=0.2)

    plt.tight_layout()
    out = os.path.join(out_dir, 'figure_parameter_recovery_error.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description='Generate figures for model estimates and parameter recovery.')
    parser.add_argument('--pipeline-out-dir', default='jungle_final_model_outputs_smoke')
    parser.add_argument('--fig-out-dir', default='jungle_final_model_outputs_smoke/figures')
    args = parser.parse_args()

    summary_csv = os.path.join(args.pipeline_out_dir, 'fit_recovery_summary.csv')
    param_csv = os.path.join(args.pipeline_out_dir, 'parameter_correspondence.csv')

    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f'Missing summary CSV: {summary_csv}')
    if not os.path.exists(param_csv):
        raise FileNotFoundError(f'Missing parameter correspondence CSV: {param_csv}')

    _ensure_dir(args.fig_out_dir)

    summary_df = pd.read_csv(summary_csv)
    param_df = pd.read_csv(param_csv)

    files = []
    files.append(_plot_fit_quality(summary_df, args.fig_out_dir))
    files.append(_boxplot_params(param_df, 'prc', args.fig_out_dir))
    files.append(_boxplot_params(param_df, 'obs', args.fig_out_dir))
    files.append(_recovery_scatter(param_df, 'prc', args.fig_out_dir))
    files.append(_recovery_scatter(param_df, 'obs', args.fig_out_dir))
    files.append(_recovery_error_bars(param_df, args.fig_out_dir))

    files = [f for f in files if f is not None]
    print('Generated figures:')
    for f in files:
        print(f)


if __name__ == '__main__':
    main()
