import argparse
import json
import math
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from HGF.code_model_wagad.obs_model.linear_volatilitydecnoise_1stlevelprecision_reward_social_config import (
    linear_volatilitydecnoise_1stlevelprecision_reward_social_config,
)


def _generated_prc_params() -> Dict[str, float]:
    # Keep this synced with simulate_wagad_from_excel.py
    p = np.array([
        0, 1, 1, 1, 0.58458, -4, 0.59936,
        0, 1, 1, 1, 0.61243, -4, 0.66502,
        0.1000, 1, 0.1000, 1,
    ], dtype=float)

    names = [
        'mu2r_0', 'sa2r_0', 'mu3r_0', 'sa3r_0', 'ka_r', 'om_r', 'th_r',
        'mu2a_0', 'sa2a_0', 'mu3a_0', 'sa3a_0', 'ka_a', 'om_a', 'th_a',
        'phi_r', 'm_r', 'phi_a', 'm_a',
    ]
    return {k: float(v) for k, v in zip(names, p)}


def _generated_obs_params(obs_source: str) -> Dict[str, float]:
    if obs_source == 'legacy_fitted':
        p = np.array([6.6606, -0.5679, 0.5793, -0.3872, -0.1581, -1.7403, 0.0164, 2.0, 3.3213, 7.6193], dtype=float)
        names = ['be0', 'be1', 'be2', 'be3', 'be4', 'be5', 'be6', 'ze', 'be_ch', 'be_wager']
        return {k: float(v) for k, v in zip(names, p)}

    if obs_source == 'config_priors':
        cfg = linear_volatilitydecnoise_1stlevelprecision_reward_social_config()
        p, pstruct = cfg.transp_obs_fun(None, cfg.priormus)
        _ = p
        return {k: float(v) for k, v in pstruct.items()}

    raise ValueError("obs_source must be one of: 'legacy_fitted', 'config_priors'")


def _safe_pct_err(recovered: float, generated: float) -> float:
    denom = max(abs(generated), 1e-12)
    return 100.0 * (recovered - generated) / denom


def _collect_rows(group: str, generated: Dict[str, float], recovered: Dict[str, float]) -> List[Dict[str, float]]:
    rows = []
    for k, g in generated.items():
        r = recovered.get(k, np.nan)
        err = r - g if np.isfinite(r) else np.nan
        abs_err = abs(err) if np.isfinite(err) else np.nan
        pct_err = _safe_pct_err(r, g) if np.isfinite(r) else np.nan
        rows.append({
            'group': group,
            'parameter': k,
            'generated': g,
            'recovered': r,
            'error': err,
            'abs_error': abs_err,
            'pct_error': pct_err,
        })
    return rows


def _summary_metrics(df: pd.DataFrame) -> Dict[str, float]:
    m = {}
    for grp in ['prc', 'obs', 'all']:
        d = df if grp == 'all' else df[df['group'] == grp]
        d = d[np.isfinite(d['generated']) & np.isfinite(d['recovered'])]
        if len(d) == 0:
            m[f'{grp}_rmse'] = math.nan
            m[f'{grp}_mae'] = math.nan
            m[f'{grp}_corr'] = math.nan
            continue

        err = d['recovered'].to_numpy() - d['generated'].to_numpy()
        m[f'{grp}_rmse'] = float(np.sqrt(np.mean(err ** 2)))
        m[f'{grp}_mae'] = float(np.mean(np.abs(err)))
        if len(d) > 1:
            m[f'{grp}_corr'] = float(np.corrcoef(d['generated'], d['recovered'])[0, 1])
        else:
            m[f'{grp}_corr'] = math.nan
    return m


def _plot_recovery(df: pd.DataFrame, out_png: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not installed; skipping recovery plot.')
        return

    d = df[np.isfinite(df['generated']) & np.isfinite(df['recovered'])]
    if len(d) == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    for grp, color in [('prc', 'tab:blue'), ('obs', 'tab:orange')]:
        dd = d[d['group'] == grp]
        if len(dd) > 0:
            ax.scatter(dd['generated'], dd['recovered'], s=40, alpha=0.8, label=grp, color=color)

    lo = min(d['generated'].min(), d['recovered'].min())
    hi = max(d['generated'].max(), d['recovered'].max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax.set_xlabel('Generating parameter value')
    ax.set_ylabel('Recovered parameter value')
    ax.set_title('Parameter Recovery')
    ax.legend(loc='best')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Create parameter recovery report from fit_summary.json.')
    parser.add_argument('--fit-summary', default='fit_outputs/fit_summary.json')
    parser.add_argument('--obs-source', choices=['legacy_fitted', 'config_priors'], default='config_priors')
    parser.add_argument('--out-dir', default='fit_outputs')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.fit_summary, 'r') as f:
        fit = json.load(f)

    recovered_prc = fit.get('p_prc', {})
    recovered_obs = fit.get('p_obs', {})

    gen_prc = _generated_prc_params()
    gen_obs = _generated_obs_params(args.obs_source)

    rows = []
    rows += _collect_rows('prc', gen_prc, recovered_prc)
    rows += _collect_rows('obs', gen_obs, recovered_obs)
    report_df = pd.DataFrame(rows)

    metrics = _summary_metrics(report_df)
    metrics['n_trials'] = fit.get('n_trials', np.nan)
    metrics['LME'] = fit.get('LME', np.nan)
    metrics['AIC'] = fit.get('AIC', np.nan)
    metrics['BIC'] = fit.get('BIC', np.nan)
    metrics['obs_source'] = args.obs_source

    csv_file = os.path.join(args.out_dir, 'recovery_report.csv')
    json_file = os.path.join(args.out_dir, 'recovery_report_summary.json')
    png_file = os.path.join(args.out_dir, 'recovery_scatter.png')

    report_df.to_csv(csv_file, index=False)
    with open(json_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    _plot_recovery(report_df, png_file)

    print(f'Saved: {csv_file}')
    print(f'Saved: {json_file}')
    if os.path.exists(png_file):
        print(f'Saved: {png_file}')

    print('Recovery summary:')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
