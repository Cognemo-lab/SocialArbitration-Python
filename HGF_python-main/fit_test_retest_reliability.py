import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from HGF.code_inversion.tapas_fitModel import tapas_fitModel
from HGF.code_inversion.tapas_quasinewton_optim_config import tapas_quasinewton_optim_config
from HGF.code_model_wagad.prc_model.hgf_binary3l_freekappa_reward_social_config import hgf_binary3l_freekappa_reward_social_config
from HGF.code_model_wagad.obs_model.linear_volatilitydecnoise_1stlevelprecision_reward_social_config import (
    linear_volatilitydecnoise_1stlevelprecision_reward_social_config,
)

PRC_PARAMS = [
    'mu2r_0', 'sa2r_0', 'mu3r_0', 'sa3r_0', 'ka_r', 'om_r', 'th_r',
    'mu2a_0', 'sa2a_0', 'mu3a_0', 'sa3a_0', 'ka_a', 'om_a', 'th_a',
    'phi_r', 'm_r', 'phi_a', 'm_a',
]
OBS_PARAMS = ['be0', 'be1', 'be2', 'be3', 'be4', 'be5', 'be6', 'ze', 'be_ch', 'be_wager']
ALL_PARAMS = PRC_PARAMS + OBS_PARAMS


def _load_behavioral(per_trial_files: List[str]) -> pd.DataFrame:
    dfs = []
    for fp in per_trial_files:
        d = pd.read_csv(fp)
        base = os.path.basename(fp)
        if 'round1' in base:
            d['timepoint'] = 't1'
            d['round_name'] = 'round1'
        elif 'round2' in base:
            d['timepoint'] = 't2'
            d['round_name'] = 'round2'
        else:
            d['timepoint'] = base
            d['round_name'] = base
        d['trial_idx_in_file'] = np.arange(len(d))
        dfs.append(d)

    df = pd.concat(dfs, axis=0, ignore_index=True)

    needed = ['prolific_id', 'advice_correctness', 'outcome', 'advice', 'advice_taken', 'slider_response', 'timepoint']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns in behavioral files: {missing}')

    df = df.sort_values(['prolific_id', 'timepoint', 'trial_idx_in_file']).reset_index(drop=True)
    return df


def _fit_one(y: np.ndarray, u: np.ndarray, max_iter: int):
    class local_opt(tapas_quasinewton_optim_config):
        def __init__(self):
            super().__init__()
            self.maxIter = max_iter
            self.verbose = False
            self.nRandInit = 0

    est = tapas_fitModel(
        y,
        u,
        c_prc=hgf_binary3l_freekappa_reward_social_config,
        c_obs=linear_volatilitydecnoise_1stlevelprecision_reward_social_config,
        c_opt=local_opt,
    )
    return est


def _icc3_1(two_col: np.ndarray) -> float:
    # two_col: n x 2 (t1, t2)
    x = np.asarray(two_col, dtype=float)
    x = x[np.isfinite(x).all(axis=1)]
    n, k = x.shape
    if n < 3 or k != 2:
        return np.nan

    grand = np.mean(x)
    mean_row = np.mean(x, axis=1, keepdims=True)
    mean_col = np.mean(x, axis=0, keepdims=True)

    ss_row = k * np.sum((mean_row - grand) ** 2)
    ss_col = n * np.sum((mean_col - grand) ** 2)
    ss_tot = np.sum((x - grand) ** 2)
    ss_err = ss_tot - ss_row - ss_col

    ms_row = ss_row / (n - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))

    denom = ms_row + (k - 1) * ms_err
    if denom == 0:
        return np.nan
    return float((ms_row - ms_err) / denom)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]
    b = b[m]
    if len(a) < 3:
        return np.nan
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    parser = argparse.ArgumentParser(description='Fit model at T1/T2 and compute test-retest reliability for parameter estimates.')
    parser.add_argument(
        '--per-trial-files',
        nargs='+',
        default=[
            '/Users/drea/Documents/CAMH/Projects/Social-Arbitration/jungle_task/behavioral_metrics/jungle-task.round1.per_trial_behavioral_metrics.csv',
            '/Users/drea/Documents/CAMH/Projects/Social-Arbitration/jungle_task/behavioral_metrics/jungle-task.round2.per_trial_behavioral_metrics.csv',
        ],
    )
    parser.add_argument('--out-dir', default='jungle_test_retest_outputs')
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--save-subject-json', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = _load_behavioral(args.per_trial_files)

    fits_long: List[Dict] = []
    fit_rows: List[Dict] = []
    failures: List[Dict] = []

    groups = list(df.groupby(['prolific_id', 'timepoint'], sort=False))

    for i, ((pid, tp), g) in enumerate(groups, start=1):
        g = g.sort_values('trial_idx_in_file').reset_index(drop=True)

        y = g[['slider_response', 'advice_taken']].to_numpy(dtype=float)
        u = g[['advice_correctness', 'outcome', 'advice']].to_numpy(dtype=float)

        n_trials = len(g)
        print(f'[{i}/{len(groups)}] fitting {pid} {tp} n={n_trials}')

        try:
            est = _fit_one(y, u, max_iter=args.max_iter)
            p_prc = {k: float(est['p_prc'][k]) for k in PRC_PARAMS}
            p_obs = {k: float(est['p_obs'][k]) for k in OBS_PARAMS}

            fit_rows.append({
                'prolific_id': str(pid),
                'timepoint': str(tp),
                'n_trials': int(n_trials),
                'LME': float(est['optim']['LME']),
                'AIC': float(est['optim']['AIC']),
                'BIC': float(est['optim']['BIC']),
            })

            for p in PRC_PARAMS:
                fits_long.append({'prolific_id': str(pid), 'timepoint': str(tp), 'group': 'prc', 'parameter': p, 'estimate': p_prc[p]})
            for p in OBS_PARAMS:
                fits_long.append({'prolific_id': str(pid), 'timepoint': str(tp), 'group': 'obs', 'parameter': p, 'estimate': p_obs[p]})

            if args.save_subject_json:
                out_json = os.path.join(args.out_dir, f'est_{pid}_{tp}.json')
                with open(out_json, 'w') as f:
                    json.dump({
                        'prolific_id': str(pid),
                        'timepoint': str(tp),
                        'n_trials': int(n_trials),
                        'LME': float(est['optim']['LME']),
                        'AIC': float(est['optim']['AIC']),
                        'BIC': float(est['optim']['BIC']),
                        'p_prc': p_prc,
                        'p_obs': p_obs,
                    }, f, indent=2)

        except Exception as e:
            failures.append({'prolific_id': str(pid), 'timepoint': str(tp), 'reason': str(e)})
            print(f'  FAILED {pid} {tp}: {e}')

    fit_summary_df = pd.DataFrame(fit_rows)
    fits_long_df = pd.DataFrame(fits_long)
    failures_df = pd.DataFrame(failures)

    fit_summary_df.to_csv(os.path.join(args.out_dir, 'fit_summary_t1_t2.csv'), index=False)
    fits_long_df.to_csv(os.path.join(args.out_dir, 'parameter_estimates_long_t1_t2.csv'), index=False)
    failures_df.to_csv(os.path.join(args.out_dir, 'fit_failures_t1_t2.csv'), index=False)

    # Build wide table t1/t2 per subject for each parameter
    wide = fits_long_df.pivot_table(index=['prolific_id', 'parameter', 'group'], columns='timepoint', values='estimate', aggfunc='first').reset_index()
    wide.to_csv(os.path.join(args.out_dir, 'parameter_estimates_wide_t1_t2.csv'), index=False)

    # Reliability per parameter among subjects with both t1 and t2
    rel_rows = []
    for p in ALL_PARAMS:
        d = wide[wide['parameter'] == p].copy()
        if 't1' not in d.columns or 't2' not in d.columns:
            continue
        d = d[['prolific_id', 'parameter', 'group', 't1', 't2']].dropna()
        if d.empty:
            continue

        t1 = d['t1'].to_numpy(dtype=float)
        t2 = d['t2'].to_numpy(dtype=float)
        rel_rows.append({
            'parameter': p,
            'group': d['group'].iloc[0],
            'n_subjects_both_timepoints': int(len(d)),
            'pearson_r_t1_t2': _safe_corr(t1, t2),
            'icc3_1_t1_t2': _icc3_1(np.column_stack([t1, t2])),
            'mean_t1': float(np.mean(t1)),
            'mean_t2': float(np.mean(t2)),
            'sd_t1': float(np.std(t1, ddof=1)) if len(t1) > 1 else np.nan,
            'sd_t2': float(np.std(t2, ddof=1)) if len(t2) > 1 else np.nan,
            'mean_abs_delta': float(np.mean(np.abs(t2 - t1))),
        })

    rel_df = pd.DataFrame(rel_rows).sort_values(['group', 'parameter']) if len(rel_rows) else pd.DataFrame()
    rel_csv = os.path.join(args.out_dir, 'test_retest_reliability_by_parameter.csv')
    rel_df.to_csv(rel_csv, index=False)

    aggregate = {
        'n_subject_timepoint_fits_total': int(len(groups)),
        'n_successful_fits': int(len(fit_summary_df)),
        'n_failed_fits': int(len(failures_df)),
        'n_unique_subjects_any': int(df['prolific_id'].nunique()),
        'n_unique_subjects_t1': int(df.loc[df['timepoint'] == 't1', 'prolific_id'].nunique()),
        'n_unique_subjects_t2': int(df.loc[df['timepoint'] == 't2', 'prolific_id'].nunique()),
        'n_unique_subjects_with_both': int(len(set(df.loc[df['timepoint'] == 't1', 'prolific_id']) & set(df.loc[df['timepoint'] == 't2', 'prolific_id']))),
        'mean_pearson_r_prc': float(rel_df.loc[rel_df['group'] == 'prc', 'pearson_r_t1_t2'].mean()) if not rel_df.empty else np.nan,
        'mean_pearson_r_obs': float(rel_df.loc[rel_df['group'] == 'obs', 'pearson_r_t1_t2'].mean()) if not rel_df.empty else np.nan,
        'mean_icc3_1_prc': float(rel_df.loc[rel_df['group'] == 'prc', 'icc3_1_t1_t2'].mean()) if not rel_df.empty else np.nan,
        'mean_icc3_1_obs': float(rel_df.loc[rel_df['group'] == 'obs', 'icc3_1_t1_t2'].mean()) if not rel_df.empty else np.nan,
    }

    with open(os.path.join(args.out_dir, 'test_retest_reliability_aggregate.json'), 'w') as f:
        json.dump(aggregate, f, indent=2)

    print(f"Saved: {os.path.join(args.out_dir, 'fit_summary_t1_t2.csv')}")
    print(f"Saved: {os.path.join(args.out_dir, 'parameter_estimates_long_t1_t2.csv')}")
    print(f"Saved: {os.path.join(args.out_dir, 'parameter_estimates_wide_t1_t2.csv')}")
    print(f"Saved: {rel_csv}")
    print(f"Saved: {os.path.join(args.out_dir, 'test_retest_reliability_aggregate.json')}")
    print('Aggregate:', json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
