import argparse
import contextlib
import glob
import json
import os
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from fit_raw_recovery_pipeline import (
    OBS_ORDER,
    PRC_ORDER,
    _corr,
    _fit_model,
    _load_and_extract,
    _rmse,
    _sim_from_params,
)


def _load_fit_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def _task_from_fit(path: str, trials: pd.DataFrame, max_iter: int, seed: int) -> dict:
    fit = _load_fit_json(path)
    pid = fit['prolific_id']
    round_name = fit['round_name']
    g = trials[(trials['prolific_id'] == pid) & (trials['round_name'] == round_name)].copy()
    if g.empty:
        raise ValueError(f'No extracted trials found for {pid} {round_name}')
    g = g.sort_values('trial_index_choice').reset_index(drop=True)
    return {
        'fit_path': path,
        'pid': pid,
        'round_name': round_name,
        'u': g[['input_advice', 'input_reward', 'advice_card_space']].to_numpy(dtype=float),
        'y_obs': g[['wager', 'choice_advice_taken']].to_numpy(dtype=float),
        'prc_fit_vec': np.array([fit['p_prc'][k] for k in PRC_ORDER], dtype=float),
        'obs_fit_vec': np.array([fit['p_obs'][k] for k in OBS_ORDER], dtype=float),
        'max_iter': max_iter,
        'seed': seed,
    }


def _recover_one(task: dict) -> dict:
    pid = task['pid']
    round_name = task['round_name']
    u = task['u']
    y_obs = task['y_obs']
    prc_fit_vec = task['prc_fit_vec']
    obs_fit_vec = task['obs_fit_vec']

    np.random.seed(task['seed'])
    y_sim = _sim_from_params(u, prc_fit_vec, obs_fit_vec, np.random.RandomState(task['seed']))

    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
        est_sim = _fit_model(y_sim, u, max_iter=task['max_iter'])

    prc_rec = {k: float(est_sim['p_prc'][k]) for k in PRC_ORDER}
    obs_rec = {k: float(est_sim['p_obs'][k]) for k in OBS_ORDER}
    prc_rec_vec = np.array([prc_rec[k] for k in PRC_ORDER], dtype=float)
    obs_rec_vec = np.array([obs_rec[k] for k in OBS_ORDER], dtype=float)

    all_fit = np.concatenate([prc_fit_vec, obs_fit_vec])
    all_rec = np.concatenate([prc_rec_vec, obs_rec_vec])

    summary = {
        'prolific_id': pid,
        'round_name': round_name,
        'n_trials': int(len(u)),
        'LME_sim': float(est_sim['optim']['LME']),
        'AIC_sim': float(est_sim['optim']['AIC']),
        'BIC_sim': float(est_sim['optim']['BIC']),
        'prc_corr': _corr(prc_fit_vec, prc_rec_vec),
        'obs_corr': _corr(obs_fit_vec, obs_rec_vec),
        'all_corr': _corr(all_fit, all_rec),
        'prc_rmse': _rmse(prc_fit_vec, prc_rec_vec),
        'obs_rmse': _rmse(obs_fit_vec, obs_rec_vec),
        'all_rmse': _rmse(all_fit, all_rec),
        'obs_advice_taken_mean': float(np.nanmean(y_obs[:, 1])),
        'sim_advice_taken_mean': float(np.nanmean(y_sim[:, 1])),
        'obs_wager_mean': float(np.nanmean(y_obs[:, 0])),
        'sim_wager_mean': float(np.nanmean(y_sim[:, 0])),
    }

    correspondence_rows: List[dict] = []
    for i, name in enumerate(PRC_ORDER):
        correspondence_rows.append({
            'prolific_id': pid,
            'round_name': round_name,
            'group': 'prc',
            'parameter': name,
            'is_fixed': False,
            'fitted_on_raw': float(prc_fit_vec[i]),
            'recovered_from_sim': float(prc_rec_vec[i]),
            'recovery_error': float(prc_rec_vec[i] - prc_fit_vec[i]),
        })
    for i, name in enumerate(OBS_ORDER):
        correspondence_rows.append({
            'prolific_id': pid,
            'round_name': round_name,
            'group': 'obs',
            'parameter': name,
            'is_fixed': False,
            'fitted_on_raw': float(obs_fit_vec[i]),
            'recovered_from_sim': float(obs_rec_vec[i]),
            'recovery_error': float(obs_rec_vec[i] - obs_fit_vec[i]),
        })

    return {'summary': summary, 'correspondence_rows': correspondence_rows}


def main():
    parser = argparse.ArgumentParser(description='Run parameter recovery from already saved fit JSONs.')
    parser.add_argument('--fit-json-dir', required=True)
    parser.add_argument(
        '--raw-files',
        nargs='+',
        default=[
            '/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/raw_data/jungle-task.round1.csv',
            '/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/raw_data/jungle-task.round2.csv',
        ],
    )
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--max-fits', type=int, default=10)
    parser.add_argument('--max-iter', type=int, default=5)
    parser.add_argument('--n-jobs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20260317)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    trials = _load_and_extract(args.raw_files)
    fit_paths = sorted(glob.glob(os.path.join(args.fit_json_dir, 'fit_*.json')))[:args.max_fits]
    if not fit_paths:
        raise RuntimeError('No fit JSONs found.')

    tasks = [
        _task_from_fit(path, trials, max_iter=args.max_iter, seed=args.seed + i)
        for i, path in enumerate(fit_paths)
    ]

    summary_rows: List[dict] = []
    correspondence_rows: List[dict] = []
    with Pool(processes=args.n_jobs) as pool:
        for i, result in enumerate(pool.imap_unordered(_recover_one, tasks), start=1):
            summary_rows.append(result['summary'])
            correspondence_rows.extend(result['correspondence_rows'])
            print(f'[{i}/{len(tasks)}] OK {result["summary"]["prolific_id"]} {result["summary"]["round_name"]} all_corr={result["summary"]["all_corr"]:.3f}')

    summary_df = pd.DataFrame(summary_rows).sort_values(['round_name', 'prolific_id']).reset_index(drop=True)
    correspondence_df = pd.DataFrame(correspondence_rows).sort_values(['group', 'parameter', 'prolific_id']).reset_index(drop=True)

    summary_csv = os.path.join(args.out_dir, 'fit_recovery_summary.csv')
    correspondence_csv = os.path.join(args.out_dir, 'parameter_correspondence.csv')
    summary_df.to_csv(summary_csv, index=False)
    correspondence_df.to_csv(correspondence_csv, index=False)

    pearsons = []
    for (group, parameter), d in correspondence_df.groupby(['group', 'parameter']):
        x = d['fitted_on_raw'].to_numpy(dtype=float)
        y = d['recovered_from_sim'].to_numpy(dtype=float)
        pearsons.append({
            'group': group,
            'parameter': parameter,
            'is_fixed': False,
            'n': int(len(d)),
            'mean_generating': float(np.mean(x)),
            'mean_recovered': float(np.mean(y)),
            'pearson_r': _corr(x, y),
            'rmse': _rmse(x, y),
            'mae': float(np.mean(np.abs(y - x))),
            'mean_abs_error': float(np.mean(np.abs(y - x))),
            'mean_signed_error': float(np.mean(y - x)),
        })
    param_report = pd.DataFrame(pearsons).sort_values(['group', 'parameter']).reset_index(drop=True)
    param_report_csv = os.path.join(args.out_dir, 'parameter_recovery_report.csv')
    param_report.to_csv(param_report_csv, index=False)

    aggregate = {
        'n_fit_sets': int(len(summary_df)),
        'mean_prc_corr': float(summary_df['prc_corr'].mean()),
        'mean_obs_corr': float(summary_df['obs_corr'].mean()),
        'mean_all_corr': float(summary_df['all_corr'].mean()),
        'mean_prc_rmse': float(summary_df['prc_rmse'].mean()),
        'mean_obs_rmse': float(summary_df['obs_rmse'].mean()),
        'mean_all_rmse': float(summary_df['all_rmse'].mean()),
    }
    with open(os.path.join(args.out_dir, 'aggregate_metrics.json'), 'w') as f:
        json.dump(aggregate, f, indent=2)

    print(f'Saved summary: {summary_csv}')
    print(f'Saved correspondence: {correspondence_csv}')
    print(f'Saved parameter report: {param_report_csv}')
    print('Aggregate:', json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
