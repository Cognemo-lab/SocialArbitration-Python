import argparse
import contextlib
import importlib
import glob
import json
import os
from multiprocessing import Pool
from typing import Dict, List

import numpy as np
import pandas as pd

from HGF.code_inversion.tapas_fitModel import tapas_fitModel
from HGF.code_inversion.tapas_quasinewton_optim_config import tapas_quasinewton_optim_config
from HGF.code_model_wagad.obs_model.linear_volatilitydecnoise_1stlevelprecision_reward_social_sim import (
    linear_volatilitydecnoise_1stlevelprecision_reward_social_sim,
)
from HGF.code_model_wagad.prc_model.hgf_binary3l_freekappa_reward_social import (
    hgf_binary3l_freekappa_reward_social,
)
from fit_raw_recovery_pipeline import OBS_ORDER, PRC_ORDER, _corr, _load_and_extract, _rmse

PRC_PRIOR_ORDER = [
    'mu2r_0', 'sa2r_0', 'mu3r_0', 'sa3r_0', 'ka_r', 'om_r', 'th_r',
    'mu2a_0', 'sa2a_0', 'mu3a_0', 'sa3a_0', 'ka_a', 'om_a', 'th_a',
    'phi_r', 'phi_a', 'm_r', 'm_a',
]


def _resolve_class(path: str):
    module_name, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _fit_model(y: np.ndarray, u: np.ndarray, max_iter: int, prc_cls, obs_cls):
    class local_opt(tapas_quasinewton_optim_config):
        def __init__(self):
            super().__init__()
            self.maxIter = max_iter
            self.verbose = False
            self.nRandInit = 0

    return tapas_fitModel(y, u, c_prc=prc_cls, c_obs=obs_cls, c_opt=local_opt)


def _params_to_vec(est) -> Dict[str, np.ndarray]:
    return {
        'prc_vec': np.array([float(est['p_prc'][k]) for k in PRC_ORDER], dtype=float),
        'obs_vec': np.array([float(est['p_obs'][k]) for k in OBS_ORDER], dtype=float),
    }


def _free_mask(config_cls, names: List[str], prior_order: List[str] = None) -> np.ndarray:
    cfg = config_cls()
    priorsas = np.asarray(cfg.priorsas, dtype=float).reshape(-1)
    order = prior_order or names
    if len(priorsas) != len(order):
        raise ValueError(f'Prior variance length mismatch for {config_cls.__name__}')
    free_by_name = {name: bool(priorsas[i] != 0) for i, name in enumerate(order)}
    return np.array([free_by_name[name] for name in names], dtype=bool)


def _masked_corr(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    return _corr(x[mask], y[mask])


def _masked_rmse(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    return _rmse(x[mask], y[mask])


def _simulate(u: np.ndarray, prc_vec: np.ndarray, obs_vec: np.ndarray, seed: int):
    np.random.seed(seed)
    r = {'u': u, 'ign': []}
    traj, inf_states = hgf_binary3l_freekappa_reward_social(r, prc_vec)
    y_sim, _ = linear_volatilitydecnoise_1stlevelprecision_reward_social_sim(r, inf_states, obs_vec)
    return y_sim


def _run_one(task: dict):
    prc_cls = _resolve_class(task['prc_config'])
    obs_cls = _resolve_class(task['obs_config'])
    pid = task['pid']
    round_name = task['round_name']
    u = task['u']
    y_obs = task['y_obs']
    prc_free_mask = task['prc_free_mask']
    obs_free_mask = task['obs_free_mask']

    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
        est_obs = _fit_model(y_obs, u, task['max_iter'], prc_cls, obs_cls)
    fit = _params_to_vec(est_obs)

    y_sim = _simulate(u, fit['prc_vec'], fit['obs_vec'], task['seed'])

    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
        est_sim = _fit_model(y_sim, u, task['max_iter'], prc_cls, obs_cls)
    rec = _params_to_vec(est_sim)

    all_fit = np.concatenate([fit['prc_vec'], fit['obs_vec']])
    all_rec = np.concatenate([rec['prc_vec'], rec['obs_vec']])
    all_free_mask = np.concatenate([prc_free_mask, obs_free_mask])

    summary = {
        'prolific_id': pid,
        'round_name': round_name,
        'n_trials': int(len(u)),
        'LME_obs': float(est_obs['optim']['LME']),
        'AIC_obs': float(est_obs['optim']['AIC']),
        'BIC_obs': float(est_obs['optim']['BIC']),
        'LME_sim': float(est_sim['optim']['LME']),
        'AIC_sim': float(est_sim['optim']['AIC']),
        'BIC_sim': float(est_sim['optim']['BIC']),
        'n_prc_free': int(np.sum(prc_free_mask)),
        'n_obs_free': int(np.sum(obs_free_mask)),
        'n_all_free': int(np.sum(all_free_mask)),
        'prc_corr': _masked_corr(fit['prc_vec'], rec['prc_vec'], prc_free_mask),
        'obs_corr': _masked_corr(fit['obs_vec'], rec['obs_vec'], obs_free_mask),
        'all_corr': _masked_corr(all_fit, all_rec, all_free_mask),
        'prc_rmse': _masked_rmse(fit['prc_vec'], rec['prc_vec'], prc_free_mask),
        'obs_rmse': _masked_rmse(fit['obs_vec'], rec['obs_vec'], obs_free_mask),
        'all_rmse': _masked_rmse(all_fit, all_rec, all_free_mask),
    }

    rows: List[dict] = []
    for i, name in enumerate(PRC_ORDER):
        rows.append({
            'prolific_id': pid,
            'round_name': round_name,
            'group': 'prc',
            'parameter': name,
            'is_fixed': bool(not prc_free_mask[i]),
            'fitted_on_raw': float(fit['prc_vec'][i]),
            'recovered_from_sim': float(rec['prc_vec'][i]),
            'recovery_error': float(rec['prc_vec'][i] - fit['prc_vec'][i]),
        })
    for i, name in enumerate(OBS_ORDER):
        rows.append({
            'prolific_id': pid,
            'round_name': round_name,
            'group': 'obs',
            'parameter': name,
            'is_fixed': bool(not obs_free_mask[i]),
            'fitted_on_raw': float(fit['obs_vec'][i]),
            'recovered_from_sim': float(rec['obs_vec'][i]),
            'recovery_error': float(rec['obs_vec'][i] - fit['obs_vec'][i]),
        })

    return {'summary': summary, 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description='Fit and run parameter recovery on a subset of raw datasets.')
    parser.add_argument('--raw-files', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--max-subjects', type=int, default=10)
    parser.add_argument('--max-iter', type=int, default=5)
    parser.add_argument('--n-jobs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20260317)
    parser.add_argument('--fit-json-dir', default='')
    parser.add_argument(
        '--prc-config',
        required=True,
        help='Import path to perceptual config class.',
    )
    parser.add_argument(
        '--obs-config',
        required=True,
        help='Import path to observation config class.',
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    trials = _load_and_extract(args.raw_files)
    prc_cls = _resolve_class(args.prc_config)
    obs_cls = _resolve_class(args.obs_config)
    prc_free_mask = _free_mask(prc_cls, PRC_ORDER, PRC_PRIOR_ORDER)
    obs_free_mask = _free_mask(obs_cls, OBS_ORDER)

    if args.fit_json_dir:
        selected = []
        for path in sorted(glob.glob(os.path.join(args.fit_json_dir, 'fit_*.json')))[:args.max_subjects]:
            base = os.path.basename(path)
            prefix = base[4:-5]
            pid, round_name = prefix.rsplit('_', 1)
            selected.append((pid, round_name))
        group_map = {k: g for k, g in trials.groupby(['prolific_id', 'round_name'], sort=False)}
        groups = [((pid, round_name), group_map[(pid, round_name)]) for pid, round_name in selected]
    else:
        groups = list(trials.groupby(['prolific_id', 'round_name'], sort=False))[:args.max_subjects]

    tasks = []
    for i, ((pid, round_name), g) in enumerate(groups):
        g = g.sort_values('trial_index_choice').reset_index(drop=True)
        tasks.append({
            'pid': str(pid),
            'round_name': str(round_name),
            'u': g[['input_advice', 'input_reward', 'advice_card_space']].to_numpy(dtype=float),
            'y_obs': g[['wager', 'choice_advice_taken']].to_numpy(dtype=float),
            'max_iter': args.max_iter,
            'seed': args.seed + i,
            'prc_config': args.prc_config,
            'obs_config': args.obs_config,
            'prc_free_mask': prc_free_mask,
            'obs_free_mask': obs_free_mask,
        })

    summary_rows: List[dict] = []
    correspondence_rows: List[dict] = []
    with Pool(processes=args.n_jobs) as pool:
        for i, result in enumerate(pool.imap_unordered(_run_one, tasks), start=1):
            summary_rows.append(result['summary'])
            correspondence_rows.extend(result['rows'])
            print(f'[{i}/{len(tasks)}] OK {result["summary"]["prolific_id"]} {result["summary"]["round_name"]} all_corr={result["summary"]["all_corr"]:.3f}')

    summary_df = pd.DataFrame(summary_rows).sort_values(['round_name', 'prolific_id']).reset_index(drop=True)
    corr_df = pd.DataFrame(correspondence_rows).sort_values(['group', 'parameter', 'prolific_id']).reset_index(drop=True)
    summary_df.to_csv(os.path.join(args.out_dir, 'fit_recovery_summary.csv'), index=False)
    corr_df.to_csv(os.path.join(args.out_dir, 'parameter_correspondence.csv'), index=False)

    report_rows = []
    for (group, parameter), d in corr_df.groupby(['group', 'parameter']):
        is_fixed = bool(d['is_fixed'].iloc[0])
        x = d['fitted_on_raw'].to_numpy(dtype=float)
        y = d['recovered_from_sim'].to_numpy(dtype=float)
        report_rows.append({
            'group': group,
            'parameter': parameter,
            'is_fixed': is_fixed,
            'n': int(len(d)),
            'pearson_r': np.nan if is_fixed else _corr(x, y),
            'rmse': np.nan if is_fixed else _rmse(x, y),
            'mae': np.nan if is_fixed else float(np.mean(np.abs(y - x))),
            'mean_abs_error': np.nan if is_fixed else float(np.mean(np.abs(y - x))),
            'mean_signed_error': np.nan if is_fixed else float(np.mean(y - x)),
        })
    report_df = pd.DataFrame(report_rows).sort_values(['group', 'parameter']).reset_index(drop=True)
    report_df.to_csv(os.path.join(args.out_dir, 'parameter_recovery_report.csv'), index=False)

    aggregate = {
        'n_fit_sets': int(len(summary_df)),
        'n_prc_free': int(np.sum(prc_free_mask)),
        'n_obs_free': int(np.sum(obs_free_mask)),
        'n_all_free': int(np.sum(prc_free_mask) + np.sum(obs_free_mask)),
        'mean_prc_corr': float(summary_df['prc_corr'].mean()),
        'mean_obs_corr': float(summary_df['obs_corr'].mean()),
        'mean_all_corr': float(summary_df['all_corr'].mean()),
        'mean_prc_rmse': float(summary_df['prc_rmse'].mean()),
        'mean_obs_rmse': float(summary_df['obs_rmse'].mean()),
        'mean_all_rmse': float(summary_df['all_rmse'].mean()),
        'prc_config': args.prc_config,
        'obs_config': args.obs_config,
    }
    with open(os.path.join(args.out_dir, 'aggregate_metrics.json'), 'w') as f:
        json.dump(aggregate, f, indent=2)

    print('Aggregate:', json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
