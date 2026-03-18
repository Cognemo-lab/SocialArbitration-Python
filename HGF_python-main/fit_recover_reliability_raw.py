import argparse
import contextlib
import importlib
import json
import os
from multiprocessing import Pool
from typing import List

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
from fit_raw_recovery_pipeline import _corr, _extract_trials_from_raw, _load_and_extract, _rmse


PRC_ORDER = [
    'mu2r_0', 'sa2r_0', 'mu3r_0', 'sa3r_0', 'ka_r', 'om_r', 'th_r',
    'mu2a_0', 'sa2a_0', 'mu3a_0', 'sa3a_0', 'ka_a', 'om_a', 'th_a',
    'phi_r', 'm_r', 'phi_a', 'm_a',
]
PRC_PRIOR_ORDER = [
    'mu2r_0', 'sa2r_0', 'mu3r_0', 'sa3r_0', 'ka_r', 'om_r', 'th_r',
    'mu2a_0', 'sa2a_0', 'mu3a_0', 'sa3a_0', 'ka_a', 'om_a', 'th_a',
    'phi_r', 'phi_a', 'm_r', 'm_a',
]
OBS_ORDER = ['be0', 'be1', 'be2', 'be3', 'be4', 'be5', 'be6', 'ze', 'be_ch', 'be_wager']


def _resolve_class(path: str):
    module_name, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _free_mask(config_cls, names: List[str], prior_order: List[str] = None) -> np.ndarray:
    cfg = config_cls()
    priorsas = np.asarray(cfg.priorsas, dtype=float).reshape(-1)
    order = prior_order or names
    free_by_name = {name: bool(priorsas[i] != 0) for i, name in enumerate(order)}
    return np.array([free_by_name[name] for name in names], dtype=bool)


def _fit_model(y: np.ndarray, u: np.ndarray, max_iter: int, prc_cls, obs_cls):
    class local_opt(tapas_quasinewton_optim_config):
        def __init__(self):
            super().__init__()
            self.maxIter = max_iter
            self.verbose = False
            self.nRandInit = 0

    return tapas_fitModel(y, u, c_prc=prc_cls, c_obs=obs_cls, c_opt=local_opt)


def _icc3_1(two_col: np.ndarray) -> float:
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


def _simulate(u: np.ndarray, prc_vec: np.ndarray, obs_vec: np.ndarray, seed: int):
    np.random.seed(seed)
    r = {'u': u, 'ign': []}
    traj, inf_states = hgf_binary3l_freekappa_reward_social(r, prc_vec)
    y_sim, _ = linear_volatilitydecnoise_1stlevelprecision_reward_social_sim(r, inf_states, obs_vec)
    return y_sim


def _masked_corr(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    return _corr(x[mask], y[mask])


def _masked_rmse(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    return _rmse(x[mask], y[mask])


def _run_one(task: dict):
    try:
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

        p_prc = {k: float(est_obs['p_prc'][k]) for k in PRC_ORDER}
        p_obs = {k: float(est_obs['p_obs'][k]) for k in OBS_ORDER}
        prc_fit_vec = np.array([p_prc[k] for k in PRC_ORDER], dtype=float)
        obs_fit_vec = np.array([p_obs[k] for k in OBS_ORDER], dtype=float)

        y_sim = _simulate(u, prc_fit_vec, obs_fit_vec, task['seed'])

        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
            est_sim = _fit_model(y_sim, u, task['max_iter'], prc_cls, obs_cls)

        prc_rec = {k: float(est_sim['p_prc'][k]) for k in PRC_ORDER}
        obs_rec = {k: float(est_sim['p_obs'][k]) for k in OBS_ORDER}
        prc_rec_vec = np.array([prc_rec[k] for k in PRC_ORDER], dtype=float)
        obs_rec_vec = np.array([obs_rec[k] for k in OBS_ORDER], dtype=float)

        all_fit = np.concatenate([prc_fit_vec, obs_fit_vec])
        all_rec = np.concatenate([prc_rec_vec, obs_rec_vec])
        all_free_mask = np.concatenate([prc_free_mask, obs_free_mask])

        fit_payload = {
            'prolific_id': pid,
            'round_name': round_name,
            'timepoint': 't1' if round_name == 'round1' else ('t2' if round_name == 'round2' else round_name),
            'n_trials': int(len(u)),
            'LME': float(est_obs['optim']['LME']),
            'AIC': float(est_obs['optim']['AIC']),
            'BIC': float(est_obs['optim']['BIC']),
            'prc_config': task['prc_config'],
            'obs_config': task['obs_config'],
            'p_prc': p_prc,
            'p_obs': p_obs,
        }

        summary = {
            'prolific_id': pid,
            'round_name': round_name,
            'timepoint': fit_payload['timepoint'],
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
            'prc_corr': _masked_corr(prc_fit_vec, prc_rec_vec, prc_free_mask),
            'obs_corr': _masked_corr(obs_fit_vec, obs_rec_vec, obs_free_mask),
            'all_corr': _masked_corr(all_fit, all_rec, all_free_mask),
            'prc_rmse': _masked_rmse(prc_fit_vec, prc_rec_vec, prc_free_mask),
            'obs_rmse': _masked_rmse(obs_fit_vec, obs_rec_vec, obs_free_mask),
            'all_rmse': _masked_rmse(all_fit, all_rec, all_free_mask),
        }

        correspondence_rows = []
        estimate_rows = []
        for i, name in enumerate(PRC_ORDER):
            is_fixed = bool(not prc_free_mask[i])
            correspondence_rows.append({
                'prolific_id': pid,
                'round_name': round_name,
                'timepoint': fit_payload['timepoint'],
                'group': 'prc',
                'parameter': name,
                'is_fixed': is_fixed,
                'fitted_on_raw': float(prc_fit_vec[i]),
                'recovered_from_sim': float(prc_rec_vec[i]),
                'recovery_error': float(prc_rec_vec[i] - prc_fit_vec[i]),
            })
            estimate_rows.append({
                'prolific_id': pid,
                'round_name': round_name,
                'timepoint': fit_payload['timepoint'],
                'group': 'prc',
                'parameter': name,
                'is_fixed': is_fixed,
                'estimate': float(prc_fit_vec[i]),
            })
        for i, name in enumerate(OBS_ORDER):
            is_fixed = bool(not obs_free_mask[i])
            correspondence_rows.append({
                'prolific_id': pid,
                'round_name': round_name,
                'timepoint': fit_payload['timepoint'],
                'group': 'obs',
                'parameter': name,
                'is_fixed': is_fixed,
                'fitted_on_raw': float(obs_fit_vec[i]),
                'recovered_from_sim': float(obs_rec_vec[i]),
                'recovery_error': float(obs_rec_vec[i] - obs_fit_vec[i]),
            })
            estimate_rows.append({
                'prolific_id': pid,
                'round_name': round_name,
                'timepoint': fit_payload['timepoint'],
                'group': 'obs',
                'parameter': name,
                'is_fixed': is_fixed,
                'estimate': float(obs_fit_vec[i]),
            })

        return {
            'status': 'ok',
            'fit_payload': fit_payload,
            'summary': summary,
            'correspondence_rows': correspondence_rows,
            'estimate_rows': estimate_rows,
        }
    except Exception as e:
        return {
            'status': 'failed',
            'prolific_id': task['pid'],
            'round_name': task['round_name'],
            'reason': str(e),
        }


def main():
    parser = argparse.ArgumentParser(description='Fit, recover, and test-retest reliability on raw data.')
    parser.add_argument('--raw-files', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--max-subjects', type=int, default=0, help='0 means all subject-round datasets.')
    parser.add_argument('--max-iter', type=int, default=5)
    parser.add_argument('--n-jobs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20260321)
    parser.add_argument('--prc-config', required=True)
    parser.add_argument('--obs-config', required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fit_json_dir = os.path.join(args.out_dir, 'fit_jsons')
    os.makedirs(fit_json_dir, exist_ok=True)

    trials = _load_and_extract(args.raw_files)
    extracted_csv = os.path.join(args.out_dir, 'extracted_model_trials.csv')
    trials.to_csv(extracted_csv, index=False)

    prc_cls = _resolve_class(args.prc_config)
    obs_cls = _resolve_class(args.obs_config)
    prc_free_mask = _free_mask(prc_cls, PRC_ORDER, PRC_PRIOR_ORDER)
    obs_free_mask = _free_mask(obs_cls, OBS_ORDER)

    groups = list(trials.groupby(['prolific_id', 'round_name'], sort=False))
    if args.max_subjects > 0:
        groups = groups[:args.max_subjects]

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

    summary_rows = []
    correspondence_rows = []
    estimate_rows = []
    failures = []
    with Pool(processes=args.n_jobs) as pool:
        for i, result in enumerate(pool.imap_unordered(_run_one, tasks), start=1):
            if result['status'] == 'ok':
                summary_rows.append(result['summary'])
                correspondence_rows.extend(result['correspondence_rows'])
                estimate_rows.extend(result['estimate_rows'])
                fit_payload = result['fit_payload']
                out_json = os.path.join(fit_json_dir, f'fit_{fit_payload["prolific_id"]}_{fit_payload["round_name"]}.json')
                with open(out_json, 'w') as f:
                    json.dump(fit_payload, f, indent=2)
                print(f'[{i}/{len(tasks)}] OK {fit_payload["prolific_id"]} {fit_payload["round_name"]} all_corr={result["summary"]["all_corr"]:.3f}')
            else:
                failures.append({
                    'prolific_id': result['prolific_id'],
                    'round_name': result['round_name'],
                    'reason': result['reason'],
                })
                print(f'[{i}/{len(tasks)}] FAILED {result["prolific_id"]} {result["round_name"]}: {result["reason"]}')

    summary_df = pd.DataFrame(summary_rows).sort_values(['round_name', 'prolific_id']).reset_index(drop=True)
    corr_df = pd.DataFrame(correspondence_rows).sort_values(['group', 'parameter', 'prolific_id', 'round_name']).reset_index(drop=True)
    estimates_df = pd.DataFrame(estimate_rows).sort_values(['group', 'parameter', 'prolific_id', 'round_name']).reset_index(drop=True)
    failures_df = pd.DataFrame(failures)

    summary_df.to_csv(os.path.join(args.out_dir, 'fit_recovery_summary.csv'), index=False)
    corr_df.to_csv(os.path.join(args.out_dir, 'parameter_correspondence.csv'), index=False)
    estimates_df.to_csv(os.path.join(args.out_dir, 'parameter_estimates_long_t1_t2.csv'), index=False)
    failures_df.to_csv(os.path.join(args.out_dir, 'fit_failures.csv'), index=False)

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

    wide = estimates_df.pivot_table(
        index=['prolific_id', 'parameter', 'group', 'is_fixed'],
        columns='timepoint',
        values='estimate',
        aggfunc='first',
    ).reset_index()
    wide.to_csv(os.path.join(args.out_dir, 'parameter_estimates_wide_t1_t2.csv'), index=False)

    rel_rows = []
    for _, row in wide.iterrows():
        is_fixed = bool(row['is_fixed'])
        t1 = row.get('t1', np.nan)
        t2 = row.get('t2', np.nan)
        if pd.isna(t1) or pd.isna(t2):
            continue
        rel_rows.append({
            'parameter': row['parameter'],
            'group': row['group'],
            'is_fixed': is_fixed,
            'prolific_id': row['prolific_id'],
            't1': float(t1),
            't2': float(t2),
        })
    rel_long = pd.DataFrame(rel_rows)

    rel_param_rows = []
    if not rel_long.empty:
        for (group, parameter), d in rel_long.groupby(['group', 'parameter']):
            is_fixed = bool(d['is_fixed'].iloc[0])
            x = d['t1'].to_numpy(dtype=float)
            y = d['t2'].to_numpy(dtype=float)
            rel_param_rows.append({
                'parameter': parameter,
                'group': group,
                'is_fixed': is_fixed,
                'n_subjects_both_timepoints': int(len(d)),
                'pearson_r_t1_t2': np.nan if is_fixed else _corr(x, y),
                'icc3_1_t1_t2': np.nan if is_fixed else _icc3_1(np.column_stack([x, y])),
                'mean_t1': float(np.mean(x)),
                'mean_t2': float(np.mean(y)),
                'sd_t1': float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
                'sd_t2': float(np.std(y, ddof=1)) if len(y) > 1 else np.nan,
                'mean_abs_delta': np.nan if is_fixed else float(np.mean(np.abs(y - x))),
            })
    rel_df = pd.DataFrame(rel_param_rows)
    if not rel_df.empty:
        rel_df = rel_df.sort_values(['group', 'parameter']).reset_index(drop=True)
    rel_df.to_csv(os.path.join(args.out_dir, 'test_retest_reliability_by_parameter.csv'), index=False)

    aggregate = {
        'n_subject_round_total': int(len(groups)),
        'n_successful_fits': int(len(summary_df)),
        'n_failed_fits': int(len(failures_df)),
        'n_unique_subjects_any': int(estimates_df['prolific_id'].nunique()) if len(estimates_df) else 0,
        'n_unique_subjects_t1': int(estimates_df.loc[estimates_df['timepoint'] == 't1', 'prolific_id'].nunique()) if len(estimates_df) else 0,
        'n_unique_subjects_t2': int(estimates_df.loc[estimates_df['timepoint'] == 't2', 'prolific_id'].nunique()) if len(estimates_df) else 0,
        'n_unique_subjects_with_both': int(rel_long['prolific_id'].nunique()) if len(rel_long) else 0,
        'n_prc_free': int(np.sum(prc_free_mask)),
        'n_obs_free': int(np.sum(obs_free_mask)),
        'n_all_free': int(np.sum(prc_free_mask) + np.sum(obs_free_mask)),
        'mean_prc_recovery_corr': float(summary_df['prc_corr'].mean()) if len(summary_df) else np.nan,
        'mean_obs_recovery_corr': float(summary_df['obs_corr'].mean()) if len(summary_df) else np.nan,
        'mean_all_recovery_corr': float(summary_df['all_corr'].mean()) if len(summary_df) else np.nan,
        'mean_prc_recovery_rmse': float(summary_df['prc_rmse'].mean()) if len(summary_df) else np.nan,
        'mean_obs_recovery_rmse': float(summary_df['obs_rmse'].mean()) if len(summary_df) else np.nan,
        'mean_all_recovery_rmse': float(summary_df['all_rmse'].mean()) if len(summary_df) else np.nan,
        'mean_pearson_r_prc_t1_t2': float(rel_df.loc[(rel_df['group'] == 'prc') & (~rel_df['is_fixed']), 'pearson_r_t1_t2'].mean()) if len(rel_df) else np.nan,
        'mean_pearson_r_obs_t1_t2': float(rel_df.loc[(rel_df['group'] == 'obs') & (~rel_df['is_fixed']), 'pearson_r_t1_t2'].mean()) if len(rel_df) else np.nan,
        'mean_icc3_1_prc_t1_t2': float(rel_df.loc[(rel_df['group'] == 'prc') & (~rel_df['is_fixed']), 'icc3_1_t1_t2'].mean()) if len(rel_df) else np.nan,
        'mean_icc3_1_obs_t1_t2': float(rel_df.loc[(rel_df['group'] == 'obs') & (~rel_df['is_fixed']), 'icc3_1_t1_t2'].mean()) if len(rel_df) else np.nan,
        'prc_config': args.prc_config,
        'obs_config': args.obs_config,
    }
    with open(os.path.join(args.out_dir, 'aggregate_metrics.json'), 'w') as f:
        json.dump(aggregate, f, indent=2)

    print('Aggregate:', json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
