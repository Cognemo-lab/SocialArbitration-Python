import argparse
import contextlib
import json
import os
from dataclasses import dataclass
from multiprocessing import Pool
from typing import List

import numpy as np
import pandas as pd

from HGF.code_inversion.tapas_fitModel import tapas_fitModel
from HGF.code_inversion.tapas_quasinewton_optim_config import tapas_quasinewton_optim_config
from HGF.code_model_wagad.obs_model.linear_volatilitydecnoise_1stlevelprecision_reward_social_config import (
    linear_volatilitydecnoise_1stlevelprecision_reward_social_config,
)
from HGF.code_model_wagad.prc_model.hgf_binary3l_freekappa_reward_social_config import (
    hgf_binary3l_freekappa_reward_social_config,
)
from fit_raw_recovery_pipeline import _load_and_extract


PRC_ORDER = [
    'mu2r_0', 'sa2r_0', 'mu3r_0', 'sa3r_0', 'ka_r', 'om_r', 'th_r',
    'mu2a_0', 'sa2a_0', 'mu3a_0', 'sa3a_0', 'ka_a', 'om_a', 'th_a',
    'phi_r', 'm_r', 'phi_a', 'm_a',
]
OBS_ORDER = ['be0', 'be1', 'be2', 'be3', 'be4', 'be5', 'be6', 'ze', 'be_ch', 'be_wager']


@dataclass
class FitSummaryRow:
    prolific_id: str
    round_name: str
    n_trials: int
    n_valid_choice: int
    n_valid_wager: int
    LME: float
    AIC: float
    BIC: float


def _fit_model(y: np.ndarray, u: np.ndarray, max_iter: int):
    class local_opt(tapas_quasinewton_optim_config):
        def __init__(self):
            super().__init__()
            self.maxIter = max_iter
            self.verbose = False
            self.nRandInit = 0

    return tapas_fitModel(
        y,
        u,
        c_prc=hgf_binary3l_freekappa_reward_social_config,
        c_obs=linear_volatilitydecnoise_1stlevelprecision_reward_social_config,
        c_opt=local_opt,
    )


def _parameter_payload(est) -> dict:
    prc = {k: float(est['p_prc'][k]) for k in PRC_ORDER}
    obs = {k: float(est['p_obs'][k]) for k in OBS_ORDER}
    return {
        'LME': float(est['optim']['LME']),
        'AIC': float(est['optim']['AIC']),
        'BIC': float(est['optim']['BIC']),
        'perceptual_model': 'hgf_binary3l_freekappa_reward_social',
        'observation_model': 'linear_volatilitydecnoise_1stlevelprecision_reward_social',
        'p_prc': prc,
        'p_obs': obs,
    }


def _fit_one_group(task: dict) -> dict:
    pid = task['pid']
    round_name = task['round_name']
    g = task['group']
    max_iter = task['max_iter']
    min_trials = task['min_trials']

    g = g.sort_values('trial_index_choice').reset_index(drop=True)
    n_trials = len(g)
    if n_trials < min_trials:
        return {'status': 'failed', 'prolific_id': str(pid), 'round_name': str(round_name), 'reason': f'too_few_trials:{n_trials}'}

    u = g[['input_advice', 'input_reward', 'advice_card_space']].to_numpy(dtype=float)
    y = g[['wager', 'choice_advice_taken']].to_numpy(dtype=float)

    try:
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
            est = _fit_model(y, u, max_iter=max_iter)
        return {
            'status': 'ok',
            'summary': FitSummaryRow(
                prolific_id=str(pid),
                round_name=str(round_name),
                n_trials=int(n_trials),
                n_valid_choice=int(np.sum(np.isfinite(y[:, 1]))),
                n_valid_wager=int(np.sum(np.isfinite(y[:, 0]))),
                LME=float(est['optim']['LME']),
                AIC=float(est['optim']['AIC']),
                BIC=float(est['optim']['BIC']),
            ).__dict__,
            'payload': {
                'prolific_id': str(pid),
                'round_name': str(round_name),
                'n_trials': int(n_trials),
                **_parameter_payload(est),
            },
        }
    except Exception as e:
        return {'status': 'failed', 'prolific_id': str(pid), 'round_name': str(round_name), 'reason': str(e)}


def main():
    parser = argparse.ArgumentParser(
        description='Fit hgf_binary3l_freekappa_reward_social + '
        'linear_volatilitydecnoise_1stlevelprecision_reward_social to raw jungle-task data.'
    )
    parser.add_argument(
        '--raw-files',
        nargs='+',
        default=[
            '/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/raw_data/jungle-task.round1.csv',
            '/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/raw_data/jungle-task.round2.csv',
        ],
    )
    parser.add_argument('--out-dir', default='/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/hgf_fit_outputs')
    parser.add_argument('--max-subjects', type=int, default=0, help='0 means all subject-round datasets.')
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--min-trials', type=int, default=40)
    parser.add_argument('--n-jobs', type=int, default=8)
    parser.add_argument('--save-est-json', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    trials = _load_and_extract(args.raw_files)
    if trials.empty:
        raise RuntimeError('No usable trials extracted from raw data.')

    extracted_csv = os.path.join(args.out_dir, 'extracted_model_trials.csv')
    trials.to_csv(extracted_csv, index=False)

    groups = list(trials.groupby(['prolific_id', 'round_name'], sort=False))
    if args.max_subjects > 0:
        groups = groups[:args.max_subjects]

    summary_rows: List[dict] = []
    failures: List[dict] = []

    tasks = [{
        'pid': pid,
        'round_name': round_name,
        'group': g,
        'max_iter': args.max_iter,
        'min_trials': args.min_trials,
    } for (pid, round_name), g in groups]

    print(f'Fitting {len(tasks)} subject-round datasets with n_jobs={args.n_jobs}, max_iter={args.max_iter}')
    with Pool(processes=args.n_jobs) as pool:
        for i, result in enumerate(pool.imap_unordered(_fit_one_group, tasks), start=1):
            if result['status'] == 'ok':
                payload = result['payload']
                summary_rows.append(result['summary'])
                print(f'[{i}/{len(tasks)}] OK {payload["prolific_id"]} {payload["round_name"]} LME={payload["LME"]:.3f}')
                if args.save_est_json:
                    out_json = os.path.join(args.out_dir, f'fit_{payload["prolific_id"]}_{payload["round_name"]}.json')
                    with open(out_json, 'w') as f:
                        json.dump(payload, f, indent=2)
            else:
                failures.append({
                    'prolific_id': result['prolific_id'],
                    'round_name': result['round_name'],
                    'reason': result['reason'],
                })
                print(f'[{i}/{len(tasks)}] FAILED {result["prolific_id"]} {result["round_name"]}: {result["reason"]}')

    summary_df = pd.DataFrame(summary_rows)
    failures_df = pd.DataFrame(failures)

    summary_csv = os.path.join(args.out_dir, 'fit_summary.csv')
    failures_csv = os.path.join(args.out_dir, 'fit_failures.csv')
    summary_df.to_csv(summary_csv, index=False)
    failures_df.to_csv(failures_csv, index=False)

    aggregate = {
        'n_subject_round_total': int(len(groups)),
        'n_success': int(len(summary_df)),
        'n_failed': int(len(failures_df)),
        'mean_LME': float(summary_df['LME'].mean()) if len(summary_df) else np.nan,
        'mean_AIC': float(summary_df['AIC'].mean()) if len(summary_df) else np.nan,
        'mean_BIC': float(summary_df['BIC'].mean()) if len(summary_df) else np.nan,
    }
    aggregate_json = os.path.join(args.out_dir, 'aggregate_metrics.json')
    with open(aggregate_json, 'w') as f:
        json.dump(aggregate, f, indent=2)

    print(f'Saved extracted trials: {extracted_csv}')
    print(f'Saved summary: {summary_csv}')
    print(f'Saved failures: {failures_csv}')
    print(f'Saved aggregate metrics: {aggregate_json}')
    print('Aggregate:', json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
