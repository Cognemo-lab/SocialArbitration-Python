import argparse
import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from HGF.code_inversion.tapas_fitModel import tapas_fitModel
from HGF.code_inversion.tapas_quasinewton_optim_config import tapas_quasinewton_optim_config
from HGF.code_model_wagad.prc_model.hgf_binary3l_freekappa_reward_social_config import hgf_binary3l_freekappa_reward_social_config
from HGF.code_model_wagad.obs_model.linear_volatilitydecnoise_1stlevelprecision_reward_social_config import (
    linear_volatilitydecnoise_1stlevelprecision_reward_social_config,
)


@dataclass
class SubjectResult:
    prolific_id: str
    round_name: str
    n_trials: int
    n_valid_choice: int
    n_valid_wager: int
    LME: float
    AIC: float
    BIC: float


def _load_per_trial(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        d = pd.read_csv(p)
        base = os.path.basename(p)
        if 'round1' in base:
            d['round_name'] = 'round1'
        elif 'round2' in base:
            d['round_name'] = 'round2'
        else:
            d['round_name'] = os.path.splitext(base)[0]
        d['trial_index_within_file'] = np.arange(len(d))
        dfs.append(d)
    out = pd.concat(dfs, axis=0, ignore_index=True)
    return out


def _build_model_arrays(sub_df: pd.DataFrame):
    # Model input mapping for WAGAD port:
    # u[:,0] advice correctness (helpful=1, misleading=0)
    # u[:,1] reward outcome side (right=1, left=0)
    # u[:,2] advice side (right=1, left=0)
    input_u = sub_df[['advice_correctness', 'outcome', 'advice']].to_numpy(dtype=float)

    # Response mapping (wager first, choice second — matches observation model):
    # y[:,0] wager as slider response (continuous)
    # y[:,1] choice as advice taking (followed advice=1, not followed=0) (binary)
    y = sub_df[['slider_response', 'advice_taken']].to_numpy(dtype=float)

    return y, input_u


def _fit_one_subject(sub_df: pd.DataFrame, max_iter: int):
    y, input_u = _build_model_arrays(sub_df)

    class local_opt(tapas_quasinewton_optim_config):
        def __init__(self):
            super().__init__()
            self.maxIter = max_iter
            self.verbose = False

    est = tapas_fitModel(
        y,
        input_u,
        c_prc=hgf_binary3l_freekappa_reward_social_config,
        c_obs=linear_volatilitydecnoise_1stlevelprecision_reward_social_config,
        c_opt=local_opt,
    )
    return est


def main():
    parser = argparse.ArgumentParser(description='Fit WAGAD model to jungle task behavioral data per participant.')
    parser.add_argument(
        '--per-trial-files',
        nargs='+',
        default=[
            '/Users/drea/Documents/CAMH/Projects/Social-Arbitration/jungle_task/behavioral_metrics/jungle-task.round1.per_trial_behavioral_metrics.csv',
            '/Users/drea/Documents/CAMH/Projects/Social-Arbitration/jungle_task/behavioral_metrics/jungle-task.round2.per_trial_behavioral_metrics.csv',
        ],
    )
    parser.add_argument('--out-dir', default='jungle_fit_outputs')
    parser.add_argument('--max-subjects', type=int, default=0, help='0 means all subjects.')
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--save-est-json', action='store_true', help='Save per-subject parameter JSON files.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = _load_per_trial(args.per_trial_files)
    needed_cols = ['prolific_id', 'advice_correctness', 'outcome', 'advice', 'advice_taken', 'slider_response', 'round_name']
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    # Keep trial ordering as provided in each source file
    df = df.sort_values(['round_name', 'prolific_id', 'trial_index_within_file']).reset_index(drop=True)

    groups = list(df.groupby(['prolific_id', 'round_name'], sort=False))
    if args.max_subjects > 0:
        groups = groups[: args.max_subjects]

    summary_rows = []

    for i, ((pid, round_name), sub_df) in enumerate(groups, start=1):
        print(f'[{i}/{len(groups)}] Fitting prolific_id={pid} round={round_name} n_trials={len(sub_df)}')
        est = _fit_one_subject(sub_df, max_iter=args.max_iter)

        row = SubjectResult(
            prolific_id=str(pid),
            round_name=str(round_name),
            n_trials=int(len(sub_df)),
            n_valid_choice=int(np.sum(np.isfinite(sub_df['advice_taken'].to_numpy(dtype=float)))),
            n_valid_wager=int(np.sum(np.isfinite(sub_df['slider_response'].to_numpy(dtype=float)))),
            LME=float(est['optim']['LME']),
            AIC=float(est['optim']['AIC']),
            BIC=float(est['optim']['BIC']),
        )
        summary_rows.append(row.__dict__)

        if args.save_est_json:
            out_json = os.path.join(args.out_dir, f'fit_{pid}_{round_name}.json')
            payload = {
                'prolific_id': str(pid),
                'round_name': str(round_name),
                'n_trials': int(len(sub_df)),
                'LME': float(est['optim']['LME']),
                'AIC': float(est['optim']['AIC']),
                'BIC': float(est['optim']['BIC']),
                'p_prc': {k: float(v) for k, v in est['p_prc'].items() if isinstance(v, (int, float, np.floating))},
                'p_obs': {k: float(v) for k, v in est['p_obs'].items() if isinstance(v, (int, float, np.floating))},
            }
            with open(out_json, 'w') as f:
                json.dump(payload, f, indent=2)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.out_dir, 'jungle_fit_summary.csv')
    summary_df.to_csv(summary_csv, index=False)

    print(f'Saved summary: {summary_csv}')
    if len(summary_df) > 0:
        print('Aggregate metrics:')
        print(summary_df[['LME', 'AIC', 'BIC']].describe().to_string())


if __name__ == '__main__':
    main()
