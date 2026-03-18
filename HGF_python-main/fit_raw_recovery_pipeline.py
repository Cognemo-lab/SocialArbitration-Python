import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from HGF.code_inversion.tapas_fitModel import tapas_fitModel
from HGF.code_inversion.tapas_quasinewton_optim_config import tapas_quasinewton_optim_config
from HGF.code_model_wagad.prc_model.hgf_binary3l_freekappa_reward_social import hgf_binary3l_freekappa_reward_social
from HGF.code_model_wagad.prc_model.hgf_binary3l_freekappa_reward_social_config import hgf_binary3l_freekappa_reward_social_config
from HGF.code_model_wagad.obs_model.linear_volatilitydecnoise_1stlevelprecision_reward_social_config import (
    linear_volatilitydecnoise_1stlevelprecision_reward_social_config,
)
from HGF.code_model_wagad.obs_model.linear_volatilitydecnoise_1stlevelprecision_reward_social_sim import (
    linear_volatilitydecnoise_1stlevelprecision_reward_social_sim,
)


PRC_ORDER = [
    'mu2r_0', 'sa2r_0', 'mu3r_0', 'sa3r_0', 'ka_r', 'om_r', 'th_r',
    'mu2a_0', 'sa2a_0', 'mu3a_0', 'sa3a_0', 'ka_a', 'om_a', 'th_a',
    'phi_r', 'm_r', 'phi_a', 'm_a',
]
OBS_ORDER = ['be0', 'be1', 'be2', 'be3', 'be4', 'be5', 'be6', 'ze', 'be_ch', 'be_wager']


@dataclass
class FitRow:
    prolific_id: str
    round_name: str
    n_trials: int
    LME_obs: float
    AIC_obs: float
    BIC_obs: float
    LME_sim: float
    AIC_sim: float
    BIC_sim: float
    prc_corr: float
    obs_corr: float
    all_corr: float
    prc_rmse: float
    obs_rmse: float
    all_rmse: float
    obs_advice_taken_mean: float
    sim_advice_taken_mean: float
    obs_wager_mean: float
    sim_wager_mean: float


def _to_side(v):
    if pd.isna(v):
        return None
    if isinstance(v, str):
        vv = v.strip().lower()
        if vv == 'l':
            return 0
        if vv == 'r':
            return 1
        return None
    try:
        iv = int(float(v))
        if iv in (0, 1):
            return iv
    except Exception:
        pass
    return None


def _is_practice_true(v):
    if pd.isna(v):
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {'true', '1', 'yes'}
    try:
        return int(v) == 1
    except Exception:
        return False


def _extract_trials_from_raw(raw_df: pd.DataFrame, round_name: str) -> pd.DataFrame:
    raw_df = raw_df.copy()
    raw_df = raw_df.sort_values(['PROLIFIC_PID', 'trial_index', 'time_elapsed']).reset_index(drop=True)

    out = []
    for pid, g in raw_df.groupby('PROLIFIC_PID', sort=False):
        pending = None

        for _, row in g.iterrows():
            tt = row.get('trial_type', None)
            practice = _is_practice_true(row.get('practice', np.nan))

            advice_side = _to_side(row.get('advice', np.nan))
            outcome_side = _to_side(row.get('outcome', np.nan))

            # Choice trial row
            if tt == 'html-button-response' and (advice_side is not None) and (outcome_side is not None) and (not practice):
                response_side = _to_side(row.get('response', np.nan))
                if response_side is None:
                    pending = None
                    continue

                advice_correctness = 1.0 if advice_side == outcome_side else 0.0
                advice_taken = 1.0 if response_side == advice_side else 0.0

                pending = {
                    'prolific_id': str(pid),
                    'round_name': round_name,
                    'input_advice': advice_correctness,
                    'input_reward': float(outcome_side),
                    'advice_card_space': float(advice_side),
                    'choice_advice_taken': advice_taken,
                    'choice_side': float(response_side),
                    'trial_index_choice': row.get('trial_index', np.nan),
                }
                continue

            # Wager row (expected immediately after choice row)
            if pending is not None and tt == 'html-slider-response':
                try:
                    wager = float(row.get('response', np.nan))
                except Exception:
                    wager = np.nan

                if np.isfinite(wager):
                    rec = dict(pending)
                    rec['wager'] = wager
                    rec['trial_index_wager'] = row.get('trial_index', np.nan)
                    out.append(rec)

                pending = None

    out_df = pd.DataFrame(out)
    return out_df


def _load_and_extract(raw_paths: List[str]) -> pd.DataFrame:
    all_trials = []
    for p in raw_paths:
        df = pd.read_csv(p)
        if 'PROLIFIC_PID' not in df.columns:
            continue
        base = os.path.basename(p)
        round_name = 'round1' if 'round1' in base else ('round2' if 'round2' in base else os.path.splitext(base)[0])
        d = _extract_trials_from_raw(df, round_name)
        all_trials.append(d)

    if len(all_trials) == 0:
        return pd.DataFrame()

    out = pd.concat(all_trials, axis=0, ignore_index=True)
    out = out.sort_values(['round_name', 'prolific_id', 'trial_index_choice']).reset_index(drop=True)
    return out


def _fit_model(y: np.ndarray, u: np.ndarray, max_iter: int):
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


def _params_to_vec(est) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, float]]:
    p_prc = {k: float(est['p_prc'][k]) for k in PRC_ORDER}
    p_obs = {k: float(est['p_obs'][k]) for k in OBS_ORDER}
    prc_vec = np.array([p_prc[k] for k in PRC_ORDER], dtype=float)
    obs_vec = np.array([p_obs[k] for k in OBS_ORDER], dtype=float)
    return prc_vec, obs_vec, p_prc, p_obs


def _sim_from_params(u: np.ndarray, prc_vec: np.ndarray, obs_vec: np.ndarray, rng: np.random.RandomState):
    r = {'u': u, 'ign': []}
    traj, inf_states = hgf_binary3l_freekappa_reward_social(r, prc_vec)
    # Updated sim returns (y, prob) where y is [wager, choice] —
    # choice is already sampled internally using np.random.
    y_sim, _prob = linear_volatilitydecnoise_1stlevelprecision_reward_social_sim(r, inf_states, obs_vec)
    return y_sim


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return np.nan
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    parser = argparse.ArgumentParser(description='Fit finalized model to jungle raw data and run simulation-recovery correspondence.')
    parser.add_argument(
        '--raw-files',
        nargs='+',
        default=[
            '/Users/drea/Documents/CAMH/Projects/Social-Arbitration/jungle_task/raw_data/jungle-task.round1.csv',
            '/Users/drea/Documents/CAMH/Projects/Social-Arbitration/jungle_task/raw_data/jungle-task.round2.csv',
        ],
    )
    parser.add_argument('--out-dir', default='jungle_final_model_outputs')
    parser.add_argument('--max-subjects', type=int, default=0, help='0 means all subject-round datasets.')
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260316)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    trials = _load_and_extract(args.raw_files)
    if trials.empty:
        raise RuntimeError('No usable trials extracted from raw data.')

    trials_file = os.path.join(args.out_dir, 'extracted_model_trials.csv')
    trials.to_csv(trials_file, index=False)

    groups = list(trials.groupby(['prolific_id', 'round_name'], sort=False))
    if args.max_subjects > 0:
        groups = groups[: args.max_subjects]

    summary_rows: List[dict] = []
    correspondence_rows: List[dict] = []
    failures: List[dict] = []

    for i, ((pid, round_name), g) in enumerate(groups, start=1):
        g = g.sort_values('trial_index_choice').reset_index(drop=True)
        u = g[['input_advice', 'input_reward', 'advice_card_space']].to_numpy(dtype=float)
        y = g[['wager', 'choice_advice_taken']].to_numpy(dtype=float)

        n_trials = len(g)
        if n_trials < 40:
            failures.append({'prolific_id': pid, 'round_name': round_name, 'reason': f'too_few_trials:{n_trials}'})
            continue

        print(f'[{i}/{len(groups)}] Fitting {pid} {round_name} (n={n_trials})')

        try:
            est_obs = _fit_model(y, u, max_iter=args.max_iter)
            prc_fit_vec, obs_fit_vec, prc_fit, obs_fit = _params_to_vec(est_obs)

            y_sim = _sim_from_params(u, prc_fit_vec, obs_fit_vec, rng)
            est_sim = _fit_model(y_sim, u, max_iter=args.max_iter)
            prc_rec_vec, obs_rec_vec, prc_rec, obs_rec = _params_to_vec(est_sim)

            prc_corr = _corr(prc_fit_vec, prc_rec_vec)
            obs_corr = _corr(obs_fit_vec, obs_rec_vec)
            all_fit = np.concatenate([prc_fit_vec, obs_fit_vec])
            all_rec = np.concatenate([prc_rec_vec, obs_rec_vec])
            all_corr = _corr(all_fit, all_rec)

            prc_rmse = _rmse(prc_fit_vec, prc_rec_vec)
            obs_rmse = _rmse(obs_fit_vec, obs_rec_vec)
            all_rmse = _rmse(all_fit, all_rec)

            obs_advice_taken_mean = float(np.nanmean(y[:, 1]))
            sim_advice_taken_mean = float(np.nanmean(y_sim[:, 1]))
            obs_wager_mean = float(np.nanmean(y[:, 0]))
            sim_wager_mean = float(np.nanmean(y_sim[:, 0]))

            summary = FitRow(
                prolific_id=str(pid),
                round_name=str(round_name),
                n_trials=int(n_trials),
                LME_obs=float(est_obs['optim']['LME']),
                AIC_obs=float(est_obs['optim']['AIC']),
                BIC_obs=float(est_obs['optim']['BIC']),
                LME_sim=float(est_sim['optim']['LME']),
                AIC_sim=float(est_sim['optim']['AIC']),
                BIC_sim=float(est_sim['optim']['BIC']),
                prc_corr=prc_corr,
                obs_corr=obs_corr,
                all_corr=all_corr,
                prc_rmse=prc_rmse,
                obs_rmse=obs_rmse,
                all_rmse=all_rmse,
                obs_advice_taken_mean=obs_advice_taken_mean,
                sim_advice_taken_mean=sim_advice_taken_mean,
                obs_wager_mean=obs_wager_mean,
                sim_wager_mean=sim_wager_mean,
            )
            summary_rows.append(summary.__dict__)

            for name in PRC_ORDER:
                correspondence_rows.append({
                    'prolific_id': str(pid), 'round_name': str(round_name), 'group': 'prc', 'parameter': name,
                    'fitted_on_raw': prc_fit[name], 'sim_generating': prc_fit[name], 'recovered_from_sim': prc_rec[name],
                    'recovery_error': prc_rec[name] - prc_fit[name],
                })
            for name in OBS_ORDER:
                correspondence_rows.append({
                    'prolific_id': str(pid), 'round_name': str(round_name), 'group': 'obs', 'parameter': name,
                    'fitted_on_raw': obs_fit[name], 'sim_generating': obs_fit[name], 'recovered_from_sim': obs_rec[name],
                    'recovery_error': obs_rec[name] - obs_fit[name],
                })

        except Exception as e:
            failures.append({'prolific_id': str(pid), 'round_name': str(round_name), 'reason': str(e)})
            print(f'  FAILED {pid} {round_name}: {e}')

    summary_df = pd.DataFrame(summary_rows)
    corr_df = pd.DataFrame(correspondence_rows)
    fail_df = pd.DataFrame(failures)

    summary_csv = os.path.join(args.out_dir, 'fit_recovery_summary.csv')
    correspondence_csv = os.path.join(args.out_dir, 'parameter_correspondence.csv')
    failures_csv = os.path.join(args.out_dir, 'fit_failures.csv')

    summary_df.to_csv(summary_csv, index=False)
    corr_df.to_csv(correspondence_csv, index=False)
    fail_df.to_csv(failures_csv, index=False)

    aggregate = {
        'n_subject_round_total': int(len(groups)),
        'n_success': int(len(summary_df)),
        'n_failed': int(len(fail_df)),
        'mean_prc_corr': float(summary_df['prc_corr'].mean()) if len(summary_df) else np.nan,
        'mean_obs_corr': float(summary_df['obs_corr'].mean()) if len(summary_df) else np.nan,
        'mean_all_corr': float(summary_df['all_corr'].mean()) if len(summary_df) else np.nan,
        'mean_prc_rmse': float(summary_df['prc_rmse'].mean()) if len(summary_df) else np.nan,
        'mean_obs_rmse': float(summary_df['obs_rmse'].mean()) if len(summary_df) else np.nan,
        'mean_all_rmse': float(summary_df['all_rmse'].mean()) if len(summary_df) else np.nan,
    }

    agg_json = os.path.join(args.out_dir, 'aggregate_metrics.json')
    with open(agg_json, 'w') as f:
        json.dump(aggregate, f, indent=2)

    print(f'Saved extracted trials: {trials_file}')
    print(f'Saved summary: {summary_csv}')
    print(f'Saved parameter correspondence: {correspondence_csv}')
    print(f'Saved failures: {failures_csv}')
    print(f'Saved aggregate metrics: {agg_json}')
    print('Aggregate:', json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
