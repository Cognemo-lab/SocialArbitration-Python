import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from HGF.code_model_wagad.prc_model.hgf_binary3l_freekappa_reward_social import (
    hgf_binary3l_freekappa_reward_social,
)
from HGF.code_inversion.tapas_sgm import tapas_sgm


PRC_ORDER = [
    'mu2r_0', 'sa2r_0', 'mu3r_0', 'sa3r_0', 'ka_r', 'om_r', 'th_r',
    'mu2a_0', 'sa2a_0', 'mu3a_0', 'sa3a_0', 'ka_a', 'om_a', 'th_a',
    'phi_r', 'm_r', 'phi_a', 'm_a',
]


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


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]
    b = b[m]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _load_fit_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def _state_summaries(traj: Dict[str, np.ndarray], u: np.ndarray, p_obs: Dict[str, float], p_prc: Dict[str, float]) -> Dict[str, float]:
    out = {}
    eps3_r = 0.5 * p_prc['ka_r'] * traj['w_r'] * traj['da_r'][:, 1] / traj['sa_r'][:, 2]
    eps3_a = 0.5 * p_prc['ka_a'] * traj['w_a'] * traj['da_a'][:, 1] / traj['sa_a'][:, 2]

    metric_map = {
        'mu1hat_r_mean': traj['muhat_r'][:, 0],
        'mu1hat_a_mean': traj['muhat_a'][:, 0],
        'sa1hat_r_mean': traj['sahat_r'][:, 0],
        'sa1hat_a_mean': traj['sahat_a'][:, 0],
        'eps1_r_mean': traj['da_r'][:, 0],
        'eps1_a_mean': traj['da_a'][:, 0],
        'eps2_r_mean': traj['da_r'][:, 1],
        'eps2_a_mean': traj['da_a'][:, 1],
        'eps3_r_mean': eps3_r,
        'eps3_a_mean': eps3_a,
        'abs_eps1_r_mean': np.abs(traj['da_r'][:, 0]),
        'abs_eps1_a_mean': np.abs(traj['da_a'][:, 0]),
        'abs_eps2_r_mean': np.abs(traj['da_r'][:, 1]),
        'abs_eps2_a_mean': np.abs(traj['da_a'][:, 1]),
        'abs_eps3_r_mean': np.abs(eps3_r),
        'abs_eps3_a_mean': np.abs(eps3_a),
    }

    reward_location = u[:, 0]
    advice_helpfulness = u[:, 1]
    advice_in_loc_coords = 1.0 - np.abs(reward_location - advice_helpfulness)

    mu1hat_a = traj['muhat_a'][:, 0]
    mu1hat_r = traj['muhat_r'][:, 0]
    mu2hat_a = traj['muhat_a'][:, 1]
    mu2hat_r = traj['muhat_r'][:, 1]
    sa2hat_r = traj['sahat_r'][:, 1]
    sa2hat_a = traj['sahat_a'][:, 1]
    mu3hat_r = traj['muhat_r'][:, 2]
    mu3hat_a = traj['muhat_a'][:, 2]

    transformed_mu1hat_r = mu1hat_r ** advice_in_loc_coords * (1.0 - mu1hat_r) ** (1.0 - advice_in_loc_coords)
    px = 1.0 / (mu1hat_a * (1.0 - mu1hat_a))
    pc = 1.0 / (mu1hat_r * (1.0 - mu1hat_r))
    ze = p_obs['ze']
    be_ch = p_obs['be_ch']

    wx = ze * px / (ze * px + pc)
    wc = pc / (ze * px + pc)
    b = np.clip(wx * mu1hat_a + wc * transformed_mu1hat_r, 1e-8, 1.0 - 1e-8)
    decision_noise = np.exp(-mu3hat_r - mu3hat_a) / be_ch
    surp = -np.log2(b)
    inferv_a = tapas_sgm(mu2hat_a, 1.0) * (1.0 - tapas_sgm(mu2hat_a, 1.0)) * sa2hat_a
    inferv_r = tapas_sgm(mu2hat_r, 1.0) * (1.0 - tapas_sgm(mu2hat_r, 1.0)) * sa2hat_r
    pv_a = tapas_sgm(mu2hat_a, 1.0) * (1.0 - tapas_sgm(mu2hat_a, 1.0)) * np.exp(mu3hat_a)
    pv_r = tapas_sgm(mu2hat_r, 1.0) * (1.0 - tapas_sgm(mu2hat_r, 1.0)) * np.exp(mu3hat_r)
    wager_pred = (
        p_obs['be0']
        + p_obs['be1'] * surp
        + p_obs['be2'] * wx
        + p_obs['be3'] * inferv_a
        + p_obs['be4'] * inferv_r
        + p_obs['be5'] * pv_a
        + p_obs['be6'] * pv_r
    )

    metric_map.update({
        'arbitration_mean': wx,
        'arbitration_reward_mean': wc,
        'surprise_mean': surp,
        'integrated_belief_mean': b,
        'decision_noise_mean': decision_noise,
        'inferv_a_mean': inferv_a,
        'inferv_r_mean': inferv_r,
        'pv_a_mean': pv_a,
        'pv_r_mean': pv_r,
        'wager_pred_mean': wager_pred,
        'wager_pred_sd': np.full_like(wager_pred, np.nanstd(wager_pred, ddof=1)),
        'abs_arbitration_dev_mean': np.abs(wx - 0.5),
    })

    for name, vals in metric_map.items():
        out[name] = float(np.nanmean(vals))
    return out


def main():
    parser = argparse.ArgumentParser(description='Assess test-retest reliability of estimated latent-state summaries.')
    parser.add_argument('--fit-json-dir', required=True)
    parser.add_argument('--trials-csv', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    trials = pd.read_csv(args.trials_csv)
    fit_paths = sorted(glob.glob(os.path.join(args.fit_json_dir, 'fit_*.json')))
    if not fit_paths:
        raise RuntimeError('No fit JSONs found.')

    state_rows: List[dict] = []
    for path in fit_paths:
        fit = _load_fit_json(path)
        pid = fit['prolific_id']
        round_name = fit['round_name']
        tp = 't1' if round_name == 'round1' else ('t2' if round_name == 'round2' else round_name)
        g = trials[(trials['prolific_id'] == pid) & (trials['round_name'] == round_name)].copy()
        if g.empty:
            continue
        g = g.sort_values('trial_index_choice').reset_index(drop=True)
        u = g[['input_advice', 'input_reward', 'advice_card_space']].to_numpy(dtype=float)
        prc_vec = np.array([fit['p_prc'][k] for k in PRC_ORDER], dtype=float)
        r = {'u': u, 'ign': []}
        traj, _ = hgf_binary3l_freekappa_reward_social(r, prc_vec)
        summaries = _state_summaries(traj, u, fit['p_obs'], fit['p_prc'])
        for metric, estimate in summaries.items():
            state_rows.append({
                'prolific_id': pid,
                'round_name': round_name,
                'timepoint': tp,
                'state_metric': metric,
                'estimate': estimate,
            })

    states_long = pd.DataFrame(state_rows).sort_values(['state_metric', 'prolific_id', 'timepoint']).reset_index(drop=True)
    states_long.to_csv(os.path.join(args.out_dir, 'state_estimates_long_t1_t2.csv'), index=False)

    wide = states_long.pivot_table(
        index=['prolific_id', 'state_metric'],
        columns='timepoint',
        values='estimate',
        aggfunc='first',
    ).reset_index()
    wide.to_csv(os.path.join(args.out_dir, 'state_estimates_wide_t1_t2.csv'), index=False)

    rel_rows = []
    for metric, d in wide.groupby('state_metric'):
        if 't1' not in d.columns or 't2' not in d.columns:
            continue
        d = d[['prolific_id', 'state_metric', 't1', 't2']].dropna()
        if d.empty:
            continue
        t1 = d['t1'].to_numpy(dtype=float)
        t2 = d['t2'].to_numpy(dtype=float)
        rel_rows.append({
            'state_metric': metric,
            'n_subjects_both_timepoints': int(len(d)),
            'pearson_r_t1_t2': _corr(t1, t2),
            'icc3_1_t1_t2': _icc3_1(np.column_stack([t1, t2])),
            'mean_t1': float(np.mean(t1)),
            'mean_t2': float(np.mean(t2)),
            'sd_t1': float(np.std(t1, ddof=1)) if len(t1) > 1 else np.nan,
            'sd_t2': float(np.std(t2, ddof=1)) if len(t2) > 1 else np.nan,
            'mean_abs_delta': float(np.mean(np.abs(t2 - t1))),
        })

    rel_df = pd.DataFrame(rel_rows).sort_values('state_metric').reset_index(drop=True)
    rel_df.to_csv(os.path.join(args.out_dir, 'state_test_retest_reliability.csv'), index=False)

    aggregate = {
        'n_state_metrics': int(len(rel_df)),
        'mean_pearson_r': float(rel_df['pearson_r_t1_t2'].mean()) if len(rel_df) else np.nan,
        'mean_icc3_1': float(rel_df['icc3_1_t1_t2'].mean()) if len(rel_df) else np.nan,
        'best_pearson_metric': str(rel_df.sort_values('pearson_r_t1_t2', ascending=False).iloc[0]['state_metric']) if len(rel_df) else '',
        'best_pearson_value': float(rel_df['pearson_r_t1_t2'].max()) if len(rel_df) else np.nan,
        'best_icc_metric': str(rel_df.sort_values('icc3_1_t1_t2', ascending=False).iloc[0]['state_metric']) if len(rel_df) else '',
        'best_icc_value': float(rel_df['icc3_1_t1_t2'].max()) if len(rel_df) else np.nan,
    }
    with open(os.path.join(args.out_dir, 'state_reliability_aggregate.json'), 'w') as f:
        json.dump(aggregate, f, indent=2)

    print('Saved state-level outputs:')
    print(os.path.join(args.out_dir, 'state_estimates_long_t1_t2.csv'))
    print(os.path.join(args.out_dir, 'state_estimates_wide_t1_t2.csv'))
    print(os.path.join(args.out_dir, 'state_test_retest_reliability.csv'))
    print(os.path.join(args.out_dir, 'state_reliability_aggregate.json'))
    print('Aggregate:', json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
