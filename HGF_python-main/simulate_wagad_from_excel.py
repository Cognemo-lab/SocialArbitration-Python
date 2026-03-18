import argparse
import os
import numpy as np
import pandas as pd
from scipy.io import savemat

from HGF.code_model_wagad.prc_model.hgf_binary3l_freekappa_reward_social import hgf_binary3l_freekappa_reward_social
from HGF.code_model_wagad.obs_model.linear_volatilitydecnoise_1stlevelprecision_reward_social_sim import linear_volatilitydecnoise_1stlevelprecision_reward_social_sim
from HGF.code_model_wagad.obs_model.linear_volatilitydecnoise_1stlevelprecision_reward_social_namep import linear_volatilitydecnoise_1stlevelprecision_reward_social_namep
from HGF.code_model_wagad.obs_model.linear_volatilitydecnoise_1stlevelprecision_reward_social_config import linear_volatilitydecnoise_1stlevelprecision_reward_social_config
from HGF.code_model_wagad.predicted_wager import calculate_predicted_wager


def _numeric_col(df, i):
    return pd.to_numeric(df.iloc[:, i], errors='coerce').to_numpy(dtype=float)


def _get_default_p_obs(obs_source):
    if obs_source == 'legacy_fitted':
        return np.array([6.6606, -0.5679, 0.5793, -0.3872, -0.1581, -1.7403, 0.0164, 2.0, 3.3213, 7.6193], dtype=float)

    if obs_source == 'config_priors':
        obs_cfg = linear_volatilitydecnoise_1stlevelprecision_reward_social_config()
        return obs_cfg.transp_obs_fun(None, obs_cfg.priormus)[0].reshape(-1)

    raise ValueError("obs_source must be one of: 'legacy_fitted', 'config_priors'")


def _plot_simulation(out_df, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not installed; skipping plots.')
        return []

    trials = out_df['trial'].to_numpy()
    files = []

    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax[0].plot(trials, out_df['choice_prob'], color='tab:blue', lw=2, label='choice_prob')
    ax[0].scatter(trials, out_df['sim_choice'], color='tab:orange', s=14, alpha=0.75, label='sim_choice')
    ax[0].set_ylabel('Choice')
    ax[0].set_title('Simulated Choice Dynamics')
    ax[0].legend(loc='upper right')
    ax[0].set_ylim(-0.05, 1.05)

    ax[1].plot(trials, out_df['sim_wager'], color='tab:green', lw=1.8, label='sim_wager')
    ax[1].plot(trials, out_df['predicted_wager_z'], color='tab:red', lw=1.8, label='predicted_wager_z')
    ax[1].set_ylabel('Wager')
    ax[1].set_title('Simulated vs Predicted Wager')
    ax[1].legend(loc='upper right')

    ax[2].plot(trials, out_df['input_advice'], color='tab:purple', lw=1.6, label='input_advice')
    ax[2].plot(trials, out_df['input_reward'], color='tab:brown', lw=1.6, label='input_reward')
    if 'probability_advice' in out_df.columns:
        ax[2].plot(trials, out_df['probability_advice'], color='tab:cyan', lw=1.2, alpha=0.9, label='probability_advice')
    if 'probability_card' in out_df.columns:
        ax[2].plot(trials, out_df['probability_card'], color='tab:gray', lw=1.2, alpha=0.9, label='probability_card')
    ax[2].set_xlabel('Trial')
    ax[2].set_ylabel('Input / Probability')
    ax[2].set_title('Input Streams')
    ax[2].legend(loc='upper right')

    plt.tight_layout()
    ts_file = os.path.join(output_dir, 'simulated_wagad_timeseries.png')
    fig.savefig(ts_file, dpi=180, bbox_inches='tight')
    plt.close(fig)
    files.append(ts_file)

    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4))
    ax2[0].hist(out_df['sim_wager'], bins=20, color='tab:green', alpha=0.8)
    ax2[0].set_title('Distribution of Simulated Wager')
    ax2[0].set_xlabel('sim_wager')
    ax2[0].set_ylabel('count')

    ax2[1].hist(out_df['choice_prob'], bins=20, color='tab:blue', alpha=0.8)
    ax2[1].set_title('Distribution of Choice Probability')
    ax2[1].set_xlabel('choice_prob')
    ax2[1].set_ylabel('count')

    plt.tight_layout()
    dist_file = os.path.join(output_dir, 'simulated_wagad_distributions.png')
    fig2.savefig(dist_file, dpi=180, bbox_inches='tight')
    plt.close(fig2)
    files.append(dist_file)

    return files


def simulate(input_xlsx, sheet, output_dir, n_sims=1, seed=20260303, obs_source='legacy_fitted'):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_excel(input_xlsx, sheet_name=sheet)
    inputs_advice = _numeric_col(df, 0)
    inputs_reward = _numeric_col(df, 1)
    advice_card = _numeric_col(df, 2)

    if np.isnan(inputs_advice).any() or np.isnan(inputs_reward).any() or np.isnan(advice_card).any():
        raise ValueError('NaNs detected in the first 3 columns of the input sheet after numeric conversion.')

    input_u = np.column_stack([inputs_advice, inputs_reward, advice_card])

    p_prc = np.array([
        0, 1, 1, 1, 0.58458, -4, 0.59936,
        0, 1, 1, 1, 0.61243, -4, 0.66502,
        0.1000, 1, 0.1000, 1
    ], dtype=float)

    # Observation parameter source:
    # - legacy_fitted: reproduces previous non-flat simulation behavior
    # - config_priors: uses means from linear_volatilitydecnoise_1stlevelprecision_reward_social_config.py
    p_obs = _get_default_p_obs(obs_source)

    np.random.seed(seed)
    print(f'Using observation parameter source: {obs_source}')
    outputs = []

    for _ in range(n_sims):
        r = {'u': input_u, 'ign': []}
        traj, inf_states = hgf_binary3l_freekappa_reward_social(r, p_prc)
        # Updated sim returns (y, prob) where y is [wager, choice] —
        # choice is already sampled internally.
        y_sim, choice_prob = linear_volatilitydecnoise_1stlevelprecision_reward_social_sim(r, inf_states, p_obs)
        sim_wager = y_sim[:, 0]
        sim_choice = y_sim[:, 1].astype(int)

        est_like = {
            'traj': traj,
            'u': input_u,
            'p_obs': linear_volatilitydecnoise_1stlevelprecision_reward_social_namep(p_obs),
        }
        predicted_wager_z = calculate_predicted_wager(est_like)

        out = pd.DataFrame({
            'trial': np.arange(1, len(inputs_advice) + 1),
            'input_advice': inputs_advice,
            'input_reward': inputs_reward,
            'advice_card_space': advice_card,
            'choice_prob': choice_prob,
            'sim_choice': sim_choice,
            'sim_wager': sim_wager,
            'predicted_wager_z': predicted_wager_z,
        })
        if df.shape[1] >= 5:
            out['probability_advice'] = _numeric_col(df, 3)
            out['probability_card'] = _numeric_col(df, 4)

        outputs.append(out)

    csv_file = os.path.join(output_dir, 'simulated_wagad_data.csv')
    mat_file = os.path.join(output_dir, 'simulated_wagad_data.mat')
    outputs[0].to_csv(csv_file, index=False)

    savemat(mat_file, {
        'input_u': input_u,
        'sim_choice': outputs[0]['sim_choice'].to_numpy(dtype=float).reshape(-1, 1),
        'sim_wager': outputs[0]['sim_wager'].to_numpy(dtype=float).reshape(-1, 1),
        'choice_prob': outputs[0]['choice_prob'].to_numpy(dtype=float).reshape(-1, 1),
        'predicted_wager_z': outputs[0]['predicted_wager_z'].to_numpy(dtype=float).reshape(-1, 1),
    })

    plot_files = _plot_simulation(outputs[0], output_dir)

    print(f'Saved CSV: {csv_file}')
    print(f'Saved MAT: {mat_file}')
    for pf in plot_files:
        print(f'Saved plot: {pf}')


def main():
    parser = argparse.ArgumentParser(description='Simulate WAGAD data from Excel inputs using Python ports of MATLAB models.')
    parser.add_argument('--input-xlsx', default='/Users/drea/Dropbox/Andreiuta/fMRI_IOIO/WAGAD_paper1/Paper/Revision/Inputs2020/final_inputs_advice_reward_withprobabilities.xlsx')
    parser.add_argument('--sheet', default='final_inputs_advice_reward')
    parser.add_argument('--output-dir', default='sim_outputs')
    parser.add_argument('--n-sims', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20260303)
    parser.add_argument('--obs-source', choices=['legacy_fitted', 'config_priors'], default='legacy_fitted')
    args = parser.parse_args()

    simulate(args.input_xlsx, args.sheet, args.output_dir, n_sims=args.n_sims, seed=args.seed, obs_source=args.obs_source)


if __name__ == '__main__':
    main()
