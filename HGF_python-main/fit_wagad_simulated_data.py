import argparse
import json
import os
import numpy as np
import pandas as pd

from HGF.code_inversion.tapas_fitModel import tapas_fitModel
from HGF.code_inversion.tapas_quasinewton_optim_config import tapas_quasinewton_optim_config

from HGF.code_model_wagad.prc_model.hgf_binary3l_freekappa_reward_social_config import hgf_binary3l_freekappa_reward_social_config
from HGF.code_model_wagad.obs_model.linear_volatilitydecnoise_1stlevelprecision_reward_social_config import (
    linear_volatilitydecnoise_1stlevelprecision_reward_social_config,
)


def run_fit(sim_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(sim_csv)
    needed = ['input_advice', 'input_reward', 'advice_card_space', 'sim_choice', 'sim_wager']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns in {sim_csv}: {missing}')

    input_u = df[['input_advice', 'input_reward', 'advice_card_space']].to_numpy(dtype=float)
    y = df[['sim_wager', 'sim_choice']].to_numpy(dtype=float)

    est = tapas_fitModel(
        y,
        input_u,
        c_prc=hgf_binary3l_freekappa_reward_social_config,
        c_obs=linear_volatilitydecnoise_1stlevelprecision_reward_social_config,
        c_opt=tapas_quasinewton_optim_config,
    )

    summary = {
        'n_trials': int(len(df)),
        'LME': float(est['optim']['LME']),
        'AIC': float(est['optim']['AIC']),
        'BIC': float(est['optim']['BIC']),
        'p_prc': {k: float(v) for k, v in est['p_prc'].items() if isinstance(v, (int, float, np.floating))},
        'p_obs': {k: float(v) for k, v in est['p_obs'].items() if isinstance(v, (int, float, np.floating))},
    }

    out_json = os.path.join(out_dir, 'fit_summary.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    np.savez(
        os.path.join(out_dir, 'fit_arrays.npz'),
        optim_final=est['optim']['final'],
        optim_H=est['optim']['H'],
        optim_Sigma=est['optim']['Sigma'],
        optim_Corr=est['optim']['Corr'],
        traj_muhat_r=est['traj']['muhat_r'],
        traj_muhat_a=est['traj']['muhat_a'],
    )

    print(f"Saved summary: {out_json}")
    print(f"Saved arrays: {os.path.join(out_dir, 'fit_arrays.npz')}")
    print(f"LME={summary['LME']:.4f} AIC={summary['AIC']:.4f} BIC={summary['BIC']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Fit WAGAD model to simulated behavioral data CSV.')
    parser.add_argument('--sim-csv', default='sim_outputs/simulated_wagad_data.csv')
    parser.add_argument('--out-dir', default='fit_outputs')
    args = parser.parse_args()

    run_fit(args.sim_csv, args.out_dir)


if __name__ == '__main__':
    main()
