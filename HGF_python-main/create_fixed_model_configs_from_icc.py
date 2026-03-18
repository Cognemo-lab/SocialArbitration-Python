import argparse
import os

import pandas as pd


PRC_BASE_CLASS = 'hgf_binary3l_freekappa_reward_social_fixed_config'
OBS_BASE_CLASS = 'linear_volatilitydecnoise_1stlevelprecision_reward_social_fixed_config'


def _read_icc_table(path: str, threshold: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {'parameter', 'group'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns in ICC file: {sorted(missing)}')

    icc_col = None
    for candidate in ['icc3_1_t1_t2', 'icc', 'ICC', 'ICC3_1']:
        if candidate in df.columns:
            icc_col = candidate
            break
    if icc_col is None:
        raise ValueError('Could not find an ICC column in the provided CSV.')

    out = df[['parameter', 'group', icc_col]].copy()
    out = out.rename(columns={icc_col: 'icc'})
    out = out[pd.notnull(out['icc'])]
    out = out[out['icc'] < threshold].copy()
    return out.sort_values(['group', 'parameter']).reset_index(drop=True)


def _py_list(values):
    return '[' + ', '.join(repr(v) for v in values) + ']'


def _render_prc_module(class_name: str, fixed_parameters):
    return f"""from .hgf_binary3l_freekappa_reward_social_fixed_config import {PRC_BASE_CLASS}


class {class_name}({PRC_BASE_CLASS}):
    model = '{class_name}'
    fixed_parameters = {_py_list(fixed_parameters)}
"""


def _render_obs_module(class_name: str, fixed_parameters):
    return f"""from .linear_volatilitydecnoise_1stlevelprecision_reward_social_fixed_config import {OBS_BASE_CLASS}


class {class_name}({OBS_BASE_CLASS}):
    model = '{class_name}'
    fixed_parameters = {_py_list(fixed_parameters)}
"""


def main():
    parser = argparse.ArgumentParser(
        description='Create fixed-parameter model config modules from an ICC/reliability CSV.'
    )
    parser.add_argument('--icc-csv', required=True)
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument(
        '--prc-out',
        default='python/HGF/code_model_wagad/prc_model/hgf_binary3l_freekappa_reward_social_icc40fixed_config.py',
    )
    parser.add_argument(
        '--obs-out',
        default='python/HGF/code_model_wagad/obs_model/linear_volatilitydecnoise_1stlevelprecision_reward_social_icc40fixed_config.py',
    )
    parser.add_argument(
        '--prc-class-name',
        default='hgf_binary3l_freekappa_reward_social_icc40fixed_config',
    )
    parser.add_argument(
        '--obs-class-name',
        default='linear_volatilitydecnoise_1stlevelprecision_reward_social_icc40fixed_config',
    )
    args = parser.parse_args()

    poor_df = _read_icc_table(args.icc_csv, args.threshold)
    prc_params = poor_df.loc[poor_df['group'] == 'prc', 'parameter'].tolist()
    obs_params = poor_df.loc[poor_df['group'] == 'obs', 'parameter'].tolist()

    prc_code = _render_prc_module(args.prc_class_name, prc_params)
    obs_code = _render_obs_module(args.obs_class_name, obs_params)

    os.makedirs(os.path.dirname(args.prc_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.obs_out), exist_ok=True)

    with open(args.prc_out, 'w') as f:
        f.write(prc_code)
    with open(args.obs_out, 'w') as f:
        f.write(obs_code)

    print(f'Created perceptual config: {args.prc_out}')
    print(f'Created observation config: {args.obs_out}')
    print(f'Fixed perceptual params ({len(prc_params)}): {prc_params}')
    print(f'Fixed observation params ({len(obs_params)}): {obs_params}')


if __name__ == '__main__':
    main()
