# SocialArbitration-Python

This workspace contains the Python HGF inversion code and raw behavioral data used to fit, assess, and visualize the final selected reward-social model for the jungle task.

## Final model used

The final model selected in this workspace is:

- Perceptual model: `hgf_binary3l_freekappa_reward_social_baselinefixed`
- Observation model: `linear_volatilitydecnoise_1stlevelprecision_reward_social_baselinefixed`

Config classes:

- [hgf_binary3l_freekappa_reward_social_baselinefixed_config.py](/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/HGF_python-main/python/HGF/code_model_wagad/prc_model/hgf_binary3l_freekappa_reward_social_baselinefixed_config.py)
- [linear_volatilitydecnoise_1stlevelprecision_reward_social_baselinefixed_config.py](/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/HGF_python-main/python/HGF/code_model_wagad/obs_model/linear_volatilitydecnoise_1stlevelprecision_reward_social_baselinefixed_config.py)

This model fixes the following perceptual parameters:

- `mu2r_0`
- `mu2a_0`
- `mu3a_0`
- `om_r`
- `sa2a_0`

and leaves `zeta` (`ze`) free.

This was the best-performing tested configuration for keeping `zeta` recovery acceptable while improving overall recovery stability.

## Data

Raw behavioral files:

- [jungle-task.round1.csv](/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/raw_data/jungle-task.round1.csv)
- [jungle-task.round2.csv](/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/raw_data/jungle-task.round2.csv)

`round1` is treated as `t1` and `round2` as `t2` for test-retest reliability.

## Environment

The commands below were run with:

- Python: `/Users/drea/opt/anaconda3/bin/python`
- `PYTHONPATH=/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/HGF_python-main/python`

From the project root:

```bash
cd /Users/drea/Documents/CAMH/Projects/McGill-Collaboration/HGF_python-main
export PYTHONPATH=/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/HGF_python-main/python
```

## Using the model without raw data

If you do not have the original jsPsych raw exports, you can still use the final model in three practical ways.

### Option 1: Fit from a pre-extracted trial table

The model does not fundamentally require the raw CSVs. It requires trial-level `u` and `y` arrays in the format used by the inversion code.

Required trial columns:

- `prolific_id`
- `round_name`
- `input_advice`
- `input_reward`
- `advice_card_space`
- `choice_advice_taken`
- `wager`
- `trial_index_choice`

Interpretation:

- `u[:, 0] = input_advice`
  - advice correctness / helpfulness coded as `1` or `0`
- `u[:, 1] = input_reward`
  - reward side coded as `1` or `0`
- `u[:, 2] = advice_card_space`
  - advice side coded as `1` or `0`
- `y[:, 0] = wager`
  - continuous wager response
- `y[:, 1] = choice_advice_taken`
  - binary advice-following choice coded as `1` or `0`

If you already have such a table, you can fit one dataset directly in Python:

```python
import numpy as np
import pandas as pd

from HGF.code_inversion.tapas_fitModel import tapas_fitModel
from HGF.code_inversion.tapas_quasinewton_optim_config import tapas_quasinewton_optim_config
from HGF.code_model_wagad.prc_model.hgf_binary3l_freekappa_reward_social_baselinefixed_config import (
    hgf_binary3l_freekappa_reward_social_baselinefixed_config,
)
from HGF.code_model_wagad.obs_model.linear_volatilitydecnoise_1stlevelprecision_reward_social_baselinefixed_config import (
    linear_volatilitydecnoise_1stlevelprecision_reward_social_baselinefixed_config,
)

df = pd.read_csv("your_extracted_trials.csv")
df = df.sort_values("trial_index_choice").reset_index(drop=True)

u = df[["input_advice", "input_reward", "advice_card_space"]].to_numpy(dtype=float)
y = df[["wager", "choice_advice_taken"]].to_numpy(dtype=float)

class local_opt(tapas_quasinewton_optim_config):
    def __init__(self):
        super().__init__()
        self.maxIter = 5
        self.verbose = False
        self.nRandInit = 0

est = tapas_fitModel(
    y,
    u,
    c_prc=hgf_binary3l_freekappa_reward_social_baselinefixed_config,
    c_obs=linear_volatilitydecnoise_1stlevelprecision_reward_social_baselinefixed_config,
    c_opt=local_opt,
)
```

You can then read:

- `est["p_prc"]`
- `est["p_obs"]`
- `est["optim"]["LME"]`
- `est["optim"]["AIC"]`
- `est["optim"]["BIC"]`

### Option 2: Run recovery from saved fit JSONs

If you already have fitted parameter JSON files and only want parameter recovery, you do not need the original raw jsPsych files.

Use [recover_from_saved_fits.py](/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/HGF_python-main/recover_from_saved_fits.py).

What you need:

- a directory of fit JSON files with
  - `prolific_id`
  - `round_name`
  - `p_prc`
  - `p_obs`
- a trial table or equivalent raw-derived extraction that lets you reconstruct `u`

Run:

```bash
/Users/drea/opt/anaconda3/bin/python recover_from_saved_fits.py \
  --fit-json-dir /path/to/fit_jsons \
  --raw-files /path/to/round1.csv /path/to/round2.csv \
  --out-dir /path/to/recovery_outputs \
  --max-fits 10 \
  --max-iter 5 \
  --n-jobs 8
```

If you do not have the original raw CSVs but you do have an extracted trial table, the simplest path is to adapt the same logic and construct `u` directly as in Option 1.

### Option 3: Use the model only as a simulator or scoring function

If you want to simulate behavior from known parameters, you can call the perceptual and observation model functions directly.

Simulation requires:

- `u` with columns
  - `input_advice`
  - `input_reward`
  - `advice_card_space`
- perceptual parameters in native space ordered as:
  - `mu2r_0, sa2r_0, mu3r_0, sa3r_0, ka_r, om_r, th_r, mu2a_0, sa2a_0, mu3a_0, sa3a_0, ka_a, om_a, th_a, phi_r, m_r, phi_a, m_a`
- observation parameters in native space ordered as:
  - `be0, be1, be2, be3, be4, be5, be6, ze, be_ch, be_wager`

Core functions:

- [hgf_binary3l_freekappa_reward_social.py](/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/HGF_python-main/python/HGF/code_model_wagad/prc_model/hgf_binary3l_freekappa_reward_social.py)
- [linear_volatilitydecnoise_1stlevelprecision_reward_social.py](/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/HGF_python-main/python/HGF/code_model_wagad/obs_model/linear_volatilitydecnoise_1stlevelprecision_reward_social.py)
- [linear_volatilitydecnoise_1stlevelprecision_reward_social_sim.py](/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/HGF_python-main/python/HGF/code_model_wagad/obs_model/linear_volatilitydecnoise_1stlevelprecision_reward_social_sim.py)

Minimal simulation example:

```python
import numpy as np

from HGF.code_model_wagad.prc_model.hgf_binary3l_freekappa_reward_social import (
    hgf_binary3l_freekappa_reward_social,
)
from HGF.code_model_wagad.obs_model.linear_volatilitydecnoise_1stlevelprecision_reward_social_sim import (
    linear_volatilitydecnoise_1stlevelprecision_reward_social_sim,
)

u = np.array([
    [1, 1, 1],
    [0, 1, 0],
    [1, 0, 0],
], dtype=float)

prc = np.array([
    0.0, 1.0, 1.0, 1.0, 0.5, -4.0, 0.56,
    0.0, 1.0, 1.0, 1.0, 0.5, -4.0, 0.56,
    0.1, -2.1972245773, 0.74, 0.0,
], dtype=float)

obs = np.array([
    6.2, -0.5, 0.5, -0.3, -0.2, -1.0, 0.05, 2.0, 6.0, 7.0
], dtype=float)

r = {"u": u, "ign": []}
traj, inf_states = hgf_binary3l_freekappa_reward_social(r, prc)
y_sim, p_choice = linear_volatilitydecnoise_1stlevelprecision_reward_social_sim(r, inf_states, obs)
```

### Recommended workflow without raw data

If the original raw exports are unavailable, the recommended order is:

1. reconstruct or export a clean trial table in the extracted format above
2. fit the final baseline-fixed model from `u` and `y`
3. save one JSON per subject-session with `p_prc`, `p_obs`, and fit metrics
4. run simulation-based recovery from those saved fits
5. if both timepoints are available, assemble a wide `t1`/`t2` parameter table and compute Pearson `r` and ICC

### Important caveat

Without the raw data, you cannot use the repository’s automatic raw-to-trial extraction. That means the critical step is getting the trial coding exactly right, especially:

- binary side coding
- advice correctness coding
- choice-as-advice-taking coding
- ordering of `y = [wager, choice]`
- ordering of `u = [input_advice, input_reward, advice_card_space]`

If those mappings are wrong, the inversion will still run, but the estimates will not match the intended model.

## Full inversion, recovery, and reliability

Use [fit_recover_reliability_raw.py](/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/HGF_python-main/fit_recover_reliability_raw.py).

This script:

- extracts model trials from the raw CSVs
- fits the final model to each subject-session
- simulates behavior from the fitted parameters and re-estimates them for parameter recovery
- computes test-retest reliability across `t1` and `t2`
- marks fixed parameters and excludes them from aggregate recovery and reliability metrics

Run:

```bash
/Users/drea/opt/anaconda3/bin/python fit_recover_reliability_raw.py \
  --raw-files \
    /Users/drea/Documents/CAMH/Projects/McGill-Collaboration/raw_data/jungle-task.round1.csv \
    /Users/drea/Documents/CAMH/Projects/McGill-Collaboration/raw_data/jungle-task.round2.csv \
  --out-dir /Users/drea/Documents/CAMH/Projects/McGill-Collaboration/hgf_baselinefixed_full \
  --max-iter 5 \
  --n-jobs 8 \
  --seed 20260321 \
  --prc-config HGF.code_model_wagad.prc_model.hgf_binary3l_freekappa_reward_social_baselinefixed_config.hgf_binary3l_freekappa_reward_social_baselinefixed_config \
  --obs-config HGF.code_model_wagad.obs_model.linear_volatilitydecnoise_1stlevelprecision_reward_social_baselinefixed_config.linear_volatilitydecnoise_1stlevelprecision_reward_social_baselinefixed_config
```

## Main outputs

Full output directory:

- [hgf_baselinefixed_full](/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/hgf_baselinefixed_full)

Key files:

- [aggregate_metrics.json](/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/hgf_baselinefixed_full/aggregate_metrics.json)
- [fit_recovery_summary.csv](/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/hgf_baselinefixed_full/fit_recovery_summary.csv)
- [parameter_recovery_report.csv](/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/hgf_baselinefixed_full/parameter_recovery_report.csv)
- [test_retest_reliability_by_parameter.csv](/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/hgf_baselinefixed_full/test_retest_reliability_by_parameter.csv)
- [parameter_estimates_long_t1_t2.csv](/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/hgf_baselinefixed_full/parameter_estimates_long_t1_t2.csv)
- [parameter_estimates_wide_t1_t2.csv](/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/hgf_baselinefixed_full/parameter_estimates_wide_t1_t2.csv)
- [parameter_correspondence.csv](/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/hgf_baselinefixed_full/parameter_correspondence.csv)

## Plotting recovery

Use [plot_recovery_outputs.py](/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/HGF_python-main/plot_recovery_outputs.py).

Run:

```bash
/Users/drea/opt/anaconda3/bin/python plot_recovery_outputs.py \
  --recovery-out-dir /Users/drea/Documents/CAMH/Projects/McGill-Collaboration/hgf_baselinefixed_full \
  --fig-out-dir /Users/drea/Documents/CAMH/Projects/McGill-Collaboration/hgf_baselinefixed_full/figures_recovery
```

Generated figures include:

- overall recovery summary
- perceptual recovery scatterplots
- observation recovery scatterplots
- parameter-wise recovery Pearson `r`
- parameter-wise mean absolute recovery error
- `log(zeta)` recovery scatterplot

## Plotting test-retest reliability

Use [plot_test_retest_outputs.py](/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/HGF_python-main/plot_test_retest_outputs.py).

Run:

```bash
/Users/drea/opt/anaconda3/bin/python plot_test_retest_outputs.py \
  --reliability-csv /Users/drea/Documents/CAMH/Projects/McGill-Collaboration/hgf_baselinefixed_full/test_retest_reliability_by_parameter.csv \
  --wide-csv /Users/drea/Documents/CAMH/Projects/McGill-Collaboration/hgf_baselinefixed_full/parameter_estimates_wide_t1_t2.csv \
  --fig-out-dir /Users/drea/Documents/CAMH/Projects/McGill-Collaboration/hgf_baselinefixed_full/figures_test_retest
```

Generated figures include:

- parameter-wise test-retest Pearson `r`
- parameter-wise ICC(3,1)
- parameter-wise mean absolute T1-T2 change
- summary mean(T1) vs mean(T2)
- faceted perceptual T1-vs-T2 scatterplots
- faceted observation T1-vs-T2 scatterplots
- `log(zeta)` T1-vs-T2 scatterplot

## Final summary

From the final run in [hgf_baselinefixed_full](/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/hgf_baselinefixed_full):

- `439/439` subject-sessions fit successfully
- `246` unique participants
- `193` participants had both `t1` and `t2`
- mean recovery Pearson `r`
  - perceptual: `0.991`
  - observation: `0.966`
  - all free parameters: `0.976`
- mean test-retest reliability
  - perceptual Pearson `r`: `0.323`
  - observation Pearson `r`: `0.332`
  - perceptual ICC(3,1): `0.136`
  - observation ICC(3,1): `0.330`
- `zeta`
  - recovery Pearson `r`: `0.595`
  - test-retest Pearson `r`: `0.273`
  - test-retest ICC(3,1): `0.268`
