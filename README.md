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
