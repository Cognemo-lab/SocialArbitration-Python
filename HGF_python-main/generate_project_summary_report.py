from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
OUT_DIR = BASE / "project_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "project_summary_report.md"


def md_table(df: pd.DataFrame, decimals: int = 3) -> str:
    work = df.copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else f"{x:.{decimals}f}")
    headers = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.itertuples(index=False, name=None)]
    return "\n".join([headers, sep] + rows)


def main() -> None:
    ppc = pd.read_csv(BASE / "hgf_batch2_t1_baselinefixed_fullfidelity" / "comparison_to_original" / "aggregate_metric_comparison.csv")
    ppc_sel = ppc.loc[ppc["metric"].isin(["choice_auc", "choice_brier", "wager_corr", "wager_r2"])][
        ["metric", "original_dataset", "batch2_dataset", "delta_batch2_minus_original"]
    ]

    beh_rel = pd.read_csv(BASE / "behavior_test_retest_comparison" / "behavior_test_retest_reliability_comparison.csv")
    beh_rel_sum = (
        beh_rel.groupby("dataset")
        .agg(mean_pearson_r=("pearson_r", "mean"), mean_icc=("icc3_1", "mean"))
        .reset_index()
    )

    param_compare = pd.read_csv(BASE / "reliability_comparison_behavior_parameters" / "behavior_parameter_reliability_comparison.csv")
    param_sum = (
        param_compare.groupby(["dataset", "domain"])
        .agg(mean_pearson_r=("pearson_r_t1_t2", "mean"), mean_icc=("icc3_1_t1_t2", "mean"))
        .reset_index()
    )

    seq_sens = pd.read_csv(BASE / "parameter_reliability_sequence_sensitivity" / "parameter_reliability_sequence_sensitivity_summary.csv")
    seq_sens = seq_sens[["dataset", "subset", "group", "mean_pearson_r", "mean_icc3_1", "mean_abs_delta_vs_full_r"]]

    beh_state = pd.read_csv(BASE / "behavior_test_retest_comparison" / "state_dependence_redeployment" / "behavior_reliability_state_adjustment_summary.csv")
    beh_state_sel = beh_state[["label", "raw_pearson_r", "adjusted_pearson_r", "raw_icc3_1", "adjusted_icc3_1"]]

    param_state = pd.read_csv(BASE / "hgf_batch2_t2_parameter_reliability" / "state_dependence_redeployment" / "parameter_reliability_state_adjustment_headline.csv")

    suic_floor = pd.read_csv(BASE / "behavior_fod_moderation_comparison" / "state_and_prevalence_assessment" / "suicidality_distribution_summary.csv")
    suic_floor = suic_floor[["dataset", "count", "mean", "std", "prop_floor_4"]]

    acc_mod = pd.read_csv(BASE / "behavior_fod_moderation_comparison" / "accuracy_fod_moderation_summary.csv")
    acc_mod = acc_mod[acc_mod["scope"].isin(["pooled", "t1", "t2"])][["dataset", "scope", "interaction_beta", "interaction_p"]]

    fod_mod = pd.read_csv(BASE / "hitop_batch2" / "fod_moderation_redeployment_matched_hitop" / "moderation_summary.csv")
    fod_mod = fod_mod[fod_mod["predictor"].isin(["om_a", "inferv_a_mean", "abs_eps2_a_mean", "pc1_composite"])]
    fod_mod = fod_mod[fod_mod["scope"].isin(["pooled", "t1", "t2"])][["scope", "predictor", "interaction_beta", "interaction_p"]]

    text = f"""# Project Report: Original vs Redeployed Jungle Quest Modeling

## Executive Summary

This report summarizes the main computational, reliability, and psychopathology findings from the original and redeployed Jungle Quest datasets. The broad pattern was:

- the HGF model generalized reasonably well across deployments;
- posterior predictive checks remained stronger for wager behavior than for choice behavior;
- recovery was strongest for observation parameters and more modest for perceptual parameters;
- behavioral measures were generally as reliable as, or more reliable than, most parameter estimates;
- the original fearlessness-of-death by suicidality moderation effects did not reproduce cleanly in the redeployed sample;
- the redeployed sample showed substantial floor compression in suicidality and related symptom domains;
- pre-task questionnaire variables explained only a small amount of variance and did not materially improve reliability.

## 1. Model Fit, PPC, and Recovery

The model fit the redeployed data with the same general profile as the original dataset. Choice prediction weakened slightly, whereas wager prediction improved slightly.

{md_table(ppc_sel, decimals=3)}

![Recovery and retest comparison]({BASE / "hgf_batch2_t2_parameter_reliability" / "figure_parameter_recovery_and_retest_comparison_combined.png"})

## 2. Test-Retest Reliability

Behavioral measures were generally more reliable than perceptual parameters and closer to the better observation parameters.

{md_table(beh_rel_sum, decimals=3)}

{md_table(param_sum, decimals=3)}

![Behavior vs parameter reliability]({BASE / "reliability_comparison_behavior_parameters" / "figure_behavior_parameter_reliability_comparison.png"})

## 3. Sequence Comparability Sensitivity

We confirmed that the dominant underlying stimulus sequence was the same across original and redeployed datasets. A stricter sensitivity analysis using only matched 100-trial sessions and the canonical schedule did not materially change the main observation-parameter reliability conclusions, although redeployed perceptual summaries became noisier because the matched subset dropped to `n=43`.

{md_table(seq_sens, decimals=3)}

![Sequence sensitivity]({BASE / "parameter_reliability_sequence_sensitivity" / "figure_parameter_reliability_sequence_sensitivity.png"})

## 4. Psychopathology and Moderation Results

In the original dataset, suicidality moderated the relationship between fearlessness of death and several advice-related computational quantities. Those effects collapsed toward zero in the redeployed sample.

Behavioral accuracy moderation:

{md_table(acc_mod, decimals=3)}

Redeployed computational moderation:

{md_table(fod_mod, decimals=3)}

![Original vs redeployed FoD moderation]({BASE / "hitop_batch2" / "fod_moderation_redeployment_matched_hitop" / "figure_original_vs_redeployment_fod_moderation.png"})

## 5. Distributional Explanation: Floor Effects

The strongest explanation for the failed psychopathology replication was symptom-range restriction in the redeployed sample, especially for suicidality.

{md_table(suic_floor, decimals=3)}

![Suicidality floor effects]({BASE / "behavior_fod_moderation_comparison" / "state_and_prevalence_assessment" / "figure_suicidality_floor_effects_original_vs_redeployment.png"})

The CFA bifactor analysis suggested that this was not a broad collapse in general psychopathology, distress-specific, or antagonism-specific spectra. The more selective compression appeared in the symptom domains most relevant to the original effects.

![HiTOP bifactor CFA]({BASE / "behavior_fod_moderation_comparison" / "state_and_prevalence_assessment" / "hitop_bifactor_cfa" / "figure_hitop_bifactor_cfa_original_vs_redeployment.png"})

## 6. Pre-Task Questionnaire: T1/T2 Profile

The redeployment pre-task questionnaire was fairly stable across T1 and T2 and covered sleep, subjective rest, motivation, and recent substance/medication use.

![Redeployment pre-task summary]({BASE / "behavior_test_retest_comparison" / "state_dependence_redeployment" / "figure_redeployment_pretask_summary_t1_t2.png"})

## 7. Do Pre-Task State Variables Improve Reliability?

For behavior, the answer was no. Adjusting for the pre-task questionnaire slightly reduced reliability for all behavioral measures.

{md_table(beh_state_sel, decimals=3)}

![Behavior state adjustment]({BASE / "behavior_test_retest_comparison" / "state_dependence_redeployment" / "figure_behavior_reliability_state_adjustment.png"})

For parameters, the same general conclusion held. Pre-task state variables explained only a small amount of variance on average and did not meaningfully improve reliability.

{md_table(param_state, decimals=3)}

![Parameter state adjustment]({BASE / "hgf_batch2_t2_parameter_reliability" / "state_dependence_redeployment" / "figure_parameter_reliability_state_adjustment.png"})

## 8. Bottom Line

The core computational model transported reasonably well to the redeployment sample. PPC and recovery remained broadly intact, and the latent behavioral and computational structure of the task was reproducible. What did not generalize well were the symptom-linked moderation and association effects. The best-supported explanation is selective range restriction in the redeployed sample, especially for suicidality, reality distortion, and delusion-related measures, rather than a failure of the computational model itself.
"""

    OUT_PATH.write_text(text)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
