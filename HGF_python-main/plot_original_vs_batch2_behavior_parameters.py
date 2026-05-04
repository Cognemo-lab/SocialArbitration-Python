from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
ORIG_BEH = BASE / "hgf_baselinefixed_full" / "construct_validity_behavior" / "behavior_session_summary.csv"
BATCH2_TRIALS = BASE / "hgf_batch2_t1_baselinefixed_full" / "extracted_model_trials.csv"
ORIG_PARAMS = BASE / "hgf_baselinefixed_full" / "parameter_estimates_long_t1_t2.csv"
BATCH2_PARAMS = BASE / "hgf_batch2_t1_baselinefixed_fullfidelity" / "parameter_estimates_long_t1_t2.csv"
OUT_DIR = BASE / "hgf_batch2_t1_baselinefixed_fullfidelity" / "comparison_to_original"

BEHAVIOR_METRICS = {
    "accuracy_rate": "Accuracy",
    "advice_taking_rate": "Advice-taking",
    "win_stay_rate": "Win-stay",
    "lose_switch_rate": "Lose-switch",
    "mean_wager": "Mean wager",
}

PARAM_LABELS = {
    "ka_a": "Kappa-Advice",
    "ka_r": "Kappa-Reward",
    "m_a": "Equilibrium-Advice",
    "om_a": "Omega-Advice",
    "sa3a_0": "Prior Uncertainty-Advice",
    "sa3r_0": "Prior Uncertainty-Reward",
    "th_a": "Theta-Advice",
    "th_r": "Theta-Reward",
    "be2": "Arbitration",
    "ze": "Social bias",
    "be_ch": "Choice noise",
    "be_wager": "Wager noise",
}
PARAM_ORDER = list(PARAM_LABELS.keys())


def behavior_summary_from_trials(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in trials.groupby("prolific_id", sort=False):
        g = g.sort_values("trial_index_choice").reset_index(drop=True).copy()
        correct = (g["choice_side"] == g["input_reward"]).astype(float)
        prev_correct = correct.shift(1)
        prev_choice = g["choice_side"].shift(1)
        stay = (g["choice_side"] == prev_choice).astype(float)
        switch = 1.0 - stay
        win_mask = prev_correct == 1
        lose_mask = prev_correct == 0
        rows.append(
            {
                "prolific_id": pid,
                "accuracy_rate": float(correct.mean()),
                "advice_taking_rate": float(g["choice_advice_taken"].mean()),
                "win_stay_rate": float(stay.loc[win_mask].mean()) if win_mask.any() else float("nan"),
                "lose_switch_rate": float(switch.loc[lose_mask].mean()) if lose_mask.any() else float("nan"),
                "mean_wager": float(g["wager"].mean()),
            }
        )
    return pd.DataFrame(rows)


def make_behavior_plot(orig: pd.DataFrame, batch2: pd.DataFrame) -> Path:
    beh_orig = orig.copy()
    beh_orig["dataset"] = "Original"
    beh_batch2 = batch2.copy()
    beh_batch2["dataset"] = "Batch 2"
    long = pd.concat([beh_orig, beh_batch2], ignore_index=True).melt(
        id_vars=["prolific_id", "dataset"],
        value_vars=list(BEHAVIOR_METRICS.keys()),
        var_name="metric",
        value_name="value",
    )
    long["label"] = long["metric"].map(BEHAVIOR_METRICS)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    for ax, metric in zip(axes, list(BEHAVIOR_METRICS.keys())):
        sub = long[long["metric"] == metric].copy()
        sns.violinplot(data=sub, x="dataset", y="value", ax=ax, inner=None, cut=0, palette=["#4C78A8", "#F58518"])
        sns.boxplot(
            data=sub,
            x="dataset",
            y="value",
            ax=ax,
            width=0.28,
            showcaps=True,
            boxprops={"facecolor": "white", "zorder": 3},
            whiskerprops={"linewidth": 1.2},
            medianprops={"color": "black", "linewidth": 1.3},
            flierprops={"marker": "", "markersize": 0},
        )
        ax.set_title(BEHAVIOR_METRICS[metric])
        ax.set_xlabel("")
        ax.set_ylabel("")
    axes[-1].axis("off")
    fig.suptitle("Behavioral Comparison: Original vs Batch 2", fontsize=16, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR / "figure_behavior_comparison_original_vs_batch2.png"
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return out


def make_parameter_plot(orig_params: pd.DataFrame, batch2_params: pd.DataFrame) -> Path:
    keep = [p for p in PARAM_ORDER if p in orig_params["parameter"].unique() and p in batch2_params["parameter"].unique()]
    o = orig_params[orig_params["parameter"].isin(keep)].copy()
    b = batch2_params[batch2_params["parameter"].isin(keep)].copy()
    o["dataset"] = "Original"
    b["dataset"] = "Batch 2"
    long = pd.concat([o, b], ignore_index=True)
    long["label"] = long["parameter"].map(PARAM_LABELS)

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.ravel()
    for ax, param in zip(axes, keep):
        sub = long[long["parameter"] == param].copy()
        sns.violinplot(data=sub, x="dataset", y="estimate", ax=ax, inner=None, cut=0, palette=["#4C78A8", "#F58518"])
        sns.boxplot(
            data=sub,
            x="dataset",
            y="estimate",
            ax=ax,
            width=0.28,
            showcaps=True,
            boxprops={"facecolor": "white", "zorder": 3},
            whiskerprops={"linewidth": 1.2},
            medianprops={"color": "black", "linewidth": 1.3},
            flierprops={"marker": "", "markersize": 0},
        )
        ax.set_title(PARAM_LABELS[param], fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes[len(keep):]:
        ax.axis("off")
    fig.suptitle("Parameter Comparison: Original vs Batch 2", fontsize=16, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR / "figure_parameter_comparison_original_vs_batch2.png"
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    orig_beh = pd.read_csv(ORIG_BEH)
    orig_beh = orig_beh[orig_beh["timepoint"] == "t1"][["prolific_id", *BEHAVIOR_METRICS.keys()]].copy()
    batch2_trials = pd.read_csv(BATCH2_TRIALS)
    batch2_beh = behavior_summary_from_trials(batch2_trials)
    behavior_fig = make_behavior_plot(orig_beh, batch2_beh)

    payload = {"behavior_figure": str(behavior_fig), "parameter_figure": None}

    if BATCH2_PARAMS.exists():
        orig_params = pd.read_csv(ORIG_PARAMS)
        orig_params = orig_params[(orig_params["timepoint"] == "t1") & (~orig_params["is_fixed"])].copy()
        batch2_params = pd.read_csv(BATCH2_PARAMS)
        batch2_params = batch2_params[(batch2_params["timepoint"] == "t1") & (~batch2_params["is_fixed"])].copy()
        param_fig = make_parameter_plot(orig_params, batch2_params)
        payload["parameter_figure"] = str(param_fig)

    (OUT_DIR / "plot_outputs.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
