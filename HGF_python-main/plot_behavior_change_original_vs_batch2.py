from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
ORIG_TRIALS = BASE / "hgf_baselinefixed_full" / "extracted_model_trials.csv"
BATCH2_TRIALS = BASE / "hgf_batch2_t1_baselinefixed_fullfidelity" / "extracted_model_trials.csv"
OUT_DIR = BASE / "hgf_batch2_t1_baselinefixed_fullfidelity" / "comparison_to_original"

RATE_METRICS = [
    ("accuracy_rate", "Accuracy"),
    ("accuracy_first30", "Accuracy (first 30)"),
    ("accuracy_last20", "Accuracy (last 20)"),
    ("advice_taking_rate", "Advice-taking"),
    ("win_stay_rate", "Win-stay"),
    ("lose_switch_rate", "Lose-switch"),
]
WAGER_METRICS = [("mean_wager", "Mean wager")]

COLORS = {"Original": "#4C78A8", "Redeployment": "#F58518"}


def subject_behavior_summary(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, group in trials.groupby("prolific_id", sort=False):
        g = group.sort_values("trial_index_choice").reset_index(drop=True).copy()
        correct = (g["choice_side"] == g["input_reward"]).astype(float)
        prev_correct = correct.shift(1)
        prev_choice = g["choice_side"].shift(1)
        stay = (g["choice_side"] == prev_choice).astype(float)
        switch = 1.0 - stay
        win_mask = prev_correct == 1
        lose_mask = prev_correct == 0

        first30 = correct.iloc[:30]
        last20 = correct.iloc[-20:]
        rows.append(
            {
                "prolific_id": pid,
                "accuracy_rate": float(correct.mean()),
                "accuracy_first30": float(first30.mean()) if len(first30) else np.nan,
                "accuracy_last20": float(last20.mean()) if len(last20) else np.nan,
                "advice_taking_rate": float(g["choice_advice_taken"].mean()),
                "win_stay_rate": float(stay.loc[win_mask].mean()) if win_mask.any() else np.nan,
                "lose_switch_rate": float(switch.loc[lose_mask].mean()) if lose_mask.any() else np.nan,
                "mean_wager": float(g["wager"].mean()),
            }
        )
    return pd.DataFrame(rows)


def mean_ci(series: pd.Series) -> Tuple[float, float, float]:
    values = series.dropna().to_numpy(dtype=float)
    mean = float(np.mean(values))
    if len(values) <= 1:
        return mean, mean, mean
    sem = stats.sem(values, nan_policy="omit")
    low, high = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=sem)
    return mean, float(low), float(high)


def build_summary(orig: pd.DataFrame, batch2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, label in RATE_METRICS + WAGER_METRICS:
        o = orig[metric]
        b = batch2[metric]
        o_mean, o_low, o_high = mean_ci(o)
        b_mean, b_low, b_high = mean_ci(b)
        t_stat, p_value = stats.ttest_ind(o.dropna(), b.dropna(), equal_var=False)
        rows.append(
            {
                "metric": metric,
                "label": label,
                "original_mean": o_mean,
                "original_ci_low": o_low,
                "original_ci_high": o_high,
                "redeployment_mean": b_mean,
                "redeployment_ci_low": b_low,
                "redeployment_ci_high": b_high,
                "delta_redeployment_minus_original": b_mean - o_mean,
                "welch_t": float(t_stat),
                "p_value": float(p_value),
            }
        )
    return pd.DataFrame(rows)


def add_panel(ax: plt.Axes, summary: pd.DataFrame, metrics: List[Tuple[str, str]], xlim: Tuple[float, float], title: str) -> None:
    subset = summary.set_index("metric").loc[[m for m, _ in metrics]].reset_index()
    labels = [label for _, label in metrics]
    y = np.arange(len(labels))[::-1]

    ax.axvline(0 if xlim[0] < 0 else 0.5, color="#999999", linestyle="--", linewidth=1, alpha=0)

    for yi, (_, row) in zip(y, subset.iterrows()):
        ax.plot(
            [row["original_mean"], row["redeployment_mean"]],
            [yi, yi],
            color="#B0B0B0",
            linewidth=1.5,
            zorder=1,
        )
        ax.errorbar(
            row["original_mean"],
            yi,
            xerr=[[row["original_mean"] - row["original_ci_low"]], [row["original_ci_high"] - row["original_mean"]]],
            fmt="o",
            color=COLORS["Original"],
            ecolor=COLORS["Original"],
            elinewidth=2,
            capsize=4,
            markersize=7,
            zorder=3,
        )
        ax.errorbar(
            row["redeployment_mean"],
            yi,
            xerr=[[row["redeployment_mean"] - row["redeployment_ci_low"]], [row["redeployment_ci_high"] - row["redeployment_mean"]]],
            fmt="o",
            color=COLORS["Redeployment"],
            ecolor=COLORS["Redeployment"],
            elinewidth=2,
            capsize=4,
            markersize=7,
            zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(*xlim)
    ax.set_title(title, fontsize=13, pad=12)
    ax.grid(axis="x", alpha=0.25)


def make_figure(summary: pd.DataFrame, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 7.5),
        gridspec_kw={"width_ratios": [4.5, 1.8]},
    )

    add_panel(ax1, summary, RATE_METRICS, (0.0, 1.0), "Behavioral rates")
    add_panel(ax2, summary, WAGER_METRICS, (5.5, 8.5), "Wager")

    ax1.set_xlabel("Session mean", fontsize=12)
    ax2.set_xlabel("Session mean", fontsize=12)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["Original"], markeredgecolor=COLORS["Original"], markersize=8, label="Original"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["Redeployment"], markeredgecolor=COLORS["Redeployment"], markersize=8, label="Redeployment"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.02))

    fig.suptitle("Behavioral Change Comparison: Original vs Redeployment", fontsize=18, y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95), w_pad=2.5)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    orig_trials = pd.read_csv(ORIG_TRIALS)
    orig_trials = orig_trials.loc[orig_trials["round_name"] == "round1"].copy()
    batch2_trials = pd.read_csv(BATCH2_TRIALS)

    orig_summary = subject_behavior_summary(orig_trials)
    batch2_summary = subject_behavior_summary(batch2_trials)
    summary = build_summary(orig_summary, batch2_summary)

    summary_path = OUT_DIR / "behavior_change_comparison_original_vs_batch2.csv"
    figure_path = OUT_DIR / "figure_behavior_change_comparison_original_vs_batch2.png"

    summary.to_csv(summary_path, index=False)
    make_figure(summary, figure_path)

    print(f"Wrote {summary_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
